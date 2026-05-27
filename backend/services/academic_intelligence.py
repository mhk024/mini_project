"""
academic_intelligence.py
========================
Core intelligence layer that combines Semantic Scholar + OpenAlex data
with the uploaded paper to produce actionable research insights.

Functions:
  - extract_paper_metadata(text)           : extract title, abstract, keywords
  - compute_enhanced_quality(metadata, ss_papers, oa_works) : quality score 0-10
  - find_missing_citations(metadata, external_papers)       : citation gap analysis
  - compute_novelty_score(metadata, similar_papers)         : novelty 0-10
  - generate_research_suggestions(metadata, all_data)       : actionable suggestions
  - run_full_academic_analysis(document_text)               : orchestrates everything
"""

import re
import time
import hashlib
import logging
import asyncio
import math
import os
from collections import Counter
from datetime import datetime
from typing import Optional, Any
import json

from groq import Groq

from services.semantic_scholar import get_influential_papers
from services.openalex import get_related_concepts, get_topic_trends
from services.academic_search import search_semantic_scholar, search_openalex
from services.cache_manager import cache_manager
from services.async_utils import run_with_timeout

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 🌐 DOMAIN EXPANSION CONFIG
# ─────────────────────────────────────────────
GENERIC_TERMS = {"online", "system", "study", "analysis", "review", "approach", "method", "evaluation", "design", "model"}
WEAK_TOPICS = {"online", "paper", "research", "document", "ai", "ml"}
NOISE_LINE_PATTERNS = [
    r"\bcopyright\b",
    r"\ball rights reserved\b",
    r"\bopen access\b",
    r"\blicen[cs]e\b",
    r"\bauthor contributions?\b",
    r"\bconflict of interest\b",
    r"\bissn\b",
    r"\bdoi\b",
    r"\breceived\b",
    r"\baccepted\b",
    r"\bpublished online\b",
    r"\bjournal\b",
    r"\bpreprint\b",
]
TOPIC_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "these", "those", "using",
    "into", "are", "was", "were", "have", "has", "had", "can", "could", "should",
    "will", "would", "about", "paper", "research", "study", "document", "online",
    "approach", "method", "analysis", "results", "based", "between", "within", "their",
    "our", "its", "also", "such", "more", "most", "than", "over", "article", "authors",
}
MAX_QUERY_LENGTH = 120
DEFAULT_FALLBACK_QUERY = "artificial intelligence and machine learning trends"
GENERIC_QUERY_TERMS = {
    "research", "study", "analysis", "method", "approach", "model",
    "paper", "topic", "system", "data", "results", "overview",
}

_groq_client = None

# Journal / website / boilerplate tokens that often pollute extracted "titles"
NOISE_TOKENS = {
    "www", "http", "https", "com", "org", "net", "edu", "pdf",
    "available", "at", "copyright", "author", "authors", "rights", "reserved",
    "open", "access", "license", "licence", "creativecommons",
    "issn", "isbn", "doi", "vol", "volume", "issue", "pages", "page",
    "international", "journal", "proceedings", "conference",
    # weak filler words common in scraped titles
    "comprehensive", "concept", "concepts", "survey",
    # common scraped journal/site abbreviations
    "ijsrst", "ijert", "ijrte", "ijeat", "ijarcs", "ijcs", "ijete", "ijcse",
}

PHRASE_BOOST = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks",
    "natural language processing",
    "computer vision",
    "reinforcement learning",
    "transformer models",
    "large language models",
]


def _is_invalid_generated_query(query: str) -> bool:
    cleaned = (query or "").strip()
    if not cleaned:
        return True
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}", cleaned.lower())
    if len(words) < 3:
        return True
    if len(set(words)) < len(words):
        return True
    if all(w in GENERIC_QUERY_TERMS for w in words):
        return True
    ocr_noise_hits = sum(1 for w in words if re.search(r"[0-9]{3,}|[a-z]{1,2}[0-9]{2,}|[0-9]{2,}[a-z]{1,2}", w))
    if ocr_noise_hits >= 2:
        return True
    # Reject outputs that still look like raw PDF noise.
    bad_tokens = {
        "www", "http", "https", "issn", "volume", "issue", "department",
        "published", "copyright", "journal", "available"
    }
    if sum(1 for w in words if w in bad_tokens) >= 2:
        return True
    return False


def _extract_headings(document_text: str) -> list[str]:
    headings: list[str] = []
    lines = [l.strip() for l in (document_text or "").splitlines() if l.strip()]
    for line in lines[:180]:
        low = line.lower()
        if len(line) < 4 or len(line) > 140:
            continue
        if re.match(r"^(references|bibliography|appendix)\b", low):
            break
        if re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", line) or line.isupper():
            headings.append(line)
        elif re.match(r"^(introduction|methodology|methods|results|discussion|conclusion|related work)\b", low):
            headings.append(line)
    dedup = []
    seen = set()
    for h in headings:
        key = h.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)
    return dedup[:12]


def _extract_section_after_header(lines: list[str], header_regex: str, stop_regex: str, max_lines: int = 20) -> str:
    header = re.compile(header_regex, re.IGNORECASE)
    stopper = re.compile(stop_regex, re.IGNORECASE)
    for i, line in enumerate(lines):
        if header.search(line):
            collected: list[str] = []
            for j in range(i + 1, min(i + 1 + max_lines, len(lines))):
                if stopper.match(lines[j].strip()):
                    break
                collected.append(lines[j].strip())
            return re.sub(r"\s+", " ", " ".join(collected)).strip()
    return ""


def extract_document_structure(document_text: str) -> dict[str, Any]:
    text = (document_text or "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title_candidates = _extract_title_candidates(text)
    title = title_candidates[0] if title_candidates else (lines[0][:180] if lines else "")

    abstract = _extract_section_after_header(
        lines,
        header_regex=r"^\s*abstract\b",
        stop_regex=r"^\s*(keywords?|index terms?|introduction|1\.|i\.)\b",
        max_lines=24,
    )
    first_page = " ".join(lines[:45])[:1600]
    if not abstract:
        abstract = first_page[:700]

    introduction = _extract_section_after_header(
        lines,
        header_regex=r"^\s*(1\.?\s*)?introduction\b",
        stop_regex=r"^\s*(2\.|ii\.|background|related work|methodology|methods)\b",
        max_lines=26,
    )
    if not introduction:
        introduction = _extract_first_meaningful_paragraph(text)[:700]

    keywords: list[str] = []
    for line in lines[:60]:
        m = re.match(r"^\s*(keywords?|index terms?)\s*[:\-]\s*(.+)$", line, flags=re.IGNORECASE)
        if m:
            kws = re.split(r"[,;•·]", m.group(2))
            keywords = [k.strip() for k in kws if len(k.strip()) > 2][:10]
            break
    if not keywords:
        keywords = _extract_keywords(" ".join([title, abstract, introduction]), top_n=8)

    return {
        "title": title,
        "abstract": abstract,
        "introduction": introduction,
        "keywords": keywords,
        "headings": _extract_headings(text),
        "first_page_content": first_page,
    }


def _fallback_topic_from_structure(structure: dict[str, Any], method: str = "title") -> dict[str, Any]:
    title = (structure.get("title") or "").strip()
    abstract = (structure.get("abstract") or "").strip()
    intro = (structure.get("introduction") or "").strip()
    key_terms = _top_keywords_from_texts([title, abstract, intro], top_n=6)

    if method == "title" and title:
        query = clean_search_query(title, max_chars=90)
    elif method == "abstract" and abstract:
        query = clean_search_query(" ".join(_top_keywords_from_texts([abstract], top_n=8)), max_chars=90)
    else:
        query = clean_search_query(" ".join(key_terms[:8]), max_chars=90)

    if _is_invalid_generated_query(query):
        query = DEFAULT_FALLBACK_QUERY

    title_guess = _clean_topic_phrase(title, max_words=12) or "Academic Research Topic"
    return {
        "main_topic": title_guess,
        "concise_title": title_guess,
        "search_query": query,
        "domain": "AI Research" if "ai" in query.lower() or "artificial intelligence" in query.lower() else "Computer Science",
        "subdomain": "Machine Learning" if "learning" in query.lower() else "General",
        "confidence": 0.55,
        "fallback_method": method,
    }


def _synthesize_topic_with_llm(structure: dict[str, Any]) -> dict[str, Any] | None:
    client = _get_groq_client()
    if not client:
        return None
    prompt = f"""
You are an academic research analyst.

Given this research paper content:
- Title: {structure.get("title", "")}
- Abstract: {structure.get("abstract", "")[:1200]}
- Introduction: {structure.get("introduction", "")[:1200]}
- Keywords: {", ".join(structure.get("keywords", [])[:10])}
- Headings: {", ".join(structure.get("headings", [])[:10])}

1. Identify the main research topic
2. Generate a concise academic title
3. Generate a Semantic Scholar search query
4. Identify the research domain
5. Return only JSON

Rules:
- concise
- professional
- academic wording
- max 10 words for search query
- avoid generic keywords

Return JSON with keys:
main_topic, concise_title, search_query, domain, subdomain, confidence
"""
    try:
        res = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        raw = (res.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        data = json.loads(m.group())
        if not isinstance(data, dict):
            return None
        data["search_query"] = _clean_generated_query(str(data.get("search_query", "")))
        data["main_topic"] = _clean_topic_phrase(str(data.get("main_topic", "")), max_words=14)
        data["concise_title"] = _clean_topic_phrase(str(data.get("concise_title", data.get("main_topic", ""))), max_words=14)
        data["domain"] = _clean_topic_phrase(str(data.get("domain", "")), max_words=5)
        data["subdomain"] = _clean_topic_phrase(str(data.get("subdomain", "")), max_words=6)
        try:
            data["confidence"] = round(float(data.get("confidence", 0.0)), 2)
        except Exception:
            data["confidence"] = 0.0
        return data
    except Exception:
        logger.exception("[TOPIC SYNTHESIS] LLM synthesis failed")
        return None


def _clean_generated_query(query: str) -> str:
    q = (query or "").replace('"', "").replace("'", "").replace("\n", " ").strip().lower()
    q = re.sub(r"\s+", " ", q).strip()
    words = q.split()
    if len(words) > 12:
        q = " ".join(words[:12])
    return q


def _get_groq_client() -> Optional[Groq]:
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    _groq_client = Groq(api_key=api_key)
    return _groq_client


def generate_search_query(text: str) -> str:
    structure = extract_document_structure(text or "")
    synthesized = _synthesize_topic_with_llm(structure)
    if synthesized:
        query = _clean_generated_query(synthesized.get("search_query", ""))
        if not _is_invalid_generated_query(query):
            logger.info("[GENERATED SEARCH QUERY] %s", query)
            return query

    prompt = f"""
You are an academic research assistant.

Extract a CLEAN and SHORT academic search query
from the following research paper text.

RULES:
- Return ONLY the query.
- Max 12 words.
- Remove junk text, URLs, headers, ISSN, journal info.
- Focus only on the main research topic.
- Include important AI/ML keywords if present.
- No explanations.

Paper Text:
{(text or "")[:3000]}
"""
    query = ""
    try:
        client = _get_groq_client()
        if client:
            response = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            query = response.choices[0].message.content.strip()
    except Exception:
        logger.exception("[QUERY GEN] Groq generation failed")

    query = _clean_generated_query(query)
    if _is_invalid_generated_query(query):
        query = DEFAULT_FALLBACK_QUERY

    print("Generated Search Query:", query)
    logger.info("[GENERATED SEARCH QUERY] %s", query)
    return query


def clean_research_query(text: str, fallback_text: str = "", min_words: int = 5, max_words: int = 12) -> str:
    """
    Extract a clean semantic topic query from noisy scraped text.

    Constraints (by design):
    - removes URLs / boilerplate / identifiers (ISSN/ISBN/DOI), repeated words, extra whitespace
    - outputs 5–12 important words (when possible), lowercase normalized
    - if the input is low quality, uses `fallback_text` (abstract/introduction/first paragraph)
    """
    def _strip_noise(s: str) -> str:
        s = (s or "").replace("\u00a0", " ").strip()
        if not s:
            return ""
        low = s.lower()
        # remove URLs / domains
        low = re.sub(r"https?://\S+", " ", low)
        low = re.sub(r"\bwww\.[^\s]+\b", " ", low)
        low = re.sub(r"\bwww\b", " ", low)
        low = re.sub(r"\b\S+\.(com|org|net|edu)\b", " ", low)
        # remove boilerplate phrases
        low = re.sub(r"\bavailable\s+at\b", " ", low)
        low = re.sub(r"\ball\s+rights\s+reserved\b", " ", low)
        low = re.sub(r"\bthis\s+open\s+access\s+article\b", " ", low)
        # remove identifiers / pagination
        low = re.sub(r"\bissn\s*[:#]?\s*\d{4}\s*[-–]\s*\d{3}[\dx]\b", " ", low)
        low = re.sub(r"\bisbn\s*[:#]?\s*(97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d\b", " ", low)
        low = re.sub(r"\bdoi\s*[:#]?\s*10\.\d{4,9}/\S+\b", " ", low)
        low = re.sub(r"\bpp?\.\s*\d+(\s*[-–]\s*\d+)?\b", " ", low)
        low = re.sub(r"\bpages?\s*\d+(\s*[-–]\s*\d+)?\b", " ", low)
        low = re.sub(r"\b(19\d{2}|20\d{2})\b", " ", low)
        # normalize punctuation to spaces
        low = re.sub(r"[^a-z0-9\-\s]", " ", low)
        low = re.sub(r"\s+", " ", low).strip()
        return low

    def _tokenize(s: str) -> list[str]:
        toks = re.findall(r"\b[a-z][a-z0-9\-]{1,}\b", (s or "").lower())
        out: list[str] = []
        seen = set()
        for t in toks:
            if t in TOPIC_STOPWORDS or t in NOISE_TOKENS or t in GENERIC_TERMS or t in WEAK_TOPICS:
                continue
            if t.isdigit():
                continue
            if len(t) <= 2:
                continue
            # de-dupe repeated words (common in scraped headers)
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
        return out

    def _looks_low_quality(raw: str, tokens: list[str]) -> bool:
        if not raw or len(raw.strip()) < 10:
            return True
        low = raw.lower()
        if "available at" in low or "open access" in low or "all rights reserved" in low:
            return True
        if re.search(r"https?://|www\.", low):
            return True
        # if most tokens are noise/stopwords, treat as low quality
        if len(tokens) < 3:
            return True
        return False

    raw = text or ""
    stripped = _strip_noise(raw)
    tokens = _tokenize(stripped)

    if _looks_low_quality(raw, tokens) and fallback_text:
        stripped_fb = _strip_noise(fallback_text[:1000])
        tokens_fb = _tokenize(stripped_fb)
        if len(tokens_fb) > len(tokens):
            stripped, tokens = stripped_fb, tokens_fb

    if not tokens:
        return ""

    # Boost known keyphrases (keep phrase words together and early)
    phrase_tokens: list[str] = []
    for phrase in PHRASE_BOOST:
        ph = phrase.lower()
        if ph in stripped:
            phrase_tokens.extend([w for w in ph.split() if w not in TOPIC_STOPWORDS and w not in NOISE_TOKENS])

    # Rank by frequency in the (stripped) text blob, lightly prefer longer tokens
    freq = Counter(re.findall(r"\b[a-z][a-z0-9\-]{1,}\b", stripped))
    ranked = sorted(tokens, key=lambda t: (-freq.get(t, 0), -len(t), t))

    chosen: list[str] = []
    seen = set()
    for t in phrase_tokens + ranked:
        if t in seen:
            continue
        seen.add(t)
        chosen.append(t)
        if len(chosen) >= max_words:
            break

    # Ensure min_words if we can (but do not exceed max_words)
    if len(chosen) < min_words:
        for t in ranked:
            if t in seen:
                continue
            seen.add(t)
            chosen.append(t)
            if len(chosen) >= min_words or len(chosen) >= max_words:
                break

    return " ".join(chosen[:max_words]).strip().lower()

DOMAIN_SUBDOMAINS = {
    "artificial intelligence": ["Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning"],
    "machine learning": ["Deep Learning", "Supervised Learning", "Unsupervised Learning", "Reinforcement Learning", "Neural Networks"],
    "computer science": ["Artificial Intelligence", "Cybersecurity", "Software Engineering", "Data Science", "Computer Networks"],
    "data science": ["Machine Learning", "Data Mining", "Big Data", "Data Visualization", "Predictive Analytics"],
}

# ─────────────────────────────────────────────
# 📄  METADATA EXTRACTION
# ─────────────────────────────────────────────
def extract_paper_metadata(document_text: str) -> dict:
    """
    Extract title, abstract, keywords, and year from raw document text.
    Uses heuristics — no LLM call needed.
    """
    text = document_text.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── Title
    title = ""
    for line in lines[:15]:
        if 10 < len(line) < 200 and not line.isupper():
            title = line
            break
    if not title and lines:
        title = lines[0][:150]

    # ── Abstract
    abstract = ""
    abs_pattern = re.compile(r'\babstract\b', re.IGNORECASE)
    for i, line in enumerate(lines):
        if abs_pattern.search(line):
            section_header = re.compile(
                r'^\s*(introduction|keywords?|1\.|background|related work)\s*$',
                re.IGNORECASE
            )
            abs_lines = []
            for j in range(i + 1, min(i + 20, len(lines))):
                if section_header.match(lines[j]):
                    break
                abs_lines.append(lines[j])
            abstract = " ".join(abs_lines)[:800]
            break

    if not abstract and len(text) > len(title) + 50:
        abstract = text[len(title):len(title) + 500].strip()

    # ── Keywords
    keywords = []
    kw_pattern = re.compile(r'\bkeywords?\s*[:\-]?\s*(.+)', re.IGNORECASE)
    for line in lines[:30]:
        m = kw_pattern.match(line)
        if m:
            raw_kw = m.group(1)
            kws = re.split(r'[,;·•]', raw_kw)
            keywords = [k.strip() for k in kws if 2 < len(k.strip()) < 50][:10]
            break

    if not keywords and abstract:
        words = re.findall(r'\b[A-Za-z][a-z]{4,}\b', abstract)
        freq = {}
        for w in words:
            freq[w.lower()] = freq.get(w.lower(), 0) + 1
        stopwords = {
            "which", "their", "there", "these", "those", "about", "after",
            "before", "other", "using", "based", "paper", "study", "results",
            "method", "approach", "proposed", "present", "shows", "shown"
        }
        keywords = [
            w for w, c in sorted(freq.items(), key=lambda x: -x[1])
            if w not in stopwords
        ][:8]

    # ── Year
    year = None
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', text[:500])
    if year_match:
        year = int(year_match.group())

    return {
        "title":    title,
        "abstract": abstract,
        "keywords": keywords,
        "year":     year,
        "text_length": len(text),
    }


def _extract_first_meaningful_paragraph(document_text: str) -> str:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", document_text or "") if p.strip()]
    for para in paragraphs:
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", para)
        if len(words) >= 20:
            return para[:500]
    return ""


def _top_keywords_from_texts(texts: list[str], top_n: int = 4) -> list[str]:
    blob = " ".join([t for t in texts if t]).lower()
    tokens = re.findall(r"\b[a-z][a-z\-]{2,}\b", blob)
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "these", "those", "using",
        "into", "are", "was", "were", "have", "has", "had", "can", "could", "should",
        "will", "would", "about", "paper", "research", "study", "document", "online",
        "approach", "method", "analysis", "results", "based", "between", "within", "their",
        "your", "our", "its", "also", "such", "more", "most", "than", "over"
    }
    freq = {}
    for t in tokens:
        if t in stop or len(t) < 4:
            continue
        freq[t] = freq.get(t, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:top_n]]


def _is_weak_topic(topic: str) -> bool:
    words = re.findall(r"[a-z]+", (topic or "").lower())
    if not words:
        return True
    # Weak when topic collapses to generic terms only.
    return all(w in WEAK_TOPICS for w in words)


def _looks_like_noise_line(line: str) -> bool:
    low = (line or "").strip().lower()
    if not low:
        return True
    for pat in NOISE_LINE_PATTERNS:
        if re.search(pat, low):
            return True
    if re.match(r"^(vol|volume|issue|pages?)\b", low):
        return True
    if re.search(r"\b\d{4}\b", low) and len(low) < 24:
        return True
    return False


def _clean_topic_phrase(topic: str, max_words: int = 12) -> str:
    one_line = re.sub(r"\s+", " ", (topic or "").strip())
    one_line = re.sub(r"[^\w\s\-\(\)]", " ", one_line)
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", one_line)
    cleaned = []
    seen = set()
    for t in tokens:
        key = t.lower()
        if key in TOPIC_STOPWORDS:
            continue
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)
        if len(cleaned) >= max_words:
            break
    return " ".join(cleaned).strip()


def _extract_title_candidates(document_text: str) -> list[str]:
    raw_lines = [l.strip() for l in (document_text or "").splitlines()]
    head = [l for l in raw_lines[:80] if l]
    candidates = []
    for i, line in enumerate(head):
        if _looks_like_noise_line(line):
            continue
        if len(line) < 15 or len(line) > 180:
            continue
        if re.match(r"^\d+(\.\d+)*\s+", line):
            continue
        # Prioritize first-page title-like lines with higher lexical density.
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", line)
        if len(words) < 3:
            continue
        uppercase_ratio = sum(1 for c in line if c.isupper()) / max(1, len([c for c in line if c.isalpha()]))
        score = (3.0 if i < 18 else 0.0) + (2.0 if 0.08 <= uppercase_ratio <= 0.55 else 0.0) + min(len(words) / 8.0, 2.0)
        if ":" in line:
            score += 0.5
        candidates.append((score, line))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [c[1] for c in candidates[:5]]


def _remove_duplicate_lines(lines: list[str]) -> list[str]:
    unique = []
    seen = set()
    for line in lines:
        key = re.sub(r"\s+", " ", line.strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(line.strip())
    return unique


def _clean_document_for_topic_extraction(document_text: str) -> str:
    raw = document_text or ""
    # Ignore references/bibliography section to avoid noisy query terms.
    refs_match = re.search(r"(^|\n)\s*(references|bibliography)\s*($|\n)", raw, re.IGNORECASE)
    if refs_match:
        raw = raw[: refs_match.start()]

    raw = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", raw)
    lines = [l.strip() for l in raw.splitlines() if l and l.strip()]
    lines = _remove_duplicate_lines(lines)

    noise_patterns = NOISE_LINE_PATTERNS + [
        r"\bopen access article\b",
        r"\bauthor\b",
        r"\bcorresponding author\b",
        r"\bcreativecommons\b",
        r"\blicense\b",
        r"\bpreprint\b",
        r"\barxiv\b",
        r"\bthis article\b",
        r"\bdoi\s*[:/]",
    ]
    cleaned_lines = []
    for line in lines:
        low = line.lower()
        if any(re.search(p, low) for p in noise_patterns):
            continue
        if re.match(r"^\s*(figure|table)\s+\d+", low):
            continue
        if len(line) < 3:
            continue
        cleaned_lines.append(line)
    return re.sub(r"\s+", " ", "\n".join(cleaned_lines)).strip()


def _extract_keywords(clean_text: str, top_n: int = 8) -> list[str]:
    keyword_patterns = [
        r"\b(transformer(?:s)?|attention(?:\s+mechanism)?|llm(?:s)?|large language model(?:s)?)\b",
        r"\b(convolutional neural network(?:s)?|cnn(?:s)?|rnn(?:s)?|lstm(?:s)?)\b",
        r"\b(deep learning|machine learning|artificial intelligence|reinforcement learning)\b",
        r"\b(natural language processing|computer vision|medical imaging|time series)\b",
        r"\b(classification|segmentation|optimization|generalization|fine[- ]tuning)\b",
    ]
    kws = []
    low = clean_text.lower()
    for pat in keyword_patterns:
        kws.extend([m.group(1).strip() for m in re.finditer(pat, low, re.IGNORECASE)])

    words = re.findall(r"\b[a-z][a-z0-9\-]{3,}\b", low)
    freq = Counter(w for w in words if w not in TOPIC_STOPWORDS and w not in GENERIC_TERMS)
    for token, _ in freq.most_common(top_n * 3):
        if token not in kws and len(token) > 3:
            kws.append(token)
    dedup = []
    seen = set()
    for kw in kws:
        key = kw.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(kw)
        if len(dedup) >= top_n:
            break
    return dedup


def _compress_query(query: str, max_chars: int = MAX_QUERY_LENGTH) -> str:
    query = clean_search_query(query, max_chars=max_chars * 2)
    tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", query) if t.lower() not in TOPIC_STOPWORDS]
    compact = []
    seen = set()
    for token in tokens:
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        compact.append(token)
        candidate = " ".join(compact)
        if len(candidate) > max_chars:
            compact.pop()
            break
    return " ".join(compact)[:max_chars].strip()


def extract_research_topic(document_text: str) -> dict:
    structure = extract_document_structure(document_text)
    logger.info("[TOPIC DEBUG] extracted title=%s", structure.get("title", ""))
    logger.info("[TOPIC DEBUG] headings=%s", structure.get("headings", [])[:5])

    synthesized = _synthesize_topic_with_llm(structure)
    fallback_method = "none"
    if not synthesized or _is_invalid_generated_query(synthesized.get("search_query", "")):
        if structure.get("title"):
            synthesized = _fallback_topic_from_structure(structure, method="title")
            fallback_method = "title"
        elif structure.get("abstract"):
            synthesized = _fallback_topic_from_structure(structure, method="abstract")
            fallback_method = "abstract"
        else:
            synthesized = _fallback_topic_from_structure(structure, method="tfidf")
            fallback_method = "tfidf"

    semantic_query = _clean_generated_query(synthesized.get("search_query", ""))
    if _is_invalid_generated_query(semantic_query):
        synthesized = _fallback_topic_from_structure(structure, method="tfidf")
        semantic_query = synthesized.get("search_query", DEFAULT_FALLBACK_QUERY)
        fallback_method = "tfidf"

    topic_title = synthesized.get("concise_title") or synthesized.get("main_topic") or structure.get("title") or "Academic Research Topic"
    keywords = structure.get("keywords", [])[:8]
    if not keywords:
        keywords = _top_keywords_from_texts([structure.get("title", ""), structure.get("abstract", ""), structure.get("introduction", "")], top_n=8)

    logger.info("[TOPIC DEBUG] generated query=%s", semantic_query)
    logger.info("[TOPIC DEBUG] fallback method=%s", fallback_method)

    return {
        "title": _clean_topic_phrase(topic_title, max_words=14),
        "keywords": keywords,
        "semantic_query": semantic_query,
        "domain": synthesized.get("domain", "Computer Science"),
        "subdomain": synthesized.get("subdomain", "General"),
        "main_topic": synthesized.get("main_topic", topic_title),
        "confidence": float(synthesized.get("confidence", 0.0) or 0.0),
        "fallback_method": fallback_method if fallback_method != "none" else synthesized.get("fallback_method", "none"),
        "document_structure": structure,
        "abstract_preview": (structure.get("abstract") or structure.get("introduction") or structure.get("first_page_content", "")[:500]).strip()[:500],
    }


def clean_search_query(topic: str, max_chars: int = MAX_QUERY_LENGTH) -> str:
    query = (topic or "").replace("\n", " ").strip()
    query = re.sub(r"[^\w\s\-]", " ", query)
    query = re.sub(r"\s+", " ", query).strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{1,}", query)
    dedup = []
    seen = set()
    for w in words:
        k = w.lower()
        if k in TOPIC_STOPWORDS:
            continue
        if k in seen:
            continue
        seen.add(k)
        dedup.append(w)
    cleaned = " ".join(dedup).strip()
    return cleaned[:max_chars].strip()


def _build_fallback_queries(topic: str) -> list[str]:
    cleaned = clean_search_query(topic)
    words = cleaned.split()
    fallbacks = []
    if cleaned:
        fallbacks.append(cleaned)
    if len(words) >= 2:
        fallbacks.append(" ".join(words[:2]))
    if len(words) >= 4:
        # keyword-only compressed form
        initials = []
        for w in words:
            if w.lower() in {"artificial", "intelligence"}:
                initials.append("AI")
            else:
                initials.append(w)
        fallbacks.append(" ".join(initials[:3]))
    compact = " ".join(words[:3]).strip()
    if compact:
        fallbacks.append(compact)
    # noun-phrase-ish bigrams from the cleaned query
    if len(words) >= 5:
        bigrams = []
        for i in range(min(len(words) - 1, 10)):
            a, b = words[i], words[i + 1]
            if a.lower() in TOPIC_STOPWORDS or b.lower() in TOPIC_STOPWORDS:
                continue
            bigrams.append(f"{a} {b}")
        if bigrams:
            fallbacks.append(" ".join(bigrams[:3]))
    # preserve order and uniqueness
    seen = set()
    uniq = []
    for q in fallbacks:
        k = q.lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(q[:80])
    return uniq[:4]


def _build_multi_strategy_queries(topic_data: dict) -> list[str]:
    semantic_q = _compress_query(topic_data.get("semantic_query", ""), MAX_QUERY_LENGTH)
    if not semantic_q:
        semantic_q = DEFAULT_FALLBACK_QUERY
    # Keep all strategies anchored to the cleaned semantic query.
    shorter = " ".join(semantic_q.split()[:6]).strip()
    domain_only = clean_search_query(topic_data.get("domain", ""), max_chars=80)
    queries = [semantic_q, shorter, domain_only]
    ordered = []
    seen = set()
    for q in queries:
        if not q:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(q)
    return ordered[:4]


async def _multi_strategy_academic_search(topic_data: dict, per_source_limit: int = 8) -> tuple[list[dict], str, str]:
    queries = _build_multi_strategy_queries(topic_data)
    if not queries:
        queries = _build_fallback_queries(topic_data.get("semantic_query", ""))

    all_papers: list[dict] = []
    used_ss_query = ""
    used_oa_query = ""
    query_labels = ["title", "semantic", "keywords", "embedding"]

    for i, q in enumerate(queries):
        label = query_labels[i] if i < len(query_labels) else f"fallback_{i+1}"
        logger.info("[SEMANTIC QUERY] strategy=%s query=%s", label, q)

        ss_batch = []
        oa_batch = []
        try:
            logger.info("[SS SEARCH] strategy=%s query=%s", label, q)
            ss_batch = await run_with_timeout(search_semantic_scholar(q, limit=per_source_limit), 12, fallback_value=[]) or []
            if ss_batch and not used_ss_query:
                used_ss_query = q
        except Exception:
            logger.exception("[SS SEARCH] failed strategy=%s query=%s", label, q)

        try:
            logger.info("[OA SEARCH] strategy=%s query=%s", label, q)
            oa_batch = await run_with_timeout(search_openalex(q, limit=per_source_limit), 12, fallback_value=[]) or []
            if oa_batch and not used_oa_query:
                used_oa_query = q
        except Exception:
            logger.exception("[OA SEARCH] failed strategy=%s query=%s", label, q)

        all_papers.extend(ss_batch)
        all_papers.extend(oa_batch)
        logger.info("[SIMILAR PAPERS FOUND] strategy=%s ss=%d oa=%d cumulative=%d", label, len(ss_batch), len(oa_batch), len(all_papers))

    return all_papers, used_ss_query, used_oa_query


def _tokenize_for_similarity(text: str) -> list[str]:
    return [t for t in re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", (text or "").lower()) if t not in TOPIC_STOPWORDS]


def _tfidf_cosine_similarity(doc_a: str, doc_b: str) -> float:
    tokens_a = _tokenize_for_similarity(doc_a)
    tokens_b = _tokenize_for_similarity(doc_b)
    if not tokens_a or not tokens_b:
        return 0.0
    vocab = set(tokens_a) | set(tokens_b)
    if not vocab:
        return 0.0
    tf_a = {t: tokens_a.count(t) / len(tokens_a) for t in vocab}
    tf_b = {t: tokens_b.count(t) / len(tokens_b) for t in vocab}
    idf = {}
    for t in vocab:
        df = int(t in set(tokens_a)) + int(t in set(tokens_b))
        idf[t] = math.log((2 + 1) / (df + 1)) + 1.0
    vec_a = {t: tf_a[t] * idf[t] for t in vocab}
    vec_b = {t: tf_b[t] * idf[t] for t in vocab}
    dot = sum(vec_a[t] * vec_b[t] for t in vocab)
    mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
    mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────
# ⭐  ENHANCED QUALITY INDEX
# ─────────────────────────────────────────────
def compute_enhanced_quality(
    metadata: dict,
    similar_papers: list[dict],
    oa_works: list[dict],
) -> dict:
    current_year = datetime.now().year

    # ── Citation score
    citation_counts = [p.get("citation_count", 0) for p in similar_papers if p.get("citation_count")]
    if citation_counts:
        avg_citations = sum(citation_counts) / len(citation_counts)
        max_citations = max(citation_counts)
        citation_score = min(10.0, (avg_citations / max(max_citations, 1)) * 10)
    else:
        citation_score = 5.0

    # ── Novelty score
    novelty_score = compute_novelty_score(metadata, similar_papers)

    # ── Recency score
    paper_year = metadata.get("year")
    if paper_year:
        age = current_year - paper_year
        if age <= 1: recency_score = 10.0
        elif age <= 2: recency_score = 8.5
        elif age <= 3: recency_score = 7.0
        elif age <= 5: recency_score = 5.5
        elif age <= 8: recency_score = 3.5
        else: recency_score = 2.0
    else:
        recency_score = 5.0

    # ── Source quality
    high_quality_venues = {
        "nature", "science", "cell", "ieee", "acm", "springer", "elsevier",
        "neurips", "icml", "iclr", "cvpr", "emnlp", "acl", "aaai", "ijcai",
        "arxiv", "plos", "lancet", "nejm", "jama"
    }
    venue_scores = []
    for p in similar_papers[:5]:
        venue = (p.get("venue") or "").lower()
        if any(hq in venue for hq in high_quality_venues):
            venue_scores.append(9.0)
        elif venue:
            venue_scores.append(6.0)
        else:
            venue_scores.append(4.0)
    source_quality = sum(venue_scores) / len(venue_scores) if venue_scores else 5.0

    # ── Overall
    overall = round(
        0.25 * citation_score +
        0.30 * novelty_score +
        0.20 * recency_score +
        0.25 * source_quality,
        1
    )

    return {
        "citation_score":  round(citation_score, 1),
        "novelty_score":   round(novelty_score, 1),
        "recency_score":   round(recency_score, 1),
        "source_quality":  round(source_quality, 1),
        "overall":         overall,
        "interpretation": _interpret_quality(overall),
    }


def _interpret_quality(score: float) -> str:
    if score >= 8.5: return "Excellent — highly novel, recent, and well-positioned."
    elif score >= 7.0: return "Good — solid research with room for improvement."
    elif score >= 5.5: return "Average — consider strengthening novelty."
    elif score >= 4.0: return "Below average — significant improvements needed."
    else: return "Needs major revision — thorough literature review recommended."


# ─────────────────────────────────────────────
# 🔬  NOVELTY SCORE
# ─────────────────────────────────────────────
def compute_novelty_score(metadata: dict, similar_papers: list[dict]) -> float:
    if not similar_papers:
        return 7.5

    paper_text = f"{metadata.get('title', '')} {metadata.get('abstract', '')}".lower()
    paper_tokens = set(re.findall(r'\b[a-z]{4,}\b', paper_text))

    if not paper_tokens:
        return 5.0

    overlap_scores = []
    for p in similar_papers[:8]:
        ext_text = f"{p.get('title', '')} {p.get('abstract', '')}".lower()
        ext_tokens = set(re.findall(r'\b[a-z]{4,}\b', ext_text))
        if ext_tokens:
            intersection = len(paper_tokens & ext_tokens)
            union = len(paper_tokens | ext_tokens)
            jaccard = intersection / union if union > 0 else 0
            overlap_scores.append(jaccard)

    if not overlap_scores:
        return 7.5

    avg_overlap = sum(overlap_scores) / len(overlap_scores)
    novelty = round((1 - avg_overlap) * 10, 1)
    return max(1.0, min(10.0, novelty))


# ─────────────────────────────────────────────
# 📚  MISSING CITATIONS ANALYSIS
# ─────────────────────────────────────────────
def find_missing_citations(
    metadata: dict,
    similar_papers: list[dict],
    influential_papers: list[dict],
) -> list[dict]:
    current_year = datetime.now().year
    paper_year   = metadata.get("year") or current_year
    paper_text   = f"{metadata.get('title', '')} {metadata.get('abstract', '')}".lower()

    suggestions = []
    seen_titles  = set()
    all_candidates = similar_papers + influential_papers

    for p in all_candidates:
        title = p.get("title", "")
        if not title or title.lower() in seen_titles: continue
        seen_titles.add(title.lower())

        year = p.get("year") or 0
        citations = p.get("citation_count", 0)

        if year and paper_year and year > paper_year: continue

        title_words = set(re.findall(r'\b[a-z]{4,}\b', title.lower()))
        overlap = len(title_words & set(re.findall(r'\b[a-z]{4,}\b', paper_text)))

        if citations >= 50 and overlap < 3:
            reason = _citation_reason(citations, year, current_year)
            suggestions.append({
                "title":         title,
                "authors":       p.get("authors", [])[:3],
                "year":          year,
                "citation_count": citations,
                "venue":         p.get("venue", ""),
                "reason":        reason,
                "url":           p.get("url", ""),
            })

    suggestions.sort(key=lambda x: x["citation_count"], reverse=True)
    return suggestions[:8]


def _citation_reason(citations: int, year: int, current_year: int) -> str:
    age = current_year - (year or current_year)
    if citations >= 1000: return f"Seminal paper ({citations:,} citations)."
    elif citations >= 200: return f"Highly influential ({citations} citations)."
    elif age <= 3: return f"Recent work ({year}) with {citations} citations."
    else: return f"Well-cited paper ({citations} citations)."


# ─────────────────────────────────────────────
# 💡  RESEARCH SUGGESTIONS ENGINE
# ─────────────────────────────────────────────
def generate_research_suggestions(
    metadata: dict,
    similar_papers: list[dict],
    oa_works: list[dict],
    trends_data: dict,
    missing_citations: list[dict],
    novelty_score: float,
    quality_scores: dict,
) -> list[dict]:
    suggestions = []
    current_year = datetime.now().year
    paper_year   = metadata.get("year") or current_year

    if missing_citations:
        suggestions.append({
            "category":   "Missing Citations",
            "priority":   "High",
            "suggestion": f"Add {len(missing_citations)} potentially missing citations.",
            "details":    "Highly-cited papers not referenced strengthen your credibility.",
        })

    if paper_year and (current_year - paper_year) > 3:
        suggestions.append({
            "category":   "Outdated References",
            "priority":   "High",
            "suggestion": "Update literature review with recent work.",
            "details":    f"Add papers from {current_year - 2}–{current_year} to stay current.",
        })

    if novelty_score < 6.0:
        suggestions.append({
            "category":   "Novelty",
            "priority":   "High",
            "suggestion": "Strengthen the novelty claim.",
            "details":    "Clearly articulate what is new vs existing literature.",
        })

    trend = trends_data.get("trend", "stable")
    keyword = trends_data.get("keyword", "")
    if trend == "growing":
        suggestions.append({
            "category":   "Trend Alignment",
            "priority":   "Medium",
            "suggestion": f"Leverage growth in '{keyword}'.",
            "details":    "Emphasize contribution to this growing trend.",
        })

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions.sort(key=lambda s: priority_order.get(s["priority"], 3))
    return suggestions


# ─────────────────────────────────────────────
# 📈  DOMAIN-AWARE TRENDS ENGINE (ASYNC)
# ─────────────────────────────────────────────
async def get_domain_aware_trends_async(keyword: str) -> dict:
    clean_kw = keyword.strip().lower()
    is_generic = clean_kw in GENERIC_TERMS
    warning_message = f"'{keyword}' is too generic." if is_generic else ""

    subdomains = DOMAIN_SUBDOMAINS.get(clean_kw, [])
    queries = [keyword] + subdomains

    async def fetch_one(q):
        t = await get_topic_trends(q, years=10)
        kw = await get_trending_keywords(q, top_n=5)
        return {"name": q, "trends": t, "trending_keywords": kw}

    tasks = [fetch_one(q) for q in queries]
    done = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = {"subdomains": []}
    for res in done:
        if isinstance(res, Exception): continue
        if res["name"] == keyword:
            results["overall_trends"] = res["trends"]
            results["main_trending_keywords"] = res["trending_keywords"]
        else:
            results["subdomains"].append(res)

    if "overall_trends" not in results:
        results["overall_trends"] = {"keyword": keyword, "yearly_counts": [], "total": 0, "trend": "unknown"}
        results["main_trending_keywords"] = []

    return {
        "keyword": keyword,
        "is_generic": is_generic,
        "warning_message": warning_message,
        "overall_trends": results["overall_trends"],
        "main_trending_keywords": results["main_trending_keywords"],
        "subdomains": results["subdomains"]
    }


# ─────────────────────────────────────────────
# 🚀  FULL ANALYSIS ORCHESTRATOR (ASYNC)
# ─────────────────────────────────────────────
async def run_full_academic_analysis_async(document_text: str) -> dict:
    if not document_text or not document_text.strip():
        return {"status": "error", "message": "No text"}

    doc_hash = hashlib.md5(document_text[:3000].encode()).hexdigest()
    cached = await cache_manager.get(doc_hash, category="academic_analysis")
    if cached: return cached

    start_time = time.time()
    metadata = extract_paper_metadata(document_text)
    topic_data = extract_research_topic(document_text)
    topic = topic_data.get("semantic_query") or DEFAULT_FALLBACK_QUERY
    logger.info("[TOPIC EXTRACTION] title=%s", topic_data.get("title", ""))
    logger.info("[TOPIC EXTRACTION] keywords=%s", topic_data.get("keywords", []))
    logger.info("[SEMANTIC QUERY] %s", topic)

    # Safe defaults to prevent researcher pipeline crashes on API/network issues.
    ss_similar = []
    oa_similar = []
    similar_papers = []
    trend_data = {}
    suggestions = []
    overlap_analysis = {}
    warnings = []
    ss_influential = []
    oa_concepts = []
    oa_works = []

    all_raw_papers, used_ss_query, used_oa_query = await _multi_strategy_academic_search(topic_data, per_source_limit=8)
    ss_similar = [p for p in all_raw_papers if (p.get("source_api") or p.get("source", "")).lower().startswith("semantic")]
    oa_similar = [p for p in all_raw_papers if (p.get("source_api") or p.get("source", "")).lower().startswith("openalex")]

    try:
        ss_influential = await run_with_timeout(
            get_influential_papers(topic, limit=5),
            10,
            fallback_value=[],
        ) or []
    except Exception:
        logger.exception("[RESEARCH] Influential papers fetch failed for topic=%s", topic)
        ss_influential = []

    try:
        oa_concepts = await run_with_timeout(
            get_related_concepts(topic, limit=8),
            10,
            fallback_value=[],
        ) or []
    except Exception:
        logger.exception("[RESEARCH] Related concepts fetch failed for topic=%s", topic)
        oa_concepts = []

    # Merge -> dedupe -> schema guard -> empty filter
    similar_papers = all_raw_papers
    normalized_papers = []
    seen_titles = set()
    for paper in similar_papers:
        if not isinstance(paper, dict):
            continue
        title = (paper.get("title") or "").strip()
        if not title:
            continue
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)
        normalized_papers.append({
            "title": title,
            "authors": paper.get("authors") or [],
            "year": paper.get("year"),
            "citation_count": paper.get("citation_count", 0) or 0,
            "venue": paper.get("venue") or "",
            "url": paper.get("url") or "",
            "abstract": paper.get("abstract") or "",
            "source_api": paper.get("source_api") or paper.get("source") or "",
            "relevance_score": paper.get("relevance_score", 0.0),
        })

    doc_semantic = topic_data.get("semantic_query", "")
    doc_keywords = set(_tokenize_for_similarity(" ".join(topic_data.get("keywords") or [])))
    doc_abstract = topic_data.get("abstract_preview", "")
    ranked = []
    for p in normalized_papers:
        paper_text = f"{p.get('title', '')} {p.get('abstract', '')}"
        semantic_sim = _tfidf_cosine_similarity(doc_semantic, paper_text)
        paper_tokens = set(_tokenize_for_similarity(paper_text))
        keyword_sim = (len(doc_keywords & paper_tokens) / max(1, len(doc_keywords | paper_tokens))) if doc_keywords else 0.0
        abstract_sim = _tfidf_cosine_similarity(doc_abstract, p.get("abstract", ""))
        score = (0.4 * semantic_sim) + (0.3 * keyword_sim) + (0.3 * abstract_sim)
        p["similarity_percent"] = round(score * 100, 1)
        p["abstract_snippet"] = (p.get("abstract") or "")[:240]
        p["source"] = p.get("source_api") or p.get("source") or "Unknown"
        logger.info(
            "[SIMILARITY SCORE] title=%s semantic=%.3f keyword=%.3f abstract=%.3f final=%.3f",
            p.get("title", "")[:80], semantic_sim, keyword_sim, abstract_sim, score
        )
        ranked.append((p, score))

    ranked.sort(key=lambda x: (x[1], int(x[0].get("citation_count", 0) or 0), float(x[0].get("relevance_score", 0.0)), x[0].get("year") or 0), reverse=True)
    merged_papers = [p for p, _ in ranked[:5]]
    oa_works = oa_similar
    logger.info("[PAPERS FOUND] final_unique=%d", len(merged_papers))

    if not merged_papers:
        warnings.append("Limited academic comparison data available")

    quality_scores = compute_enhanced_quality(metadata, merged_papers, [])
    novelty_score = quality_scores["novelty_score"]
    missing_citations = find_missing_citations(metadata, merged_papers, ss_influential)

    # Trend analysis must use semantic query only.
    trend_from_api = await run_with_timeout(
        get_topic_trends(topic_data.get("semantic_query", topic), years=10),
        12,
        fallback_value={"keyword": topic, "yearly_counts": [], "total": 0, "trend": "unknown"},
    ) or {"keyword": topic, "yearly_counts": [], "total": 0, "trend": "unknown"}
    yearly_counts = trend_from_api.get("yearly_counts") or []
    total_pubs = int(trend_from_api.get("total", 0) or 0)
    trend_raw = (trend_from_api.get("trend") or "unknown").lower()
    if total_pubs >= 20000 or trend_raw == "growing":
        trend_status = "HOT"
    elif total_pubs >= 5000:
        trend_status = "STABLE"
    elif total_pubs > 0:
        trend_status = "EMERGING"
    else:
        trend_status = "UNKNOWN"

    if not merged_papers:
        fallback_msg = "Insufficient academic trend data"
        trend_data = {
            "keyword": topic_data.get("semantic_query", topic),
            "warning_message": fallback_msg,
            "overall_trends": {"keyword": topic_data.get("semantic_query", topic), "yearly_counts": yearly_counts, "total": total_pubs, "trend": trend_status.lower()},
            "main_trending_keywords": [],
            "publication_count": total_pubs,
            "render": total_pubs > 0,
            "topic_status": trend_status,
            "insufficient_data": True,
        }
        suggestions = [{
            "category": "Academic Matching",
            "priority": "Medium",
            "suggestion": fallback_msg,
            "details": "External scholarly APIs returned no strong matches for this semantic query."
        }]
    else:
        trend_data = {
            "keyword": topic_data.get("semantic_query", topic),
            "warning_message": "",
            "overall_trends": {
                "keyword": topic_data.get("semantic_query", topic),
                "yearly_counts": yearly_counts,
                "total": total_pubs,
                "trend": trend_status.lower(),
            },
            "main_trending_keywords": [],
            "publication_count": total_pubs,
            "render": total_pubs > 0,
            "topic_status": trend_status,
            "insufficient_data": total_pubs <= 0,
        }
        if total_pubs == 0:
            trend_data["warning_message"] = "Insufficient academic trend data"

        suggestions = generate_research_suggestions(
            metadata,
            merged_papers,
            [],
            trend_data.get("overall_trends", {}),
            missing_citations,
            novelty_score,
            quality_scores,
        )
        if not suggestions:
            suggestions = [{
                "category": "Academic Matching",
                "priority": "Medium",
                "suggestion": "Limited academic comparison data available",
                "details": "Try uploading a paper with a clearer abstract/title for better discovery."
            }]
    abstract_texts = [p.get("abstract", "") for p in merged_papers if (p.get("abstract") or "").strip()]
    ranked_overlap = []
    for p in merged_papers:
        score = _tfidf_cosine_similarity(document_text[:5000], p.get("abstract", ""))
        ranked_overlap.append((p, score))
    ranked_overlap.sort(key=lambda x: x[1], reverse=True)
    similarity_percent = round((sum(s for _, s in ranked_overlap) / len(ranked_overlap)) * 100, 1) if ranked_overlap else 0.0
    logger.info("[SIMILARITY SCORE] aggregate_overlap=%.1f compared=%d", similarity_percent, len(abstract_texts))

    overlap_analysis = {
        "enabled": bool(merged_papers),
        "status": "insufficient_reference_data" if not merged_papers else "available",
        "paper_count": len(merged_papers),
        "similarity_percent": similarity_percent,
        "plagiarism_message": "Limited academic comparison data available" if not merged_papers else f"Estimated overlap against scholarly abstracts: {similarity_percent}%",
        "compared_abstracts": len(abstract_texts),
        "confidence": "low" if len(abstract_texts) < 2 else ("medium" if len(abstract_texts) < 5 else "high"),
    }

    elapsed = round(time.time() - start_time, 2)
    final = {
        "metadata":           metadata,
        "similar_papers":     merged_papers,
        "influential_papers": ss_influential,
        "oa_works":           oa_works,
        "quality_index":      quality_scores,
        "novelty_score":      novelty_score,
        "trends":             trend_data,
        "related_concepts":   oa_concepts,
        "missing_citations":  missing_citations,
        "suggestions":        suggestions,
        "overlap_analysis":   overlap_analysis,
        "warnings":           warnings,
        "generated_query":    topic,
        "clean_topic_title":  topic_data.get("title") or topic,
        "extracted_topic":    topic_data.get("semantic_query", topic),
        "topic_extraction":   topic_data,
        "fallback_query_used": used_ss_query or used_oa_query or clean_search_query(topic),
        "domain": topic_data.get("domain", "Computer Science"),
        "subdomain": topic_data.get("subdomain", "General"),
        "confidence": topic_data.get("confidence", 0.0),
        "fallback_method": topic_data.get("fallback_method", "none"),
        "papers_searched":    len(merged_papers),
        "success":            True,
        "analysis_time_s":    elapsed,
        "status":             "success",
    }

    logger.info("[RESEARCH] Analysis completed successfully")
    logger.info("[TOPIC DEBUG] API response count=%d", len(merged_papers))
    logger.info("[TOPIC DEBUG] fallback method used=%s", topic_data.get("fallback_method", "none"))
    await cache_manager.set(doc_hash, final, category="academic_analysis")
    return final
