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
from datetime import datetime
from typing import Optional

from services.semantic_scholar import get_influential_papers
from services.openalex import get_related_concepts
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


def extract_research_topic(document_text: str) -> str:
    metadata = extract_paper_metadata(document_text)
    title = (metadata.get("title") or "").strip()
    abstract = (metadata.get("abstract") or "").strip()
    first_para = _extract_first_meaningful_paragraph(document_text)

    title_candidates = _extract_title_candidates(document_text)
    candidates = title_candidates + [title, abstract[:180], first_para[:180]]
    for candidate in candidates:
        if not candidate:
            continue
        # Strip "Abstract:" prefix when present.
        clean = re.sub(r"^\s*abstract\s*[:\-]?\s*", "", candidate, flags=re.IGNORECASE).strip()
        if _looks_like_noise_line(clean):
            continue
        topic = _clean_topic_phrase(clean, max_words=12)
        wc = len(topic.split())
        if topic and 3 <= wc <= 12 and not _is_weak_topic(topic):
            return topic

    fallback_keywords = _top_keywords_from_texts([title, abstract, first_para], top_n=5)
    if fallback_keywords:
        return _clean_topic_phrase(" ".join(fallback_keywords), max_words=8)
    return _clean_topic_phrase(title, max_words=10) or "Research topic"


def clean_search_query(topic: str, max_chars: int = 80) -> str:
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
    # preserve order and uniqueness
    seen = set()
    uniq = []
    for q in fallbacks:
        k = q.lower()
        if k and k not in seen:
            seen.add(k)
            uniq.append(q[:80])
    return uniq[:4]


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
    topic = extract_research_topic(document_text)
    logger.info("[TOPIC] Extracted topic: %s", topic)
    fallback_queries = _build_fallback_queries(topic)

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

    used_ss_query = ""
    for q in fallback_queries:
        logger.info("[SEARCH] Semantic Scholar query: %s", q)
        used_ss_query = q
        try:
            ss_similar = await run_with_timeout(
                search_semantic_scholar(q, limit=5),
                10,
                fallback_value=[],
            ) or []
        except Exception:
            logger.exception("[RESEARCH] Semantic Scholar search failed for query=%s", q)
            ss_similar = []
        if ss_similar:
            break

    used_oa_query = ""
    for q in fallback_queries:
        logger.info("[SEARCH] OpenAlex query: %s", q)
        used_oa_query = q
        try:
            oa_similar = await run_with_timeout(
                search_openalex(q, limit=5),
                10,
                fallback_value=[],
            ) or []
        except Exception:
            logger.exception("[RESEARCH] OpenAlex search failed for query=%s", q)
            oa_similar = []
        if oa_similar:
            break

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
    similar_papers = ss_similar + oa_similar
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

    merged_papers = sorted(
        normalized_papers,
        key=lambda p: (int(p.get("citation_count", 0) or 0), float(p.get("relevance_score", 0.0))),
        reverse=True,
    )[:8]
    oa_works = oa_similar
    logger.info("[SEARCH] Papers found: %d", len(merged_papers))

    if not merged_papers:
        warnings.append("External academic APIs unavailable")

    quality_scores = compute_enhanced_quality(metadata, merged_papers, [])
    novelty_score = quality_scores["novelty_score"]
    missing_citations = find_missing_citations(metadata, merged_papers, ss_influential)

    # Build simple trends from fetched papers only
    year_counts = {}
    for p in merged_papers:
        y = p.get("year")
        if not y:
            continue
        year_counts[y] = year_counts.get(y, 0) + 1
    yearly_counts = [
        {"year": y, "count": year_counts[y]} for y in sorted(year_counts.keys())
    ]
    total_pubs = sum(y["count"] for y in yearly_counts)

    if not merged_papers:
        fallback_msg = "No strong academic matches found."
        trend_data = {
            "keyword": topic,
            "warning_message": fallback_msg,
            "overall_trends": {"keyword": topic, "yearly_counts": [], "total": 0, "trend": "unknown"},
            "main_trending_keywords": [],
            "publication_count": 0,
            "render": False,
        }
        suggestions = [{
            "category": "Academic Matching",
            "priority": "Medium",
            "suggestion": fallback_msg,
            "details": "External scholarly APIs returned no strong matches for this topic."
        }]
    else:
        trend_label = "unknown"
        if len(yearly_counts) >= 2:
            first = yearly_counts[0]["count"]
            last = yearly_counts[-1]["count"]
            if first > 0:
                ratio = last / first
                if ratio > 1.2:
                    trend_label = "growing"
                elif ratio < 0.8:
                    trend_label = "declining"
                else:
                    trend_label = "stable"

        trend_data = {
            "keyword": topic,
            "warning_message": "",
            "overall_trends": {
                "keyword": topic,
                "yearly_counts": yearly_counts,
                "total": total_pubs,
                "trend": trend_label,
            },
            "main_trending_keywords": [],
            "publication_count": total_pubs,
            "render": total_pubs > 0,
        }
        if total_pubs == 0:
            trend_data["warning_message"] = "Not enough academic data available for trend analysis."

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
                "suggestion": "No strong academic matches found.",
                "details": "Try uploading a paper with a clearer abstract/title for better discovery."
            }]
    abstract_texts = [p.get("abstract", "") for p in merged_papers if (p.get("abstract") or "").strip()]
    ranked_overlap = []
    for p in merged_papers:
        score = _tfidf_cosine_similarity(document_text[:5000], p.get("abstract", ""))
        ranked_overlap.append((p, score))
    ranked_overlap.sort(key=lambda x: x[1], reverse=True)
    similarity_percent = round((sum(s for _, s in ranked_overlap) / len(ranked_overlap)) * 100, 1) if ranked_overlap else 0.0
    logger.info("[SIMILARITY] Compared against %d abstracts", len(abstract_texts))

    overlap_analysis = {
        "enabled": bool(merged_papers),
        "status": "insufficient_reference_data" if not merged_papers else "available",
        "paper_count": len(merged_papers),
        "similarity_percent": similarity_percent,
        "plagiarism_message": "Low confidence — insufficient comparison papers" if not merged_papers else f"Estimated overlap: {similarity_percent}%",
        "compared_abstracts": len(abstract_texts),
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
        "extracted_topic":    topic,
        "fallback_query_used": used_ss_query or used_oa_query or clean_search_query(topic),
        "papers_searched":    len(merged_papers),
        "success":            True,
        "analysis_time_s":    elapsed,
        "status":             "success",
    }

    logger.info("[RESEARCH] Analysis completed successfully")
    await cache_manager.set(doc_hash, final, category="academic_analysis")
    return final
