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
from datetime import datetime
from typing import Optional

from services.semantic_scholar import (
    search_papers,
    get_similar_papers,
    get_influential_papers,
)
from services.openalex import (
    search_works,
    get_topic_trends,
    get_related_concepts,
    get_trending_keywords,
)
from services.cache_manager import cache_manager
from services.async_utils import run_with_timeout

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 🌐 DOMAIN EXPANSION CONFIG
# ─────────────────────────────────────────────
GENERIC_TERMS = {"online", "system", "study", "analysis", "review", "approach", "method", "evaluation", "design", "model"}
WEAK_TOPICS = {"online", "paper", "research", "document", "ai", "ml"}

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


def extract_research_topic(document_text: str) -> str:
    metadata = extract_paper_metadata(document_text)
    title = (metadata.get("title") or "").strip()
    abstract = (metadata.get("abstract") or "").strip()
    first_para = _extract_first_meaningful_paragraph(document_text)

    candidates = [title, abstract[:180], first_para[:180]]
    for candidate in candidates:
        if not candidate:
            continue
        # Strip "Abstract:" prefix when present.
        clean = re.sub(r"^\s*abstract\s*[:\-]?\s*", "", candidate, flags=re.IGNORECASE).strip()
        words = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", clean)
        if not words:
            continue
        topic = " ".join(words[:8]).strip()
        if topic and not _is_weak_topic(topic) and len(topic) >= 8:
            return topic

    fallback_keywords = _top_keywords_from_texts([title, abstract], top_n=4)
    if fallback_keywords:
        return " ".join(fallback_keywords)
    return title or "No strong academic matches found."


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
    logger.info(f"RESEARCH_TOPIC={topic}")

    # Parallel tasks
    tasks = {
        "ss_similar":   search_papers(topic, limit=10),
        "ss_influential": get_influential_papers(topic, limit=5),
        "oa_works":     search_works(topic, limit=10),
        "trends":       get_domain_aware_trends_async(topic),
        "concepts":     get_related_concepts(topic, limit=8)
    }

    results = {}
    for name, coro in tasks.items():
        results[name] = await run_with_timeout(coro, 15, fallback_value=[])

    ss_similar = results["ss_similar"] or []
    ss_influential = results["ss_influential"] or []
    oa_works = results["oa_works"] or []
    domain_trends = results["trends"] or {}
    oa_concepts = results["concepts"] or []

    quality_scores = compute_enhanced_quality(metadata, ss_similar, oa_works)
    novelty_score = quality_scores["novelty_score"]
    missing_citations = find_missing_citations(metadata, ss_similar, ss_influential)

    if not ss_similar and not oa_works:
        fallback_msg = "No strong academic matches found."
        domain_trends = {
            "keyword": topic,
            "is_generic": False,
            "warning_message": fallback_msg,
            "overall_trends": {"keyword": topic, "yearly_counts": [], "total": 0, "trend": "unknown"},
            "main_trending_keywords": [],
            "subdomains": [],
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
        total_publications = int((domain_trends.get("overall_trends") or {}).get("total", 0) or 0)
        domain_trends["publication_count"] = total_publications
        domain_trends["render"] = total_publications > 0
        if total_publications <= 0:
            domain_trends["warning_message"] = "No strong academic matches found."

        suggestions = generate_research_suggestions(
            metadata,
            ss_similar,
            oa_works,
            domain_trends.get("overall_trends", {}),
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


    elapsed = round(time.time() - start_time, 2)
    final = {
        "metadata":           metadata,
        "similar_papers":     ss_similar,
        "influential_papers": ss_influential,
        "oa_works":           oa_works,
        "quality_index":      quality_scores,
        "novelty_score":      novelty_score,
        "trends":             domain_trends,
        "related_concepts":   oa_concepts,
        "missing_citations":  missing_citations,
        "suggestions":        suggestions,
        "analysis_time_s":    elapsed,
        "status":             "success",
    }

    await cache_manager.set(doc_hash, final, category="academic_analysis")
    return final
