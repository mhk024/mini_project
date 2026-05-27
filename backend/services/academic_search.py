import logging
from typing import List, Dict
import re

from services.semantic_scholar import search_papers as _ss_search_papers
from services.openalex import search_works as _oa_search_works
from services.cache_manager import cache_manager


logger = logging.getLogger(__name__)


def _tokenize_query(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", (text or "").lower())


def _relevance_score(query: str, title: str, abstract: str) -> float:
    q_tokens = set(_tokenize_query(query))
    if not q_tokens:
        return 0.0
    t_tokens = set(_tokenize_query(f"{title} {abstract}"))
    if not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(1, len(q_tokens))


def _is_low_quality_paper(paper: Dict) -> bool:
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    citations = int(paper.get("citation_count", 0) or 0)
    year = paper.get("year")
    if not title or len(title) < 12:
        return True
    # Keep recent but uncited papers if they still have meaningful abstracts.
    if not abstract and citations == 0 and (not year or int(year) < 2010):
        return True
    return False


async def search_semantic_scholar(topic: str, limit: int = 5) -> List[Dict]:
    """
    Lightweight wrapper for Semantic Scholar search.

    Returns a normalized paper schema:
    {
        "title": str,
        "authors": list,
        "year": int | None,
        "abstract": str,
        "citation_count": int,
        "url": str,
        "source": "SemanticScholar"
    }
    """
    if not topic or not topic.strip():
        return []

    key = f"acad_ss:{topic.strip().lower()}:{limit}"
    cached = await cache_manager.get(key, category="academic_search")
    if cached is not None:
        return cached

    try:
        raw_papers = await _ss_search_papers(topic, limit=limit)
    except Exception as e:
        logger.error("Semantic Scholar search failed for topic %r: %s", topic, e)
        await cache_manager.set(key, [], category="academic_search")
        return []

    norm: List[Dict] = []
    for p in raw_papers[:limit]:
        if not p:
            continue
        item = {
            "title": p.get("title", "") or "",
            "authors": p.get("authors", []) or [],
            "year": p.get("year"),
            "abstract": p.get("abstract", "") or "",
            "citation_count": int(p.get("citation_count", 0) or 0),
            "url": p.get("url", "") or "",
            "source": "SemanticScholar",
            "source_api": "Semantic Scholar",
        }
        if _is_low_quality_paper(item):
            continue
        item["relevance_score"] = round(_relevance_score(topic, item["title"], item["abstract"]), 3)
        norm.append(item)

    norm.sort(
        key=lambda p: (int(p.get("citation_count", 0) or 0), float(p.get("relevance_score", 0.0))),
        reverse=True,
    )

    await cache_manager.set(key, norm, category="academic_search")
    return norm


async def search_openalex(topic: str, limit: int = 5) -> List[Dict]:
    """
    Lightweight wrapper for OpenAlex search.

    Returns the same normalized schema as `search_semantic_scholar`, with
    source="OpenAlex".
    """
    if not topic or not topic.strip():
        return []

    key = f"acad_oa:{topic.strip().lower()}:{limit}"
    cached = await cache_manager.get(key, category="academic_search")
    if cached is not None:
        return cached

    try:
        raw_works = await _oa_search_works(topic, limit=limit)
    except Exception as e:
        logger.error("OpenAlex search failed for topic %r: %s", topic, e)
        await cache_manager.set(key, [], category="academic_search")
        return []

    norm: List[Dict] = []
    for w in raw_works[:limit]:
        if not w:
            continue
        item = {
            "title": w.get("title", "") or "",
            "authors": w.get("authors", []) or [],
            "year": w.get("year"),
            "abstract": w.get("abstract", "") or "",
            "citation_count": int(w.get("citation_count", 0) or 0),
            "url": (w.get("open_access_url") or w.get("url") or ""),
            "source": "OpenAlex",
            "source_api": "OpenAlex",
            "concepts": w.get("concepts", []) or [],
            "open_access_url": w.get("open_access_url", "") or "",
        }
        if _is_low_quality_paper(item):
            continue
        item["relevance_score"] = round(_relevance_score(topic, item["title"], item["abstract"]), 3)
        norm.append(item)

    norm.sort(
        key=lambda p: (int(p.get("citation_count", 0) or 0), float(p.get("relevance_score", 0.0))),
        reverse=True,
    )

    await cache_manager.set(key, norm, category="academic_search")
    return norm


def merge_results(primary: List[Dict], secondary: List[Dict], limit: int = 8) -> List[Dict]:
    """
    Merge + deduplicate paper lists from multiple sources by (title, year).
    Preference is given to `primary` ordering, then `secondary`.
    """
    seen = set()
    merged: List[Dict] = []

    def _add_many(items):
        for p in items:
            title = (p.get("title") or "").strip().lower()
            year = p.get("year") or 0
            key = (title, year)
            if not title or key in seen:
                continue
            seen.add(key)
            merged.append(p)
            if len(merged) >= limit:
                break

    _add_many(primary)
    if len(merged) < limit:
        _add_many(secondary)

    # Sort by citation_count desc, then year desc
    merged.sort(
        key=lambda p: (
            int(p.get("citation_count", 0) or 0),
            p.get("year") or 0,
        ),
        reverse=True,
    )
    return merged[: limit]

