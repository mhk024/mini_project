import logging
from typing import List, Dict

from services.semantic_scholar import search_papers as _ss_search_papers
from services.openalex import search_works as _oa_search_works
from services.cache_manager import cache_manager


logger = logging.getLogger(__name__)


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
        norm.append(
            {
                "title": p.get("title", "") or "",
                "authors": p.get("authors", []) or [],
                "year": p.get("year"),
                "abstract": p.get("abstract", "") or "",
                "citation_count": int(p.get("citation_count", 0) or 0),
                "url": p.get("url", "") or "",
                "source": "SemanticScholar",
            }
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
        norm.append(
            {
                "title": w.get("title", "") or "",
                "authors": w.get("authors", []) or [],
                "year": w.get("year"),
                "abstract": w.get("abstract", "") or "",
                "citation_count": int(w.get("citation_count", 0) or 0),
                "url": w.get("url", "") or "",
                "source": "OpenAlex",
            }
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

