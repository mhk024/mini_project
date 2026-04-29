"""
semantic_scholar.py
===================
Semantic Scholar Academic Graph API integration.

Provides:
  - search_papers(query, limit)       : keyword/title search
  - get_paper_details(paper_id)       : full paper metadata
  - get_similar_papers(title, abstract): find related papers
  - get_citations(paper_id, limit)    : papers that cite this paper
  - get_references(paper_id, limit)   : papers cited by this paper

All functions are synchronous (use httpx with timeout).
Results are cached in-memory with a 1-hour TTL.
API failures return empty/default structures — never raise to caller.
"""

import time
import hashlib
import logging
import asyncio
import re
import html as html_lib
from typing import Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

import json
from services.cache_manager import cache_manager
from services.async_utils import retry_async

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────
BASE_URL   = "https://api.semanticscholar.org/graph/v1"
TIMEOUT    = 15          # seconds per request
MAX_RETRIES = 2

# Fields to request for paper objects
PAPER_FIELDS = (
    "paperId,title,abstract,year,citationCount,referenceCount,"
    "authors,venue,publicationTypes,openAccessPdf,externalIds,"
    "fieldsOfStudy,influentialCitationCount,publicationDate"
)

CITATION_FIELDS = "paperId,title,abstract,year,citationCount,authors,venue"

# ─────────────────────────────────────────────
# 🌐  HTTP HELPER (ASYNC)
# ─────────────────────────────────────────────
async def _get_async(url: str, params: dict = None) -> Optional[dict]:
    """Make an async GET request and return parsed JSON or None on failure."""
    if not _HTTPX_AVAILABLE:
        logger.error("httpx not available for async requests")
        return None

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(url, params=params or {})
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                logger.warning(f"Semantic Scholar Rate Limited: {url}")
                raise Exception("Rate limited")
            else:
                logger.warning(f"Semantic Scholar API {r.status_code}: {url}")
                return None
    except Exception as e:
        logger.warning(f"Semantic Scholar request failed: {e}")
        raise

# Wrap with retry
_get_async_with_retry = retry_async(_get_async, max_retries=MAX_RETRIES)

# ─────────────────────────────────────────────
# 🔧  DATA NORMALIZER
# ─────────────────────────────────────────────
def _strip_html(text: str) -> str:
    """Clean HTML from paper text."""
    if not text: return ""
    clean = re.sub(r"<[^>]+>", " ", str(text))
    clean = html_lib.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def _normalize_paper(raw: dict) -> dict:
    """Convert raw API paper object to a clean, consistent dict."""
    if not raw:
        return {}
    authors = raw.get("authors") or []
    author_names = [a.get("name", "") for a in authors[:5]]

    pdf_url = None
    oa = raw.get("openAccessPdf")
    if isinstance(oa, dict):
        pdf_url = oa.get("url")

    # CLEAN HTML FROM ABSTRACT
    abstract = _strip_html(raw.get("abstract") or "")

    return {
        "paper_id":            raw.get("paperId", ""),
        "title":               _strip_html(raw.get("title", "Untitled")),
        "abstract":            abstract[:600],
        "year":                raw.get("year"),
        "citation_count":      raw.get("citationCount", 0) or 0,
        "reference_count":     raw.get("referenceCount", 0) or 0,
        "influential_citations": raw.get("influentialCitationCount", 0) or 0,
        "authors":             author_names,
        "venue":               raw.get("venue", ""),
        "fields_of_study":     raw.get("fieldsOfStudy") or [],
        "publication_types":   raw.get("publicationTypes") or [],
        "open_access_pdf":     pdf_url,
        "external_ids":        raw.get("externalIds") or {},
        "publication_date":    raw.get("publicationDate", ""),
        "url":                 f"https://www.semanticscholar.org/paper/{raw.get('paperId', '')}",
    }


# ─────────────────────────────────────────────
# 🔍  PUBLIC API FUNCTIONS (ASYNC)
# ─────────────────────────────────────────────

async def search_papers(query: str, limit: int = 10) -> list[dict]:
    """
    Search Semantic Scholar for papers matching the query.
    """
    if not query or not query.strip():
        return []

    ckey = f"search:{query.strip().lower()}:{limit}"
    cached = await cache_manager.get(ckey, category="semantic_scholar")
    if cached is not None:
        return cached

    url = f"{BASE_URL}/paper/search"
    params = {
        "query":  query.strip(),
        "limit":  min(limit, 25),
        "fields": PAPER_FIELDS,
    }

    try:
        data = await _get_async_with_retry(url, params)
        if not data: return []
        
        papers = [_normalize_paper(p) for p in (data.get("data") or []) if p]
        await cache_manager.set(ckey, papers, category="semantic_scholar")
        return papers
    except Exception:
        return []


async def get_paper_details(paper_id: str) -> dict:
    """
    Fetch full details for a specific paper.
    """
    if not paper_id:
        return {}

    ckey = f"details:{paper_id}"
    cached = await cache_manager.get(ckey, category="semantic_scholar")
    if cached is not None:
        return cached

    url = f"{BASE_URL}/paper/{paper_id}"
    params = {"fields": PAPER_FIELDS}

    try:
        data = await _get_async_with_retry(url, params)
        result = _normalize_paper(data) if data else {}
        await cache_manager.set(ckey, result, category="semantic_scholar")
        return result
    except Exception:
        return {}


async def get_similar_papers(title: str, abstract: str = "", limit: int = 10) -> list[dict]:
    """
    Find papers similar to the given title/abstract.
    """
    if not title:
        return []

    ckey = f"similar:{title.strip().lower()}:{limit}"
    cached = await cache_manager.get(ckey, category="semantic_scholar")
    if cached is not None:
        return cached

    seen_ids = set()
    results = []

    # Pass 1: search by title
    title_results = await search_papers(title, limit=limit)
    for p in title_results:
        pid = p.get("paper_id", "")
        if pid and pid not in seen_ids:
            seen_ids.add(pid)
            results.append(p)

    # Pass 2: if abstract provided, extract key terms and search again
    if abstract and len(results) < limit:
        # Simple keyword extraction
        words = [w for w in abstract.split() if len(w) > 5][:8]
        if words:
            kw_query = " ".join(words[:5])
            kw_results = await search_papers(kw_query, limit=limit)
            for p in kw_results:
                pid = p.get("paper_id", "")
                if pid and pid not in seen_ids:
                    seen_ids.add(pid)
                    results.append(p)
                    if len(results) >= limit:
                        break

    final = results[:limit]
    await cache_manager.set(ckey, final, category="semantic_scholar")
    return final


async def get_citations(paper_id: str, limit: int = 10) -> list[dict]:
    if not paper_id: return []

    ckey = f"citations:{paper_id}:{limit}"
    cached = await cache_manager.get(ckey, category="semantic_scholar")
    if cached is not None: return cached

    url = f"{BASE_URL}/paper/{paper_id}/citations"
    params = {"fields": CITATION_FIELDS, "limit": min(limit, 50)}

    try:
        data = await _get_async_with_retry(url, params)
        if not data: return []

        papers = []
        for item in (data.get("data") or []):
            citing = item.get("citingPaper") or {}
            p = _normalize_paper(citing)
            if p.get("paper_id"):
                papers.append(p)

        await cache_manager.set(ckey, papers, category="semantic_scholar")
        return papers
    except Exception:
        return []


async def get_references(paper_id: str, limit: int = 10) -> list[dict]:
    if not paper_id: return []

    ckey = f"references:{paper_id}:{limit}"
    cached = await cache_manager.get(ckey, category="semantic_scholar")
    if cached is not None: return cached

    url = f"{BASE_URL}/paper/{paper_id}/references"
    params = {"fields": CITATION_FIELDS, "limit": min(limit, 50)}

    try:
        data = await _get_async_with_retry(url, params)
        if not data: return []

        papers = []
        for item in (data.get("data") or []):
            cited = item.get("citedPaper") or {}
            p = _normalize_paper(cited)
            if p.get("paper_id"):
                papers.append(p)

        await cache_manager.set(ckey, papers, category="semantic_scholar")
        return papers
    except Exception:
        return []


async def get_influential_papers(query: str, limit: int = 5) -> list[dict]:
    papers = await search_papers(query, limit=limit * 2)
    papers.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
    return papers[:limit]
