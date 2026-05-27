"""
openalex.py
===========
OpenAlex Academic API integration (free, no API key required).

Provides:
  - search_works(query, limit)         : search academic works
  - get_topic_trends(keyword)          : yearly publication counts
  - get_related_concepts(topic)        : concept/topic graph
  - get_open_access_info(doi)          : open access status
  - get_author_info(author_name)       : author metadata

All functions are synchronous with in-memory TTL caching.
API failures return empty/default structures — never raise to caller.
"""

import time
import hashlib
import logging
import asyncio
import re
import html as html_lib
from typing import Optional
from datetime import datetime

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
BASE_URL    = "https://api.openalex.org"
TIMEOUT     = 15
MAX_RETRIES = 2
# Polite pool: add your email to get higher rate limits
MAILTO      = "researchbot@example.com"

# ─────────────────────────────────────────────
# 🌐  HTTP HELPER (ASYNC)
# ─────────────────────────────────────────────
async def _get_async(url: str, params: dict = None) -> Optional[dict]:
    """Make an async GET request and return parsed JSON or None on failure."""
    if not _HTTPX_AVAILABLE:
        logger.error("httpx not available for async requests")
        return None

    p = dict(params or {})
    p["mailto"] = MAILTO

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(url, params=p)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                logger.warning(f"OpenAlex Rate Limited: {url}")
                raise Exception("Rate limited")
            else:
                logger.warning(f"OpenAlex API {r.status_code}: {url}")
                return None
    except Exception as e:
        logger.warning(f"OpenAlex request failed: {e}")
        raise

# Wrap with retry
_get_async_with_retry = retry_async(_get_async, max_retries=MAX_RETRIES)

# ─────────────────────────────────────────────
# 🔧  DATA NORMALIZERS
# ─────────────────────────────────────────────
def _strip_html(text: str) -> str:
    """Clean HTML from paper text."""
    if not text: return ""
    clean = re.sub(r"<[^>]+>", " ", str(text))
    clean = html_lib.unescape(clean)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean

def _normalize_work(raw: dict) -> dict:
    """Convert raw OpenAlex work object to a clean dict."""
    if not raw:
        return {}

    # Authors
    authorships = raw.get("authorships") or []
    authors = []
    for a in authorships[:5]:
        author = a.get("author") or {}
        name = author.get("display_name", "")
        if name:
            authors.append(name)

    # Concepts / topics
    concepts = raw.get("concepts") or []
    concept_names = [c.get("display_name", "") for c in concepts[:8] if c.get("score", 0) > 0.3]

    # Open access
    oa = raw.get("open_access") or {}
    is_oa = oa.get("is_oa", False)
    oa_url = oa.get("oa_url", "")

    # DOI
    doi = raw.get("doi", "") or ""
    if doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    # Publication year
    year = raw.get("publication_year")

    # Citation count
    cited_by = raw.get("cited_by_count", 0) or 0

    # Venue / journal
    primary_location = raw.get("primary_location") or {}
    source = primary_location.get("source") or {}
    venue = source.get("display_name", "")

    # CLEAN HTML FROM ABSTRACT
    abstract = _reconstruct_abstract(raw.get("abstract_inverted_index"))
    abstract = _strip_html(abstract)

    return {
        "work_id":        raw.get("id", ""),
        "title":          _strip_html(raw.get("display_name", raw.get("title", "Untitled"))),
        "abstract":       abstract[:600],
        "year":           year,
        "citation_count": cited_by,
        "authors":        authors,
        "venue":          venue,
        "concepts":       concept_names,
        "is_open_access": is_oa,
        "open_access_url": oa_url,
        "doi":            doi,
        "url":            raw.get("id", ""),   # OpenAlex URL
        "type":           raw.get("type", ""),
        "language":       raw.get("language", ""),
    }


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    try:
        # inverted_index: {word: [position, ...], ...}
        positions = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word
        if not positions:
            return ""
        words = [positions[i] for i in sorted(positions.keys())]
        abstract = " ".join(words)
        return abstract
    except Exception:
        return ""


# ─────────────────────────────────────────────
# 🔍  PUBLIC API FUNCTIONS (ASYNC)
# ─────────────────────────────────────────────

async def search_works(query: str, limit: int = 10) -> list[dict]:
    """
    Search OpenAlex for academic works matching the query.
    """
    if not query or not query.strip():
        return []

    ckey = f"oa_search:{query.strip().lower()}:{limit}"
    cached = await cache_manager.get(ckey, category="openalex")
    if cached is not None:
        return cached

    url = f"{BASE_URL}/works"
    params = {
        "search":        query.strip(),
        "per-page":      min(limit, 25),
        "select":        (
            "id,display_name,title,abstract_inverted_index,publication_year,"
            "cited_by_count,authorships,concepts,open_access,doi,"
            "primary_location,type,language"
        ),
        "sort":          "cited_by_count:desc",
    }

    try:
        data = await _get_async_with_retry(url, params)
        if not data: return []

        works = [_normalize_work(w) for w in (data.get("results") or []) if w]
        await cache_manager.set(ckey, works, category="openalex")
        return works
    except Exception:
        return []


async def get_topic_trends(keyword: str, years: int = 10) -> dict:
    """
    Get yearly publication counts for a keyword.
    """
    if not keyword:
        return {"keyword": keyword, "yearly_counts": [], "total": 0, "trend": "unknown"}

    ckey = f"oa_trends:{keyword.strip().lower()}:{years}"
    cached = await cache_manager.get(ckey, category="openalex")
    if cached is not None:
        return cached

    current_year = datetime.now().year
    start_year   = current_year - years

    url = f"{BASE_URL}/works"
    params = {
        "search":   keyword.strip(),
        "group_by": "publication_year",
        "per-page": 200,
    }

    try:
        data = await _get_async_with_retry(url, params)
        yearly_counts = []

        if data:
            group_by = data.get("group_by") or []
            year_map = {}
            for item in group_by:
                try:
                    yr = int(item.get("key", 0))
                    cnt = int(item.get("count", 0))
                    if start_year <= yr <= current_year:
                        year_map[yr] = cnt
                except (ValueError, TypeError):
                    continue

            for yr in range(start_year, current_year + 1):
                yearly_counts.append({"year": yr, "count": year_map.get(yr, 0)})

        total = sum(y["count"] for y in yearly_counts)

        # Determine trend
        trend = "stable"
        if len(yearly_counts) >= 6:
            recent = sum(y["count"] for y in yearly_counts[-3:])
            older  = sum(y["count"] for y in yearly_counts[-6:-3])
            if older > 0:
                ratio = recent / older
                if ratio > 1.2:
                    trend = "growing"
                elif ratio < 0.8:
                    trend = "declining"

        result = {
            "keyword":       keyword,
            "yearly_counts": yearly_counts,
            "total":         total,
            "trend":         trend,
        }
        await cache_manager.set(ckey, result, category="openalex")
        return result
    except Exception:
        return {"keyword": keyword, "yearly_counts": [], "total": 0, "trend": "unknown"}


async def get_related_concepts(topic: str, limit: int = 10) -> list[dict]:
    if not topic: return []

    ckey = f"oa_concepts:{topic.strip().lower()}:{limit}"
    cached = await cache_manager.get(ckey, category="openalex")
    if cached is not None: return cached

    url = f"{BASE_URL}/concepts"
    params = {
        "search":   topic.strip(),
        "per-page": min(limit, 25),
        "select":   "id,display_name,level,works_count,description,related_concepts",
    }

    try:
        data = await _get_async_with_retry(url, params)
        if not data: return []

        concepts = []
        for c in (data.get("results") or [])[:limit]:
            concepts.append({
                "concept_id":  c.get("id", ""),
                "name":        c.get("display_name", ""),
                "level":       c.get("level", 0),
                "works_count": c.get("works_count", 0),
                "description": (c.get("description") or "")[:200],
            })

        await cache_manager.set(ckey, concepts, category="openalex")
        return concepts
    except Exception:
        return []


async def get_open_access_info(doi: str) -> dict:
    if not doi: return {"is_open_access": False, "oa_url": "", "license": ""}

    ckey = f"oa_doi:{doi.strip()}"
    cached = await cache_manager.get(ckey, category="openalex")
    if cached is not None: return cached

    clean_doi = doi.strip()
    if clean_doi.startswith("https://doi.org/"):
        clean_doi = clean_doi[len("https://doi.org/"):]

    url = f"{BASE_URL}/works/https://doi.org/{clean_doi}"
    params = {"select": "open_access,primary_location"}

    try:
        data = await _get_async_with_retry(url, params)
        result = {"is_open_access": False, "oa_url": "", "license": ""}

        if data:
            oa = data.get("open_access") or {}
            result["is_open_access"] = oa.get("is_oa", False)
            result["oa_url"]         = oa.get("oa_url", "") or ""
            result["license"]        = oa.get("license", "") or ""

        await cache_manager.set(ckey, result, category="openalex")
        return result
    except Exception:
        return {"is_open_access": False, "oa_url": "", "license": ""}


async def get_author_info(author_name: str) -> list[dict]:
    if not author_name: return []

    ckey = f"oa_author:{author_name.strip().lower()}"
    cached = await cache_manager.get(ckey, category="openalex")
    if cached is not None: return cached

    url = f"{BASE_URL}/authors"
    params = {
        "search":   author_name.strip(),
        "per-page": 5,
        "select":   "id,display_name,works_count,cited_by_count,last_known_institution,x_concepts",
    }

    try:
        data = await _get_async_with_retry(url, params)
        if not data: return []

        authors = []
        for a in (data.get("results") or [])[:5]:
            inst = a.get("last_known_institution") or {}
            concepts = a.get("x_concepts") or []
            top_concepts = [c.get("display_name", "") for c in concepts[:3]]
            authors.append({
                "author_id":      a.get("id", ""),
                "name":           a.get("display_name", ""),
                "works_count":    a.get("works_count", 0),
                "citation_count": a.get("cited_by_count", 0),
                "institution":    inst.get("display_name", ""),
                "top_concepts":   top_concepts,
            })

        await cache_manager.set(ckey, authors, category="openalex")
        return authors
    except Exception:
        return []


async def get_trending_keywords(field: str = "computer science", top_n: int = 10) -> list[dict]:
    ckey = f"oa_trending:{field.lower()}:{top_n}"
    cached = await cache_manager.get(ckey, category="openalex")
    if cached is not None: return cached

    concepts = await get_related_concepts(field, limit=top_n * 2)
    concepts.sort(key=lambda c: c.get("works_count", 0), reverse=True)

    result = [
        {
            "keyword":     c["name"],
            "works_count": c["works_count"],
            "description": c.get("description", ""),
        }
        for c in concepts[:top_n]
    ]

    await cache_manager.set(ckey, result, category="openalex")
    return result
