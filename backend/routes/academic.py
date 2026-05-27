import re
import os
import time
import logging
import asyncio
from typing import Optional, List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.academic_intelligence import (
    run_full_academic_analysis_async,
    extract_paper_metadata,
    compute_enhanced_quality,
    compute_novelty_score,
    get_domain_aware_trends_async,
)
from services.semantic_scholar import (
    get_similar_papers,
)
from services.openalex import (
    search_works,
    get_related_concepts,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Academic Intelligence"])

DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "dataset"))

# ─────────────────────────────────────────────
# 📦  REQUEST MODELS
# ─────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    filename: Optional[str] = None
    query:    Optional[str] = None

class PerQuestionEvaluationRequest(BaseModel):
    questions: List[str]
    filename:  Optional[str] = None
    context:   Optional[str] = ""

# ─────────────────────────────────────────────
# 🔧  SHARED HELPER
# ─────────────────────────────────────────────
async def _get_document_text_async(filename: Optional[str] = None) -> str:
    try:
        from research_analyzer import get_document_text
        import state

        if state.vector_db is not None:
            text = await asyncio.to_thread(get_document_text, state.vector_db, max_chars=5000)
            if text:
                return text
    except Exception:
        pass

    if not filename:
        raise HTTPException(400, "No document loaded.")

    file_path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, f"File {filename} not found.")

    # Async read
    if filename.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        docs = await asyncio.to_thread(PyPDFLoader(file_path).load)
        return "\n\n".join(d.page_content for d in docs)[:5000]
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:5000]

# ─────────────────────────────────────────────
# 🔍  ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/similar-papers")
async def similar_papers_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        metadata = extract_paper_metadata(text)
        title = request.query or metadata.get("title", "")
        
        # Parallel fetch
        ss_task = get_similar_papers(title, metadata.get("abstract", ""), limit=10)
        oa_task = search_works(title, limit=10)
        
        ss_papers, oa_papers = await asyncio.gather(ss_task, oa_task)
        
        merged = ss_papers + oa_papers
        return {"status": "success", "papers": merged[:15]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar papers error: {e}")
        raise HTTPException(500, str(e))

@router.post("/quality-index")
async def quality_index_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        metadata = extract_paper_metadata(text)
        title = request.query or metadata.get("title", "")
        
        ss_papers = await get_similar_papers(title, metadata.get("abstract", ""), limit=5)
        oa_works = await search_works(title, limit=5)
        
        scores = compute_enhanced_quality(metadata, ss_papers, oa_works)
        return {"status": "success", "quality_index": scores}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quality index error: {e}")
        raise HTTPException(500, str(e))

@router.post("/trends")
async def trends_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        metadata = extract_paper_metadata(text)
        keyword = request.query or (metadata.get("keywords") or [""])[0] or metadata.get("title", "")
        
        trends_task = get_domain_aware_trends_async(keyword)
        concepts_task = get_related_concepts(keyword, limit=8)
        
        trends, concepts = await asyncio.gather(trends_task, concepts_task)
        return {"status": "success", "trends": trends, "related_concepts": concepts}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trends error: {e}")
        raise HTTPException(500, str(e))

@router.post("/full-analysis")
async def full_analysis_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        result = await run_full_academic_analysis_async(text)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        raise HTTPException(500, str(e))

@router.post("/evaluate-questions")
async def evaluate_questions_endpoint(request: PerQuestionEvaluationRequest):
    try:
        from research_analyzer import evaluate_per_question_async
        from services.resource_manager import get_llm

        llm = get_llm()
        text = await _get_document_text_async(request.filename)
        
        result = await evaluate_per_question_async(
            llm, request.questions, text, request.context
        )
        return {"status": "success", "results": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(500, str(e))

@router.post("/suggestions")
async def suggestions_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        res = await run_full_academic_analysis_async(text)
        return {
            "status": "success",
            "suggestions": res.get("suggestions", []),
            "missing_citations": res.get("missing_citations", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        raise HTTPException(500, str(e))

@router.post("/novelty-check")
async def novelty_check_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        res = await run_full_academic_analysis_async(text)
        return {
            "status": "success",
            "novelty_score": res.get("novelty_score"),
            "interpretation": res.get("quality_index", {}).get("interpretation", "")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Novelty error: {e}")
        raise HTTPException(500, str(e))
