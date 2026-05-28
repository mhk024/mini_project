import os
import time
import logging
import asyncio
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.academic_intelligence import (
    run_full_academic_analysis_async,
    extract_paper_metadata,
    extract_research_topic,
    compute_enhanced_quality,
    compute_novelty_score,
    get_domain_aware_trends_async,
    generate_search_query,
)
from services.semantic_scholar import (
    get_similar_papers,
    get_influential_papers,
)
from services.openalex import (
    search_works,
    get_topic_trends,
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
    if not filename:
        # Last-resort fallback to currently loaded vector DB context.
        try:
            from research_analyzer import get_document_text
            import state
            if state.vector_db is not None:
                text = await asyncio.to_thread(get_document_text, state.vector_db, max_chars=12000)
                if text:
                    return text
        except Exception:
            pass
        raise HTTPException(400, "No document loaded.")

    file_path = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, f"File {filename} not found.")

    # Always prioritize raw file extraction for structural fidelity.
    if filename.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        docs = await asyncio.to_thread(PyPDFLoader(file_path).load)
        return "\n\n".join(d.page_content for d in docs)[:20000]
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:20000]


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(value).strip()
    return [s] if s else []


def _to_score(value: Any) -> Optional[float]:
    try:
        score = float(value)
    except Exception:
        return None
    if score < 0:
        score = 0.0
    if score > 10:
        score = 10.0
    return round(score, 1)


def _extract_eval_strengths_weaknesses(evaluations: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    strengths: List[str] = []
    weaknesses: List[str] = []
    ranked = []
    for ev in evaluations:
        if not isinstance(ev, dict):
            continue
        score = _to_score(ev.get("score"))
        ranked.append((score if score is not None else 0.0, ev))
    ranked.sort(key=lambda x: x[0], reverse=True)

    for score, ev in ranked[:3]:
        details = ev.get("details") if isinstance(ev.get("details"), dict) else {}
        analysis = str(details.get("analysis") or ev.get("justification") or "").strip()
        question = str(ev.get("question") or "Question").strip()
        if analysis:
            strengths.append(f"{question}: {analysis[:120]}")
        elif score >= 7:
            strengths.append(f"{question}: Demonstrates clear understanding.")
    for score, ev in ranked[-3:]:
        details = ev.get("details") if isinstance(ev.get("details"), dict) else {}
        gaps = _as_list(details.get("gaps"))
        question = str(ev.get("question") or "Question").strip()
        if gaps:
            weaknesses.append(f"{question}: {gaps[0]}")
        elif score <= 5:
            weaknesses.append(f"{question}: Requires stronger evidence and methodological detail.")

    return strengths[:3], weaknesses[:3]


async def _generate_short_summary(llm: Any, detailed_summary: str) -> str:
    base_text = (detailed_summary or "").strip()
    if not base_text:
        return "This evaluation provides a structured academic assessment of the paper's objectives, evidence quality, and contribution."
    prompt = (
        "Generate a concise academic summary in 3-5 lines from the following research paper analysis. "
        "Use professional tone and include objective, findings, and contribution briefly. "
        "Return plain text only.\n\n"
        f"Analysis:\n{base_text[:3000]}"
    )
    try:
        res = await llm.ainvoke(prompt)
        text = (res.content if hasattr(res, "content") else str(res)).strip()
        if not text:
            return (base_text[:250] + "...") if len(base_text) > 250 else base_text
        lines = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return (base_text[:250] + "...") if len(base_text) > 250 else base_text
        return "\n".join(lines[:5])
    except Exception:
        return (base_text[:250] + "...") if len(base_text) > 250 else base_text


async def _normalize_evaluation_payload(result: Dict[str, Any], questions: List[str], llm: Any) -> Dict[str, Any]:
    data = result if isinstance(result, dict) else {}
    evaluations = data.get("evaluations")
    if not isinstance(evaluations, list):
        evaluations = []

    normalized_evaluations: List[Dict[str, Any]] = []
    for idx, ev in enumerate(evaluations):
        ev = ev if isinstance(ev, dict) else {}
        details = ev.get("details") if isinstance(ev.get("details"), dict) else {}
        gaps = _as_list(details.get("gaps"))
        improvements = _as_list(details.get("improvements"))
        analysis = str(details.get("analysis") or ev.get("justification") or "").strip()
        score = _to_score(ev.get("score"))
        question_text = str(ev.get("question") or (questions[idx] if idx < len(questions) else f"Question {idx + 1}")).strip()
        answer_text = str(ev.get("answer") or "No answer generated.").strip()
        justification = str(ev.get("justification") or analysis or "Detailed justification unavailable.").strip()

        if not gaps:
            gaps = ["Further evidence and methodological details are needed."]
        if not improvements:
            improvements = ["Provide clearer evidence and stronger comparative analysis."]

        normalized_evaluations.append({
            "question": question_text,
            "answer": answer_text,
            "score": score if score is not None else 0.0,
            "justification": justification,
            "details": {
                "analysis": analysis or "No additional analysis available.",
                "gaps": gaps,
                "improvements": improvements,
            },
        })

    summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
    question_scores = [ev.get("score", 0.0) for ev in normalized_evaluations if isinstance(ev.get("score"), (int, float))]
    computed_avg = round(sum(question_scores) / len(question_scores), 1) if question_scores else 0.0
    overall_score = _to_score(summary.get("overall_score"))
    if overall_score is None or overall_score == 0:
        overall_score = computed_avg

    strengths = _as_list(data.get("strengths")) or _as_list(summary.get("key_strengths"))
    weaknesses = _as_list(data.get("weaknesses")) or _as_list(summary.get("key_weaknesses"))
    if not strengths or not weaknesses:
        auto_strengths, auto_weaknesses = _extract_eval_strengths_weaknesses(normalized_evaluations)
        strengths = strengths or auto_strengths or ["Strong conceptual framing and relevant topic coverage."]
        weaknesses = weaknesses or auto_weaknesses or ["More rigorous validation and comparative evidence is needed."]

    detailed_summary = str(data.get("detailed_summary") or data.get("analysis") or "").strip()
    if not detailed_summary:
        top_strength = strengths[0] if strengths else ""
        top_weakness = weaknesses[0] if weaknesses else ""
        detailed_summary = (
            f"The evaluation indicates an overall score of {overall_score}/10. "
            f"Key strength: {top_strength}. Key weakness: {top_weakness}."
        ).strip()
    short_summary = str(data.get("short_summary") or "").strip()
    if not short_summary:
        short_summary = await _generate_short_summary(llm, detailed_summary)

    final_payload = {
        "overall_score": overall_score,
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "evaluations": normalized_evaluations,
        "questions": [str(q).strip() for q in questions if str(q).strip()],
        "short_summary": short_summary,
        "detailed_summary": detailed_summary,
        "summary": {
            "overall_score": overall_score,
            "key_strengths": strengths[:5],
            "key_weaknesses": weaknesses[:5],
        },
    }
    print("Evaluation Response:", final_payload)
    return final_payload

# ─────────────────────────────────────────────
# 🔍  ENDPOINTS
# ─────────────────────────────────────────────

@router.post("/similar-papers")
async def similar_papers_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        # Reuse full academic pipeline so similar-paper retrieval benefits
        # from semantic + keyword + emergency fallback query strategies.
        analysis = await run_full_academic_analysis_async(text)
        papers = (analysis.get("similar_papers") or [])[:15]
        insufficient_data = len(papers) == 0
        return {
            "status": "success",
            "papers": papers,
            "paper_count": len(papers),
            "insufficient_data": insufficient_data,
            "message": "No similar papers found for this query." if insufficient_data else "",
            "extracted_topic": analysis.get("extracted_topic", ""),
            "fallback_query_used": analysis.get("fallback_query_used", ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Similar papers error: {e}")
        raise HTTPException(500, str(e))

@router.post("/quality-index")
async def quality_index_endpoint(request: AnalysisRequest):
    try:
        text = await _get_document_text_async(request.filename)
        topic_data = extract_research_topic(text)
        title = generate_search_query(request.query or topic_data.get("semantic_query") or topic_data.get("title", "") or text[:3000])
        metadata = extract_paper_metadata(text)
        
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
        topic_data = extract_research_topic(text)
        keyword = generate_search_query(request.query or topic_data.get("semantic_query", "") or text[:3000])
        
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
    except Exception:
        logger.exception("Full analysis error")
        return {
            "success": True,
            "status": "success",
            "similar_papers": [],
            "trends": {},
            "suggestions": [],
            "overlap_analysis": {},
            "warnings": ["External academic APIs unavailable"],
        }

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
        normalized = await _normalize_evaluation_payload(result, request.questions, llm)
        return {"status": "success", "results": normalized}
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
