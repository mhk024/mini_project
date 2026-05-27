import os
import json
import re
import asyncio
import logging
import time
import hashlib
from typing import Optional, List, Dict, Any

from langchain_core.prompts import PromptTemplate
from services.cache_manager import cache_manager
from services.async_utils import run_with_timeout

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 🔬  UNIFIED ANALYSIS PROMPT  (Evaluator v2 — new schema)
# ─────────────────────────────────────────────────────────────
UNIFIED_ANALYSIS_PROMPT = PromptTemplate.from_template("""
You are a research analysis tool. Generate a structured JSON report for the research paper provided.

INPUTS:
- Paper Text: {paper_text}
- Similar Papers: {similar_papers}
- Reference Text (for plagiarism): {reference_text}

TASK:
Evaluate the paper's quality, identify contributions, weaknesses, and provide actionable suggestions. 
If similar papers are provided, use them for comparison. If reference text is provided, perform a plagiarism check.

RULES:
- Respond with ONLY a valid JSON object.
- Do NOT include any markdown formatting (no ```json).
- Be concise and objective.

JSON SCHEMA:
{{
  "scores": {{
    "clarity": <0-10>,
    "novelty": <0-10>,
    "technical_depth": <0-10>,
    "methodology": <0-10>,
    "readability": <0-10>
  }},
  "contributions": ["bullet 1", "bullet 2", "bullet 3"],
  "weaknesses": ["bullet 1", "bullet 2", "bullet 3"],
  "suggestions": ["improvement 1", "improvement 2", "improvement 3"],
  "similar_papers": ["analysis of paper 1", "analysis of paper 2"],
  "plagiarism": "Estimated similarity and overlap analysis"
}}

JSON response:""")



# ─────────────────────────────────────────────────────────────
# 🎓  ACADEMIC EVALUATION ENGINE (Per-Question Basis)
# ─────────────────────────────────────────────────────────────
PER_QUESTION_EVALUATION_PROMPT = PromptTemplate.from_template("""
You are a fast academic evaluator.

Evaluate ALL the following questions in ONE pass using the provided document context.

----------------------------------
INPUT
----------------------------------
- Questions: {questions}
- Paper Content: {paper_text}
- Optional Context: {context}

----------------------------------
CORE TASK
----------------------------------
For EACH question:
1. Provide a concise answer (2–3 lines)
2. Assign a score (0–10)
3. Provide detailed explanation (for expandable UI row)

Analyze the research answer.

Return:
1. strengths
2. weaknesses
3. gaps
4. suggestions

IMPORTANT:
- gaps must NEVER be empty
- return at least 2 concise academic gaps
- focus on missing details, weak evidence, unclear methodology, or missing evaluation

----------------------------------
OUTPUT FORMAT (STRICT JSON)
----------------------------------
{{
  "evaluations": [
    {{
      "question": "...",
      "answer": "...",
      "score": <number 0-10>,
      "details": {{
        "analysis": "...",
        "gaps": ["...", "..."],
        "improvements": ["...", "..."]
      }}
    }}
  ],
  "summary": {{
    "overall_score": <number 0-10>,
    "key_strengths": ["...", "..."],
    "key_weaknesses": ["...", "..."]
  }}
}}

JSON response:
""")

# ─────────────────────────────────────────────────────────────
# 🚀  ASYNC ANALYZERS
# ─────────────────────────────────────────────────────────────

async def analyze_paper_async(
    llm,
    document_text: str,
    reference_text: str = "",
    similar_papers: list = None,
) -> dict:
    if not document_text or not document_text.strip():
        return {"error": "Input text is empty."}

    doc_key = hashlib.md5(document_text[:3000].encode()).hexdigest()
    ckey = f"unified_ana:{doc_key}:{len(reference_text)}"
    cached = await cache_manager.get(ckey, category="llm_analysis")
    if cached: return cached

    paper_excerpt = document_text[:2500]
    ref_excerpt   = reference_text[:1500] if reference_text and reference_text.strip() else "None"
    sim_text      = _format_similar_papers(similar_papers)

    try:
        prompt = UNIFIED_ANALYSIS_PROMPT.format(
            paper_text=paper_excerpt,
            reference_text=ref_excerpt,
            similar_papers=sim_text,
        )
        raw_res = await llm.ainvoke(prompt)
        raw = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
        
        # Robust JSON extraction
        match = re.search(r'\{.*\}', raw.strip(), re.DOTALL)
        if not match: raise ValueError("No JSON")
        
        result = json.loads(match.group())
        
        # Validation & fallback handled here
        result = _validate_analysis_schema(result)
        await cache_manager.set(ckey, result, category="llm_analysis")
        return result

    except Exception as e:
        logger.error(f"Unified analysis failed: {e}")
        return _fallback_result(not reference_text)


async def evaluate_per_question_async(
    llm,
    questions: list,
    paper_text: str,
    context: str = "",
) -> dict:
    if not paper_text or not questions:
        return {"error": "Missing input"}

    q_str = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    paper_excerpt = paper_text[:4000]

    try:
        prompt = PER_QUESTION_EVALUATION_PROMPT.format(
            questions=q_str,
            paper_text=paper_excerpt,
            context=context or "None",
        )
        raw_res = await llm.ainvoke(prompt)
        raw = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
        logger.info("[EVAL RAW OUTPUT] %s", raw[:2000])

        match = re.search(r"\{.*\}", raw.strip(), re.DOTALL)
        if not match:
            raise ValueError("No JSON")

        result = json.loads(match.group())
        normalized = normalize_eval_result(result)
        return normalized
    except Exception as e:
        logger.error("Per-question evaluation failed: %s", e)
        fallback = {"error": str(e), "evaluations": []}
        return normalize_eval_result(fallback)


# ─────────────────────────────────────────────────────────────
# 📈  CORE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

async def run_full_pipeline(
    llm,
    document_text: str,
    mode: str = "Research",
    reference_text: str = "",
) -> dict:
    """
    Run all independent research tasks in parallel with timeouts.
    """
    from services.academic_intelligence import (
        run_full_academic_analysis_async,
        extract_paper_metadata
    )

    logger.info(f"Pipeline started for mode: {mode}")
    start_time = time.time()
    
    # Task 1: Metadata Extraction (Fast)
    metadata = extract_paper_metadata(document_text)
    
    # Run Academic Intelligence (External APIs)
    academic_task = run_full_academic_analysis_async(document_text)
    
    # Run LLM Analysis (Groq) – unified analysis with similar papers
    
    # First, get academic data (similarity, trends)
    academic_data = await run_with_timeout(academic_task, 25, fallback_value={})
    
    similar_papers = academic_data.get("similar_papers", [])
    
    # Task 2: LLM Unified Analysis
    llm_task = analyze_paper_async(
        llm, 
        document_text, 
        reference_text=reference_text, 
        similar_papers=similar_papers
    )
    
    # Task 3: Trend Deep Dive (if mode is Research)
    trends = academic_data.get("trends", {})
    
    # Execute LLM task
    analysis_result = await run_with_timeout(llm_task, 30, fallback_value=_fallback_result(True))
    
    elapsed = round(time.time() - start_time, 2)
    
    return {
        "status": "success",
        "mode": mode,
        "metadata": metadata,
        "academic": academic_data,
        "analysis": analysis_result,
        "time_s": elapsed
    }


# ─────────────────────────────────────────────────────────────
# 🔧  HELPERS
# ─────────────────────────────────────────────────────────────

def _validate_analysis_schema(res: dict) -> dict:
    """Ensure consistency and strip HTML."""
    # Strip HTML from any text fields if they leaked in
    def clean(s):
        if not isinstance(s, str): return s
        return re.sub(r"<[^>]+>", "", s).strip()

    scores = res.get("scores", {})
    for k in ["clarity", "novelty", "technical_depth", "methodology", "readability"]:
        try: scores[k] = round(max(0.0, min(10.0, float(scores.get(k, 7.0)))), 1)
        except: scores[k] = 7.0
    
    res["scores"] = scores
    for k in ["contributions", "weaknesses", "suggestions"]:
        res[k] = [clean(str(x)) for x in res.get(k, [])[:5]]
    
    res["plagiarism"] = clean(res.get("plagiarism", "Analysis unavailable"))
    return res


def _format_similar_papers(similar_papers: list) -> str:
    if not similar_papers: return "None"
    lines = []
    for idx, p in enumerate(similar_papers[:3], 1):
        lines.append(f"{idx}. {p.get('title')} ({p.get('year')})")
    return "\n".join(lines)


def _fallback_result(no_ref: bool) -> dict:
    return {
        "scores": {"clarity": 7, "novelty": 7, "technical_depth": 7, "methodology": 7, "readability": 7},
        "contributions": ["Detected key contributions..."],
        "weaknesses": ["Further analysis needed..."],
        "suggestions": ["Consider expanding citations..."],
        "plagiarism": "No comparison data" if no_ref else "Analysis timed out",
        "status": "partial"
    }


def _ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for v in value:
            s = str(v).strip()
            if s:
                items.append(s)
        return items
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    s = str(value).strip()
    return [s] if s else []


def normalize_eval_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize evaluation output into a stable shape for Engine mode.

    Always ensures:
      - gaps: list[str]
      - suggestions: list[str]
      - improvements: list[str]
      - analysis: str
    """
    if result is None:
        result = {}

    try:
        logger.info("[EVAL RAW OUTPUT] %s", json.dumps(result, ensure_ascii=False)[:2000])
    except Exception:
        logger.info("[EVAL RAW OUTPUT] %s", str(result)[:2000])

    normalized: Dict[str, Any] = dict(result)

    # ── 1. Collect gaps from primary + alternate field names
    gaps: list[str] = []
    alt_gap_keys = [
        "gaps",
        "missing_points",
        "weaknesses",
        "deficiencies",
        "limitations",
        "missing_items",
    ]
    for key in alt_gap_keys:
        if key in result:
            gaps.extend(_ensure_list(result.get(key)))

    summary = result.get("summary") or {}
    if not gaps and isinstance(summary, dict):
        gaps.extend(_ensure_list(summary.get("key_weaknesses")))

    gaps = [g for g in gaps if g]

    # ── 2. Heuristic fallback from analysis text when gaps are empty
    if not gaps:
        analysis_sources: list[str] = []
        if isinstance(summary, dict):
            if isinstance(summary.get("analysis"), str):
                analysis_sources.append(summary.get("analysis"))  # type: ignore[arg-type]
            analysis_sources.extend(_ensure_list(summary.get("key_weaknesses")))

        evaluations = result.get("evaluations") or []
        if isinstance(evaluations, list):
            for ev in evaluations:
                if not isinstance(ev, dict):
                    continue
                details = ev.get("details") or {}
                if isinstance(details, dict):
                    if isinstance(details.get("analysis"), str):
                        analysis_sources.append(details.get("analysis"))  # type: ignore[arg-type]

        joined = ". ".join([s for s in analysis_sources if isinstance(s, str)])
        sentences = re.split(r"(?<=[.!?])\s+", joined)
        for sent in sentences:
            s = sent.strip()
            if not s:
                continue
            low = s.lower()
            if any(token in low for token in ["lack", "lacks", "missing", "does not", "no ", "insufficient", "unclear", "limited"]):
                cleaned = re.sub(r"^\b(the answer|the paper)\b\s+", "", s, flags=re.IGNORECASE)
                cleaned = cleaned.rstrip(".")
                if cleaned:
                    cleaned = cleaned[0].upper() + cleaned[1:]
                    gaps.append(cleaned)
            if len(gaps) >= 3:
                break

    # ── 3. Final fallback when no gaps could be extracted
    if not gaps:
        gaps = [
            "Limited methodological detail",
            "Insufficient supporting evidence",
            "Lack of quantitative evaluation",
        ]

    normalized["gaps"] = gaps
    print("Generated gaps:", gaps)

    # ── 4. Normalize suggestions
    suggestions = _ensure_list(result.get("suggestions"))
    if not suggestions and isinstance(summary, dict):
        suggestions = _ensure_list(summary.get("key_strengths"))
    normalized["suggestions"] = suggestions

    # ── 5. Normalize improvements
    improvements = _ensure_list(result.get("improvements"))
    if not improvements:
        eval_improvements: list[str] = []
        evaluations = result.get("evaluations") or []
        if isinstance(evaluations, list):
            for ev in evaluations:
                if not isinstance(ev, dict):
                    continue
                details = ev.get("details") or {}
                if isinstance(details, dict):
                    eval_improvements.extend(_ensure_list(details.get("improvements")))
        improvements = eval_improvements
    normalized["improvements"] = improvements

    # ── 6. Normalize analysis text
    analysis_text = result.get("analysis")
    if not isinstance(analysis_text, str) or not analysis_text.strip():
        parts: list[str] = []
        if isinstance(summary, dict):
            if "overall_score" in summary:
                try:
                    parts.append(f"Overall score: {summary['overall_score']}")
                except Exception:
                    pass
            key_weaks = _ensure_list(summary.get("key_weaknesses"))
            if key_weaks:
                parts.append("Key weaknesses: " + "; ".join(key_weaks[:3]))
            key_strengths = _ensure_list(summary.get("key_strengths"))
            if key_strengths:
                parts.append("Key strengths: " + "; ".join(key_strengths[:3]))
        analysis_text = " ".join(parts) if parts else "Evaluation completed."
    normalized["analysis"] = analysis_text

    # ── 7. Ensure each evaluation row has non-empty details.gaps and details.improvements
    evaluations = result.get("evaluations") or []
    normalized_evaluations: list[Dict[str, Any]] = []
    if isinstance(evaluations, list):
        for ev in evaluations:
            if not isinstance(ev, dict):
                continue
            ev_copy = dict(ev)
            details = ev_copy.get("details")
            if not isinstance(details, dict):
                details = {}
            details_copy: Dict[str, Any] = dict(details)

            details_gaps = _ensure_list(details_copy.get("gaps"))
            if not details_gaps:
                details_gaps = list(gaps[:3])
            details_copy["gaps"] = details_gaps

            details_improvements = _ensure_list(details_copy.get("improvements"))
            if not details_improvements:
                details_improvements = list(suggestions[:3]) if suggestions else [
                    "Clarify methodology and experimental setup.",
                    "Add stronger evidence with comparative evaluation.",
                ]
            details_copy["improvements"] = details_improvements

            if not isinstance(details_copy.get("analysis"), str) or not details_copy.get("analysis", "").strip():
                details_copy["analysis"] = analysis_text

            ev_copy["details"] = details_copy
            normalized_evaluations.append(ev_copy)
    normalized["evaluations"] = normalized_evaluations

    # ── 8. Stable top-level response mapping for API consumers
    normalized["result"] = {
        "gaps": list(normalized.get("gaps", [])),
        "suggestions": list(normalized.get("suggestions", [])),
        "analysis": normalized.get("analysis", ""),
    }

    # ── 9. Debug logging for gaps extraction
    try:
        logger.info(
            "[EVAL NORMALIZED] %s",
            json.dumps(
                {
                    "gaps_preview": normalized["gaps"][:3],
                    "suggestions_preview": normalized["suggestions"][:3],
                    "improvements_preview": normalized["improvements"][:3],
                    "analysis_preview": str(normalized["analysis"])[:200],
                },
                ensure_ascii=False,
            ),
        )
    except Exception:
        pass
    logger.info("[GAPS EXTRACTED] count=%d", len(normalized.get("gaps", [])))

    return normalized

def get_document_text(db, max_chars: int = 5000) -> str:
    try:
        collection = db._collection
        result     = collection.get(include=["documents"])
        docs       = result.get("documents", [])
        return "\n\n".join(docs)[:max_chars]
    except Exception:
        return ""
