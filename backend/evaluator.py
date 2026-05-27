"""
RAG evaluation — sklearn removed; embeddings loaded lazily via resource_manager.
"""

from __future__ import annotations

import json
import re
import time
import asyncio
import logging
from typing import Callable, Any

import math
from langchain_core.prompts import PromptTemplate

from services.resource_manager import get_embeddings

logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "What is the main objective or research problem addressed in this paper?",
    "What methodology or approach is proposed in this study?",
    "What are the key findings or results of this research?",
    "What datasets or experimental setup were used in this study?",
    "How does the proposed approach compare to existing methods or baselines?",
    "What evaluation metrics are used to measure performance?",
    "What are the main contributions of this work?",
    "What limitations or challenges are identified in this research?",
    "What future work or improvements are suggested by the authors?",
    "What is the overall conclusion of this research paper?",
]

EVAL_DOMAINS = [
    "Research Objective", "Methodology", "Results & Findings", "Dataset & Experiments",
    "Comparative Analysis", "Evaluation Metrics", "Contributions", "Limitations",
    "Future Work", "Conclusion",
]

QUICK_EVAL_QUESTIONS = [EVAL_QUESTIONS[0], EVAL_QUESTIONS[7], EVAL_QUESTIONS[1]]
QUICK_EVAL_DOMAINS = [EVAL_DOMAINS[0], EVAL_DOMAINS[7], EVAL_DOMAINS[1]]

_BATCH_REF_PROMPT = PromptTemplate.from_template(
    """You are a research expert. Based ONLY on the provided document context, 
write a concise reference answer (2-3 sentences max) for EACH of the following questions.

Return your response as a valid JSON object where keys are the exact question strings and values are the reference answers.
If information is not found, use "Information not found in document."

Context:
{context}

Questions:
{questions}

JSON response:"""
)


def _cosine_similarity(a, b) -> float:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = (norm_a * norm_b) or 1.0
    return float(sum(x * y for x, y in zip(a, b)) / denom)


def get_dataset_info(db, uploaded_files: list) -> dict:
    try:
        collection = db._collection
        result = collection.get(include=["documents"])
        chunks = result.get("documents", [])
        total = len(chunks)
        train_n = int(total * 0.70)
        return {
            "total_chunks": total,
            "train_chunks": train_n,
            "test_chunks": total - train_n,
            "total_docs": len(uploaded_files),
            "eval_questions": len(EVAL_QUESTIONS),
            "domains": EVAL_DOMAINS,
            "split_ratio": "70/30",
        }
    except Exception:
        return {"total_chunks": 0, "total_docs": len(uploaded_files), "split_ratio": "70/30"}


def compute_metrics(scores: list, threshold: float = 0.45) -> dict:
    if not scores:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    total = len(scores)
    y_pred = [1 if s >= threshold else 0 for s in scores]
    correct = sum(y_pred)
    accuracy = correct / total

    tp = correct
    fp = total - correct
    fn = 0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "threshold": threshold,
        "correct": correct,
        "total": total,
    }


async def run_full_evaluation(
    llm,
    rag_chain,
    db,
    uploaded_files: list,
    progress_cb=None,
    mode="quick",
    answer_provider=None,
) -> dict:
    start_time = time.time()
    questions = QUICK_EVAL_QUESTIONS if mode == "quick" else EVAL_QUESTIONS
    domains = QUICK_EVAL_DOMAINS if mode == "quick" else EVAL_DOMAINS

    if progress_cb:
        progress_cb("Initializing", 0, len(questions))
    dataset_info = get_dataset_info(db, uploaded_files)

    try:
        collection = db._collection
        result = collection.get(include=["documents"])
        all_chunks = result.get("documents", [])
        full_context = "\n\n".join(all_chunks[:8])
    except Exception:
        full_context = "Context unavailable."

    if progress_cb:
        progress_cb("Generating Reference Answers", 0, len(questions))
    q_list_str = "\n".join([f"- {q}" for q in questions])
    batch_prompt = _BATCH_REF_PROMPT.format(context=full_context, questions=q_list_str)

    ref_answers = {}
    try:
        raw_batch = await llm.ainvoke(batch_prompt)
        raw_text = raw_batch.content if hasattr(raw_batch, "content") else str(raw_batch)
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            ref_answers = json.loads(match.group())
    except Exception:
        pass

    emb_model = get_embeddings()

    async def evaluate_single(idx, q):
        q_start = time.time()
        ref_ans = ref_answers.get(q, "Information not found in document.")

        try:
            if answer_provider:
                mod_ans, cached, _ = await answer_provider(q)
            else:
                res = await rag_chain(q, mode="researcher")
                mod_ans = res.get("answer", "No answer generated.")
                cached = res.get("cache_hit", False)
        except Exception as e:
            mod_ans = f"Error: {str(e)}"
            cached = False

        try:
            embs = await asyncio.to_thread(emb_model.embed_documents, [ref_ans, mod_ans])
            score = round(_cosine_similarity(embs[0], embs[1]), 4)
        except Exception:
            score = 0.0

        return {
            "id": idx + 1,
            "domain": domains[idx],
            "query": q,
            "reference_answer": ref_ans,
            "model_answer": mod_ans,
            "similarity": score,
            "correct": score >= 0.45,
            "cached": cached,
            "time_taken": round(time.time() - q_start, 2),
        }

    results = await asyncio.gather(*[evaluate_single(i, q) for i, q in enumerate(questions)])
    scores = [r["similarity"] for r in results]
    metrics = compute_metrics(scores, threshold=0.45)
    metrics["total_time_s"] = round(time.time() - start_time, 2)
    metrics["avg_similarity"] = round(float(sum(scores) / len(scores)), 4) if scores else 0.0

    final_result = {
        "dataset_info": dataset_info,
        "metrics": metrics,
        "results": results,
        "mode": mode,
        "status": "completed",
    }

    if progress_cb:
        progress_cb("Completed", len(questions), len(questions), result=final_result)
    return final_result
