import os
import json
import time
import numpy as np
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate

# Standard questions
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
    "Future Work", "Conclusion"
]

QUICK_EVAL_QUESTIONS = [EVAL_QUESTIONS[0], EVAL_QUESTIONS[7], EVAL_QUESTIONS[1]]
QUICK_EVAL_DOMAINS = [EVAL_DOMAINS[0], EVAL_DOMAINS[7], EVAL_DOMAINS[1]]

_emb_model = None

def get_embedding_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _emb_model

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
            "split_ratio": "70/30"
        }
    except Exception:
        return {"total_chunks": 0, "total_docs": len(uploaded_files), "split_ratio": "70/30"}

def compute_metrics(scores: list, threshold: float = 0.45) -> dict:
    if not scores:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    total = len(scores)
    # y_true is 1 for all because we expect the model to answer correctly
    y_true = [1] * total
    y_pred = [1 if s >= threshold else 0 for s in scores]
    
    correct = sum(y_pred)
    accuracy = correct / total
    
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "threshold": threshold,
        "correct": correct,
        "total": total
    }

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

async def run_full_evaluation(llm, rag_chain, db, uploaded_files: list, progress_cb=None, mode="quick", answer_provider=None) -> dict:
    start_time = time.time()
    questions = QUICK_EVAL_QUESTIONS if mode == "quick" else EVAL_QUESTIONS
    domains = QUICK_EVAL_DOMAINS if mode == "quick" else EVAL_DOMAINS
    
    if progress_cb: progress_cb("Initializing", 0, len(questions))
    dataset_info = get_dataset_info(db, uploaded_files)
    
    # Get context for reference answers
    try:
        collection = db._collection
        result = collection.get(include=["documents"])
        all_chunks = result.get("documents", [])
        full_context = "\n\n".join(all_chunks[:8]) # Limit to stay within context window
    except:
        full_context = "Context unavailable."

    # Batch reference generation
    if progress_cb: progress_cb("Generating Reference Answers", 0, len(questions))
    q_list_str = "\n".join([f"- {q}" for q in questions])
    batch_prompt = _BATCH_REF_PROMPT.format(context=full_context, questions=q_list_str)
    
    ref_answers = {}
    try:
        raw_batch = await llm.ainvoke(batch_prompt)
        match = re.search(r'\{.*\}', raw_batch, re.DOTALL)
        if match:
            ref_answers = json.loads(match.group())
    except Exception:
        # Fallback will be handled in the loop
        pass

    emb_model = get_embedding_model()

    async def evaluate_single(idx, q):
        q_start = time.time()
        ref_ans = ref_answers.get(q, "Information not found in document.")
        
        # Get Model Answer
        try:
            if answer_provider:
                mod_ans, cached, _ = await answer_provider(q)
            else:
                # Call async RAG chain
                res = await rag_chain(q, mode="researcher")
                mod_ans = res.get("answer", "No answer generated.")
                cached = res.get("cache_hit", False)
        except Exception as e:
            mod_ans = f"Error: {str(e)}"
            cached = False

        # Compute Similarity
        try:
            # Embedding calls are heavy, run in thread
            embs = await asyncio.to_thread(emb_model.embed_documents, [ref_ans, mod_ans])
            v_ref = np.array(embs[0]).reshape(1, -1)
            v_mod = np.array(embs[1]).reshape(1, -1)
            score = float(cosine_similarity(v_ref, v_mod)[0][0])
            score = round(score, 4)
        except:
            score = 0.0

        q_time = round(time.time() - q_start, 2)
        return {
            "id": idx + 1,
            "domain": domains[idx],
            "query": q,
            "reference_answer": ref_ans,
            "model_answer": mod_ans,
            "similarity": score,
            "correct": score >= 0.45,
            "cached": cached,
            "time_taken": q_time
        }

    # Parallelize evaluations
    tasks = [evaluate_single(i, q) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)
    
    scores = [r["similarity"] for r in results]
    metrics = compute_metrics(scores, threshold=0.45)
    
    total_time = round(time.time() - start_time, 2)
    metrics["total_time_s"] = total_time
    metrics["avg_similarity"] = round(float(np.mean(scores)), 4) if scores else 0.0

    final_result = {
        "dataset_info": dataset_info,
        "metrics": metrics,
        "results": results,
        "mode": mode,
        "status": "completed"
    }

    if progress_cb: progress_cb("Completed", len(questions), len(questions), result=final_result)
    return final_result

import re
