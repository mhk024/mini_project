"""
RAG pipeline — all heavy dependencies are lazy-loaded via services.resource_manager.
"""

from __future__ import annotations

import os
import re
import asyncio
import logging
import hashlib
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from services.resource_manager import (
    get_embeddings,
    get_llm,
    get_cross_encoder,
    CHROMA_PERSIST_DIR,
)
from config import DUPLICATE_JACCARD_THRESHOLD, RETRIEVAL_RAW_K, RETRIEVAL_KEEP_K

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
logger = logging.getLogger(__name__)

_ENHANCE_PROMPT = PromptTemplate.from_template(
    "Optimize this research query for vector search. Return ONLY the optimized query.\nQuery: {question}"
)

_QA_PROMPT = PromptTemplate.from_template(
    """Use the following context to answer the research question.
Context: {context}
Question: {question}
Answer:"""
)


def _collection_name(file_path: str) -> str:
    digest = hashlib.md5(os.path.abspath(file_path).encode()).hexdigest()[:16]
    return f"doc_{digest}"


def load_document(file_path: str):
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    if file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    raise ValueError("Unsupported file type")


def create_or_load_db(file_path: str):
    """Create or load a persisted Chroma collection for a document (sync — run in thread)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings()
    collection = _collection_name(file_path)
    persist_dir = os.path.join(CHROMA_PERSIST_DIR, collection)

    if os.path.isdir(persist_dir):
        logger.info("Loading existing Chroma collection at %s", persist_dir)
        return Chroma(
            collection_name=collection,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )

    logger.info("Building new Chroma index for %s", file_path)
    documents = load_document(file_path)
    documents = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    docs = splitter.split_documents(documents)
    # Deduplicate chunks before persisting
    docs = _deduplicate_chunks(docs)

    os.makedirs(persist_dir, exist_ok=True)
    db = Chroma.from_documents(
        docs,
        embeddings,
        collection_name=collection,
        persist_directory=persist_dir,
    )
    return db


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts based on word sets."""
    set_a = set(text_a.lower().split())
    set_b = set(text_b.lower().split())
    if not set_a and not set_b:
        return 1.0
    intersection = set_a.intersection(set_b)
    union = set_a.union(set_b)
    return float(len(intersection) / len(union))


def _deduplicate_chunks(chunks: List[Any]) -> List[Any]:
    """Remove near‑duplicate chunks based on Jaccard similarity.

    Chunks are kept in order; a chunk is discarded if its content is
    similar (≥ threshold) to any previously kept chunk.
    """
    deduped: List[Any] = []
    for chunk in chunks:
        content = getattr(chunk, "page_content", "")
        if any(_jaccard_similarity(content, getattr(c, "page_content", "")) >= DUPLICATE_JACCARD_THRESHOLD for c in deduped):
            continue
        deduped.append(chunk)
    return deduped
    encoder = get_cross_encoder()
    pairs = [(query, d.page_content) for d in documents]
    scores = encoder.predict(pairs)
    scored = []
    for doc, score in zip(documents, scores):
        scored.append({
            "content": doc.page_content,
            "score": float(score),
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def _rerank_with_embeddings(query: str, documents: List[Any]) -> List[Dict[str, Any]]:
    """Lightweight rerank using the same embedding model (no extra torch model)."""
    import numpy as np

    embeddings = get_embeddings()
    texts = [d.page_content for d in documents]
    vectors = embeddings.embed_documents([query] + texts)
    q_vec = np.array(vectors[0])
    scored = []
    for doc, doc_vec, text in zip(documents, vectors[1:], texts):
        d_vec = np.array(doc_vec)
        denom = (np.linalg.norm(q_vec) * np.linalg.norm(d_vec)) or 1.0
        score = float(np.dot(q_vec, d_vec) / denom)
        scored.append({
            "content": text,
            "score": score,
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def rerank_documents(query: str, documents: List[Any]) -> List[Dict[str, Any]]:
    if not documents:
        return []
    if get_cross_encoder() is not None:
        return _rerank_with_cross_encoder(query, documents)
    return _rerank_with_embeddings(query, documents)


async def enhance_query_async(llm, question: str) -> str:
    try:
        enhanced = await llm.ainvoke(_ENHANCE_PROMPT.format(question=question))
        return enhanced.strip()
    except Exception:
        return question


async def generate_summary_async(llm, context: str) -> Dict[str, str]:
    prompt = (
        f"Summarize this research context in two parts: SHORT (50 words) and DETAILED (300 words). "
        f"Use 'DETAILED:' as separator.\nContext: {context[:4000]}"
    )
    try:
        raw = await llm.ainvoke(prompt)
        if "DETAILED:" in raw:
            parts = raw.split("DETAILED:", 1)
            return {"short": parts[0].strip(), "detailed": parts[1].strip()}
        return {"short": raw[:200], "detailed": raw}
    except Exception:
        return {"short": "Error", "detailed": "Error"}


def build_rag_chain(db):
    llm = get_llm()

    async def rag_chain(question: str, mode: str = "student") -> Dict[str, Any]:
        enhanced_task = asyncio.create_task(enhance_query_async(llm, question))
        raw_docs = await asyncio.to_thread(db.similarity_search, question, k=RETRIEVAL_RAW_K)
        # Apply deduplication to retrieved chunks
        deduped = _deduplicate_chunks(raw_docs)
        # Keep top N after deduplication
        raw_docs = deduped[:RETREIVAL_KEEP_K] if len(deduped) > RETRIEVAL_KEEP_K else deduped
        enhanced_query = await enhanced_task

        reranked = await asyncio.to_thread(rerank_documents, enhanced_query, raw_docs)
        context = "\n\n".join([d["content"] for d in reranked[:3]])

        answer = await llm.ainvoke(_QA_PROMPT.format(context=context, question=question))

        result = {
            "original_query": question,
            "enhanced_query": enhanced_query,
            "answer": answer.strip(),
            "retrieved_documents": reranked,
            "reranked_documents": reranked,
        }

        if mode == "student":
            summary = await generate_summary_async(llm, context)
            result.update({"summary": summary, "key_points": [], "simplified_explanation": ""})

        return result

    return rag_chain
