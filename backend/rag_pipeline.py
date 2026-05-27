"""
RAG pipeline — all heavy dependencies are lazy-loaded via services.resource_manager.
"""

from __future__ import annotations

import os
import hashlib
import asyncio
import logging
# import numpy as np  # Removed for memory optimization
import math
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from backend.services.resource_manager import (
    get_embeddings,
    get_llm,
    CHROMA_PERSIST_DIR,
    GROQ_MODEL,
    GROQ_API_KEY,
)
from backend.services.cache_manager import cache_manager
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    docs = splitter.split_documents(documents)
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
        duplicate = False
        for kept in deduped:
            if _jaccard_similarity(chunk.page_content, kept.page_content) >= DUPLICATE_JACCARD_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            deduped.append(chunk)
    return deduped


def _rerank_with_embeddings(query: str, documents: List[Any]) -> List[Dict[str, Any]]:
    embeddings = get_embeddings()
    scored = []
    for doc in documents:        # Compute similarity using pure python
        a = embeddings.embed_query(query)
        b = embeddings.embed_query(doc.page_content)
        dot = sum(x*y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x*x for x in a))
        norm_b = math.sqrt(sum(y*y for y in b))
        denom = (norm_a * norm_b) or 1.0
        score = float(dot / denom)
        scored.append({
            "content": doc.page_content,
            "score": score,
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


def rerank_documents(query: str, documents: List[Any]) -> List[Dict[str, Any]]:
    if not documents:
        return []
    return _rerank_with_embeddings(query, documents)


async def enhance_query_async(llm, question: str) -> str:
    try:
        enhanced = await llm.ainvoke(_ENHANCE_PROMPT.format(question=question))
        return enhanced.strip()
    except Exception:
        return question


async def generate_document_summaries(file_path: str) -> Dict[str, str]:
    from langchain_groq import ChatGroq
    import re
    import json

    logger.info("Initializing Groq client for document summarization (max_tokens=4096)...")
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEY is not set.")
        return {
            "short": "Error: GROQ_API_KEY not configured.",
            "simplified": "Error: GROQ_API_KEY not configured.",
            "detailed": "Error: GROQ_API_KEY not configured."
        }

    llm = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=4096,
    )

    # Load document
    docs = await asyncio.to_thread(load_document, file_path)
    if not docs:
        return {
            "short": "Empty document.",
            "simplified": "Empty document.",
            "detailed": "Empty document."
        }

    # Check page count
    is_pdf = file_path.endswith(".pdf")
    page_count = len(docs) if is_pdf else 1

    if is_pdf and page_count > 8:
        # Long PDF: use map-reduce/chunk aggregation
        logger.info("PDF has %d pages (>8). Running map-reduce summarization.", page_count)
        
        # 1. Map phase: group pages into chunks of 3 pages to capture enough detail
        group_size = 3
        chunks = []
        for i in range(0, page_count, group_size):
            group_docs = docs[i : i + group_size]
            group_text = "\n\n".join([doc.page_content for doc in group_docs])
            chunks.append((i + 1, min(i + group_size, page_count), group_text))

        # Summarize sections
        map_summaries = []
        for start_page, end_page, text in chunks:
            prompt = (
                f"You are an academic assistant. Analyze pages {start_page} to {end_page} of a research paper.\n"
                f"Generate a detailed, technical section summary of the main points, methodologies, and results "
                f"presented in this section. Do NOT write a brief overview; write a comprehensive summary of about 350-400 words.\n\n"
                f"Section Text:\n{text[:12000]}"
            )
            try:
                res = await llm.ainvoke(prompt)
                summary_text = res.content if hasattr(res, "content") else str(res)
                map_summaries.append(f"### Section: Pages {start_page}-{end_page}\n\n{summary_text.strip()}")
            except Exception as e:
                logger.error("Error summarizing section pages %d-%d: %s", start_page, end_page, e)
                map_summaries.append(f"### Section: Pages {start_page}-{end_page}\n\n(Summary unavailable for this section.)")

        concat_map_summaries = "\n\n".join(map_summaries)

        # 2. Reduce phase:
        # Detailed Summary is the synthesis overview + map summaries, guaranteeing >1500 words
        overview_prompt = (
            f"You are an academic researcher. Synthesize a comprehensive, high-level academic overview "
            f"(about 500-600 words) of the research paper based on the following section summaries. "
            f"Focus on the main objectives, methodology, key findings, and contributions.\n\n"
            f"Section Summaries:\n{concat_map_summaries}"
        )
        try:
            res = await llm.ainvoke(overview_prompt)
            overview_text = res.content if hasattr(res, "content") else str(res)
        except Exception as e:
            logger.error("Error generating overview: %s", e)
            overview_text = "Detailed overview of the paper."

        detailed_summary = (
            f"# Comprehensive Academic Summary\n\n"
            f"{overview_text.strip()}\n\n"
            f"## Detailed Section-by-Section Analysis\n\n"
            f"{concat_map_summaries}"
        )

        word_count = len(detailed_summary.split())
        logger.info("Generated detailed summary has %d words.", word_count)

        # Generate short summary (~150 words)
        short_prompt = (
            f"You are an academic assistant. Generate a concise, high-quality short summary (approximately 150 words) "
            f"of the research paper based on the following section summaries. Be precise and objective.\n\n"
            f"Section Summaries:\n{concat_map_summaries}"
        )
        try:
            res = await llm.ainvoke(short_prompt)
            short_summary = (res.content if hasattr(res, "content") else str(res)).strip()
        except Exception as e:
            logger.error("Error generating short summary: %s", e)
            short_summary = "Short summary unavailable."

        # Generate simplified explanation (~300-500 words)
        simplified_prompt = (
            f"You are a science communicator. Explain the research paper in simple, layman's terms "
            f"(approximately 300 to 500 words) for a general science reader or student. Avoid heavy jargon, "
            f"explain the 'why' and 'how' simply, and focus on the practical implications.\n\n"
            f"Section Summaries:\n{concat_map_summaries}"
        )
        try:
            res = await llm.ainvoke(simplified_prompt)
            simplified_summary = (res.content if hasattr(res, "content") else str(res)).strip()
        except Exception as e:
            logger.error("Error generating simplified summary: %s", e)
            simplified_summary = "Simplified explanation unavailable."

        return {
            "short": short_summary,
            "simplified": simplified_summary,
            "detailed": detailed_summary
        }
    else:
        # Short PDF (<= 8 pages) or TXT file: generate all three summaries in a single LLM call with JSON mode
        logger.info("Document has page count %d (<= 8) or is TXT. Running single-pass summarization.", page_count)
        all_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            f"You are an academic assistant. Analyze the provided research paper text and generate three distinct summary modes:\n\n"
            f"1. SHORT SUMMARY: A concise summary of about 150 words.\n"
            f"2. SIMPLIFIED EXPLANATION: A layman's terms explanation of about 300-500 words for students/general audience.\n"
            f"3. DETAILED SUMMARY: A detailed academic summary covering methodology, results, and contributions (about 600-800 words).\n\n"
            f"Return ONLY a JSON object with the keys 'short', 'simplified', and 'detailed'. Use this JSON format:\n"
            f"{{\n"
            f"  \"short\": \"...\",\n"
            f"  \"simplified\": \"...\",\n"
            f"  \"detailed\": \"...\"\n"
            f"}}\n\n"
            f"Paper Text:\n{all_text[:16000]}"
        )
        try:
            res = await llm.ainvoke(prompt)
            raw = res.content if hasattr(res, "content") else str(res)
            match = re.search(r'\{.*\}', raw.strip(), re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                return {
                    "short": parsed.get("short", "").strip(),
                    "simplified": parsed.get("simplified", "").strip(),
                    "detailed": parsed.get("detailed", "").strip()
                }
            else:
                raise ValueError("JSON not found in response")
        except Exception as e:
            logger.error("Error in single-pass summarization: %s", e)
            sentences = [s.strip() for s in all_text.replace("\n", " ").split('. ') if s]
            return {
                "short": (sentences[0] + ".") if sentences else "Short summary unavailable.",
                "simplified": "Simplified summary unavailable.",
                "detailed": all_text[:2000]
            }


async def get_document_summary(file_path: str) -> Dict[str, str]:
    if not file_path or not os.path.exists(file_path):
        return {
            "short": "No active document loaded.",
            "simplified": "No active document loaded.",
            "detailed": "No active document loaded."
        }

    cache_key = _collection_name(file_path)
    ckey = f"doc_summary:{cache_key}"

    # Check cache
    cached = await cache_manager.get(ckey, category="doc_summaries")
    if cached:
        logger.info("Loaded summaries for %s from cache.", file_path)
        return cached

    # Not cached, generate it!
    logger.info("Generating summaries for %s...", file_path)
    summaries = await generate_document_summaries(file_path)

    # Save to cache
    await cache_manager.set(ckey, summaries, category="doc_summaries")
    return summaries


def build_rag_chain(db, file_path: str = None):
    # Initialize LLM with safe fallback
    try:
        llm = get_llm()
    except Exception as e:
        logger.error("Failed to initialize LLM: %s", e)
        llm = None

    # If no DB is provided, return a stub chain that reports missing index
    if db is None:
        async def rag_chain(question: str, mode: str = "student") -> Dict[str, Any]:
            summary = {
                "short": "No DB loaded.",
                "simplified": "No DB loaded.",
                "detailed": "The backend does not have a vector store configured."
            }
            if file_path:
                try:
                    summary = await get_document_summary(file_path)
                except Exception:
                    pass
            return {
                "answer": "Document database not configured.",
                "summary": summary,
                "sources": [],
                "pipeline": {
                    "original_query": question,
                    "enhanced_query": question,
                    "retrieved_docs": [],
                    "reranked_docs": []
                }
            }
        return rag_chain

    async def rag_chain(question: str, mode: str = "student") -> Dict[str, Any]:
        try:
            # Ensure LLM is available
            if llm is None:
                raise RuntimeError("LLM not initialized")
            enhanced_q = await enhance_query_async(llm, question)
            raw_docs = await asyncio.to_thread(db.similarity_search, enhanced_q, k=RETRIEVAL_RAW_K)
            deduped = _deduplicate_chunks(raw_docs)
            reranked = rerank_documents(enhanced_q, deduped[:RETRIEVAL_KEEP_K])
            context = "\n\n".join([r["content"] for r in reranked])
            answer = await llm.ainvoke(_QA_PROMPT.format(context=context, question=question))
            answer_content = answer.content if hasattr(answer, "content") else str(answer)

            # Format retrieved docs for pipeline
            retrieved_docs_formatted = []
            for doc in raw_docs:
                retrieved_docs_formatted.append({
                    "content": doc.page_content,
                    "source": os.path.basename(doc.metadata.get("source", "unknown")),
                    "score": 0.0
                })

            summary = await get_document_summary(file_path)

            return {
                "answer": answer_content,
                "summary": summary,
                "sources": [r["source"] for r in reranked],
                "pipeline": {
                    "original_query": question,
                    "enhanced_query": enhanced_q,
                    "retrieved_docs": retrieved_docs_formatted,
                    "reranked_docs": reranked
                }
            }
        except Exception as e:
            logger.exception("Error in rag_chain for question %s", question)
            summary = {
                "short": "Error occurred.",
                "simplified": "Error occurred.",
                "detailed": str(e)
            }
            if file_path:
                try:
                    summary = await get_document_summary(file_path)
                except Exception:
                    pass
            return {
                "answer": "An error occurred while processing your request.",
                "summary": summary,
                "sources": [],
                "pipeline": {
                    "original_query": question,
                    "enhanced_query": question,
                    "retrieved_docs": [],
                    "reranked_docs": []
                }
            }

    return rag_chain
