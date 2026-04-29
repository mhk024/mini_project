import os
import re
import asyncio
import logging
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from sentence_transformers import CrossEncoder

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
logger = logging.getLogger(__name__)

# =========================
# 🛠️ HELPERS
# =========================
def load_document(file_path: str):
    if file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    raise ValueError("Unsupported file type")

def create_or_load_db(file_path: str):
    """Synchronous core for DB creation (run in thread)."""
    documents = load_document(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    docs = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Batch processing is automatic in Chroma.from_documents
    db = Chroma.from_documents(docs, embeddings)
    return db

# =========================
# 🧠 ASYNC CORE
# =========================
_ENHANCE_PROMPT = PromptTemplate.from_template(
    "Optimize this research query for vector search. Return ONLY the optimized query.\nQuery: {question}"
)

async def enhance_query_async(llm, question: str) -> str:
    try:
        enhanced = await llm.ainvoke(_ENHANCE_PROMPT.format(question=question))
        return enhanced.strip()
    except Exception:
        return question

_cross_encoder = None
def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder

def rerank_documents(query: str, documents: List[Any]) -> List[Dict[str, Any]]:
    if not documents: return []
    encoder = get_cross_encoder()
    pairs = [(query, d.page_content) for d in documents]
    scores = encoder.predict(pairs)
    
    scored = []
    for doc, score in zip(documents, scores):
        scored.append({
            "content": doc.page_content,
            "score": float(score),
            "source": os.path.basename(doc.metadata.get("source", "unknown"))
        })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

# =========================
# 🎓 STUDENT MODE (ASYNC)
# =========================
async def generate_summary_async(llm, context: str) -> Dict[str, str]:
    prompt = f"Summarize this research context in two parts: SHORT (50 words) and DETAILED (300 words). Use 'DETAILED:' as separator.\nContext: {context[:4000]}"
    try:
        raw = await llm.ainvoke(prompt)
        if "DETAILED:" in raw:
            parts = raw.split("DETAILED:", 1)
            return {"short": parts[0].strip(), "detailed": parts[1].strip()}
        return {"short": raw[:200], "detailed": raw}
    except Exception:
        return {"short": "Error", "detailed": "Error"}

# =========================
# ⛓️ RAG PIPELINE
# =========================
_QA_PROMPT = PromptTemplate.from_template(
    """Use the following context to answer the research question.
Context: {context}
Question: {question}
Answer:"""
)

def build_rag_chain(db):
    llm = OllamaLLM(model="phi", num_predict=1024)
    
    async def rag_chain(question: str, mode: str = "student") -> Dict[str, Any]:
        # Step 1: Parallel retrieval and enhancement
        enhanced_task = asyncio.create_task(enhance_query_async(llm, question))
        
        # Retrieval (sync call in thread)
        raw_docs = await asyncio.to_thread(db.similarity_search, question, k=5)
        enhanced_query = await enhanced_task
        
        # Step 2: Re-ranking
        reranked = rerank_documents(enhanced_query, raw_docs)
        context = "\n\n".join([d["content"] for d in reranked[:3]])
        
        # Step 3: Generation
        answer = await llm.ainvoke(_QA_PROMPT.format(context=context, question=question))
        
        result = {
            "original_query": question,
            "enhanced_query": enhanced_query,
            "answer": answer.strip(),
            "retrieved_documents": reranked,
            "reranked_documents": reranked
        }
        
        if mode == "student":
            summary = await generate_summary_async(llm, context)
            result.update({"summary": summary, "key_points": [], "simplified_explanation": ""})
            
        return result

    return rag_chain
