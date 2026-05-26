# Centralized lazy loaders for ML / vector resources.
"""
Centralized lazy loaders for ML / vector resources.
No heavy imports at module import time. Models load lazily inside get_*() functions, protected by a lock for thread safety.
"""

from __future__ import annotations

import os
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------- Configuration ----------
USE_FASTEMBED = os.getenv("USE_FASTEMBED", "true").lower() in ("1", "true", "yes")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5" if USE_FASTEMBED else "sentence-transformers/all-MiniLM-L6-v2",
)
# Groq configuration – pulled from env vars
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set. LLM will not be available.")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "false").lower() in ("1", "true", "yes")

_lock = threading.Lock()
_embeddings: Optional[Any] = None
_llm: Optional[Any] = None
_cross_encoder: Optional[Any] = None

# ---------- Embeddings ----------
def get_embeddings() -> Any:
    """Singleton embedding model (FastEmbed by default, HuggingFace optional)."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    with _lock:
        if _embeddings is not None:
            return _embeddings
        if USE_FASTEMBED:
            from langchain_community.embeddings import FastEmbedEmbeddings
            logger.info("Loading FastEmbed embeddings: %s", EMBEDDING_MODEL)
            _embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.info("Loading HuggingFace embeddings: %s", EMBEDDING_MODEL)
            _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return _embeddings

# ---------- LLM (Groq) ----------
def get_llm() -> Any:
    """Singleton Groq LLM client (created on first request)."""
    global _llm
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is not None:
            return _llm
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is missing; cannot initialize LLM")
            raise RuntimeError("GROQ_API_KEY environment variable not set")
        logger.info(
            "Initializing Groq LLM model=%s temperature=%.2f max_tokens=%d",
            GROQ_MODEL,
            GROQ_TEMPERATURE,
            GROQ_MAX_TOKENS,
        )
        _llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )
        return _llm

# ---------- Cross‑encoder (optional) ----------
def get_cross_encoder() -> Optional[Any]:
    """Optional cross‑encoder reranker (~80–120 MB). Disabled by default on Render.
    Returns None when USE_CROSS_ENCODER is false.
    """
    global _cross_encoder
    if not USE_CROSS_ENCODER:
        return None
    if _cross_encoder is not None:
        return _cross_encoder
    with _lock:
        if _cross_encoder is not None:
            return _cross_encoder
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross‑encoder (high memory)")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _cross_encoder

# ---------- Cleanup ----------
def release_heavy_models(keep_embeddings: bool = True) -> None:
    """Drop cached models to free RAM (e.g., after idle period)."""
    global _llm, _cross_encoder, _embeddings
    with _lock:
        _llm = None
        _cross_encoder = None
        if not keep_embeddings:
            _embeddings = None
    logger.info("Released heavy models (keep_embeddings=%s)", keep_embeddings)
