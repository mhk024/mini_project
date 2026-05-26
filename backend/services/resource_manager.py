"""
Centralized lazy loaders for ML / vector resources.

Nothing in this module imports torch, sentence-transformers, or chromadb at import time.
Models load on first use inside get_*() functions, guarded by locks for thread safety.
"""

from __future__ import annotations

import os
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Prefer ONNX-based embeddings on memory-constrained hosts (e.g. Render 512MB).
USE_FASTEMBED = os.getenv("USE_FASTEMBED", "true").lower() in ("1", "true", "yes")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-small-en-v1.5" if USE_FASTEMBED else "sentence-transformers/all-MiniLM-L6-v2",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "false").lower() in ("1", "true", "yes")

_lock = threading.Lock()
_embeddings: Optional[Any] = None
_llm: Optional[Any] = None
_cross_encoder: Optional[Any] = None


def get_embeddings():
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


def get_llm():
    """Singleton Ollama LLM client (created on first RAG / analysis request)."""
    global _llm
    if _llm is not None:
        return _llm

    with _lock:
        if _llm is not None:
            return _llm

        from langchain_ollama import OllamaLLM

        logger.info("Initializing Ollama LLM model=%s base_url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
        _llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "1024")),
        )
        return _llm


def get_cross_encoder():
    """
    Optional cross-encoder reranker (~80–120MB). Disabled by default on Render.
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

        logger.info("Loading cross-encoder (high memory)")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        return _cross_encoder


def release_heavy_models(keep_embeddings: bool = True):
    """Drop cached models to free RAM (e.g. after idle period)."""
    global _llm, _cross_encoder, _embeddings
    with _lock:
        _llm = None
        _cross_encoder = None
        if not keep_embeddings:
            _embeddings = None
    logger.info("Released heavy models (keep_embeddings=%s)", keep_embeddings)
