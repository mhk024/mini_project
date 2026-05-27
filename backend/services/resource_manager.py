import os
import logging
import threading
from typing import Any, Optional
import asyncio
import httpx

logger = logging.getLogger(__name__)

# ---------- Configuration ----------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set. LLM will not be available.")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "1024"))

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

_lock = threading.Lock()
_embeddings: Optional[Any] = None
_llm: Optional[Any] = None

# ---------- Embeddings ----------
class LocalSentenceTransformerEmbeddings:
    """Wrapper around local SentenceTransformer for LangChain compatibility."""
    def __init__(self, model: Any):
        self.model = model

    def embed_query(self, text: str) -> list[float]:
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error("Failed to embed query: %s", e)
            return [0.0] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self.model.encode(texts).tolist()
        except Exception as e:
            logger.error("Failed to embed documents: %s", e)
            return [[0.0] * 384 for _ in texts]

class FallbackEmbeddings:
    """Fallback embedding generator that returns zero-filled vectors if the main model fails."""
    def __init__(self, error: Exception):
        self.error = error
        logger.warning("Using FallbackEmbeddings due to model load failure: %s", error)

    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 384

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in texts]

def get_embeddings() -> Any:
    """Singleton for local SentenceTransformer embeddings."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    with _lock:
        if _embeddings is not None:
            return _embeddings
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading local SentenceTransformer model 'sentence-transformers/all-MiniLM-L6-v2'...")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            _embeddings = LocalSentenceTransformerEmbeddings(model)
            logger.info("Initialized local SentenceTransformer embeddings successfully.")
        except Exception as e:
            logger.error("Failed to load local SentenceTransformer: %s", e, exc_info=True)
            _embeddings = FallbackEmbeddings(e)
        return _embeddings

# ---------- LLM (Groq) ----------
from langchain_groq import ChatGroq

def get_llm() -> Any:
    """Singleton Groq LLM client (created on first request)."""
    global _llm
    if _llm is not None:
        return _llm
    with _lock:
        if _llm is not None:
            return _llm
        if not GROQ_API_KEY:
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

# ---------- Cleanup ----------
def release_heavy_models(keep_embeddings: bool = True) -> None:
    """Drop cached models to free RAM (e.g., after idle period)."""
    global _llm, _embeddings
    with _lock:
        _llm = None
        if not keep_embeddings:
            _embeddings = None
    logger.info("Released heavy models (keep_embeddings=%s)", keep_embeddings)
