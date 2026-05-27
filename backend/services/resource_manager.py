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

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    logger.error("TOGETHER_API_KEY environment variable not set. Embeddings will not be available.")
TOGETHER_EMBEDDING_MODEL = os.getenv("TOGETHER_EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-8k")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")

_lock = threading.Lock()
_embeddings: Optional[Any] = None
_llm: Optional[Any] = None

# ---------- Embeddings ----------
class TogetherEmbedding:
    """Simple wrapper around Together AI embeddings API."""
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(timeout=30)

    async def embed_query(self, text: str):
        payload = {"model": self.model, "input": [text]}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = await self.client.post("https://api.together.xyz/v1/embeddings", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Expected format: {"data": [{"embedding": [...]}]}
        return data["data"][0]["embedding"]

    async def embed_documents(self, texts: list[str]):
        # Batch request
        payload = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = await self.client.post("https://api.together.xyz/v1/embeddings", json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return [item["embedding"] for item in data["data"]]

def get_embeddings() -> Any:
    """Singleton for Together AI embeddings client."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings
    with _lock:
        if _embeddings is not None:
            return _embeddings
        if not TOGETHER_API_KEY:
            raise RuntimeError("TOGETHER_API_KEY environment variable not set")
        # Use a lightweight wrapper; for compatibility with LangChain we provide embed_query method.
        class SimpleEmbedding:
            def __init__(self, client: TogetherEmbedding):
                self.client = client
            def embed_query(self, text: str):
                # Synchronous wrapper around async call
                return asyncio.run(self.client.embed_query(text))
            def embed_documents(self, texts: list[str]):
                return asyncio.run(self.client.embed_documents(texts))
        _embeddings = SimpleEmbedding(TogetherEmbedding(TOGETHER_API_KEY, TOGETHER_EMBEDDING_MODEL))
        logger.info("Initialized Together AI embeddings with model %s", TOGETHER_EMBEDDING_MODEL)
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
