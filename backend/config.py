#config.py
"""Configuration constants for RAG Research Assistant"""

# Jaccard similarity threshold for chunk deduplication (0.0-1.0)
DUPLICATE_JACCARD_THRESHOLD = 0.8

# Number of raw documents to retrieve before deduplication
RETRIEVAL_RAW_K = 20

# Number of documents to keep after deduplication
RETRIEVAL_KEEP_K = 5

# Similarity threshold used by evaluator/plagiarism detection
SIMILARITY_THRESHOLD = 0.45

# ---------- New defaults for upload & chunking ----------
import os

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", 20))
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
# Ensure the uploads directory exists at import time (lightweight)
os.makedirs(UPLOAD_DIR, exist_ok=True)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

