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
