"""Shared runtime state (no heavy imports)."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, Callable, Optional

# Max in-memory document contexts (1 on Render free tier).
MAX_LOADED_CONTEXTS = int(os.getenv("RAG_MAX_CONTEXTS", "1"))

qa_chain: Optional[Callable] = None
vector_db: Any = None
loaded_contexts: "OrderedDict[str, dict]" = OrderedDict()
