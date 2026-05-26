"""
FastAPI backend — cold start stays light: no torch / embeddings / chroma at import time.
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dotenv import load_dotenv
load_dotenv()

# Directory for uploaded and processed files. Render's slug is read‑only, so use a writable path.
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "dataset"))
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.cache_manager import cache_manager
from routes.academic import router as academic_router
from services.resource_manager import get_llm
import state

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

_context_lock = asyncio.Lock()


def _evict_contexts_if_needed():
    while len(state.loaded_contexts) > state.MAX_LOADED_CONTEXTS:
        evicted_name, _ = state.loaded_contexts.popitem(last=False)
        logger.info("Evicted context for %s (max=%s)", evicted_name, state.MAX_LOADED_CONTEXTS)


async def _load_context_async(filename: str):
    """Load vector DB + RAG chain on demand (thread pool for CPU/IO heavy work)."""
    if filename in state.loaded_contexts:
        ctx = state.loaded_contexts[filename]
        state.loaded_contexts.move_to_end(filename)
        state.vector_db = ctx["db"]
        state.qa_chain = ctx["qa"]
        return

    file_path = os.path.join("dataset", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    async with _context_lock:
        if filename in state.loaded_contexts:
            ctx = state.loaded_contexts[filename]
            state.vector_db = ctx["db"]
            state.qa_chain = ctx["qa"]
            return

        from rag_pipeline import create_or_load_db, build_rag_chain

        db = await asyncio.to_thread(create_or_load_db, file_path)
        try:
            qa = await asyncio.to_thread(build_rag_chain, db)
        except Exception as e:
            logger.error("Failed to initialize LLM or RAG chain: %s", e)
            raise HTTPException(status_code=500, detail="LLM initialization error")
        
        state.loaded_contexts[filename] = {"db": db, "qa": qa}
        _evict_contexts_if_needed()
        state.vector_db = db
        state.qa_chain = qa


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Research Assistant API starting (lazy-load mode)")
    for route in app.routes:
        if hasattr(route, "methods"):
            logger.info("Route: %s methods: %s", route.path, route.methods)
    yield
    logger.info("Research Assistant API shutting down")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(academic_router, prefix="/academic")


class QuestionRequest(BaseModel):
    username: str
    question: str
    filename: str | None = None
    mode: str = "student"


class AnalyzePaperRequest(BaseModel):
    filename: str
    reference_text: str = ""


class EvaluateRequest(BaseModel):
    filename: str
    questions: list | None = None
    mode: str = "quick"


class SetFileRequest(BaseModel):
    filename: str


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "groq": os.getenv("GROQ_API_KEY", "not_set"),
            "rag": "ready" if state.qa_chain else "idle",
            "cache": "active",
            "loaded_contexts": len(state.loaded_contexts),
            "max_contexts": state.MAX_LOADED_CONTEXTS,
        },
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    logger.info("Entry /ask: %s", request)
    start_time = time.time()
    try:
        if request.filename:
            await _load_context_async(request.filename)
        if not state.qa_chain:
            raise HTTPException(status_code=503, detail="Context not loaded")
        ckey = f"ask:{request.filename}:{request.question}:{request.mode}"
        cached = await cache_manager.get(ckey, category="rag_chat")
        if cached:
            cached["cache_hit"] = True
            return cached
        result = await state.qa_chain(request.question, mode=request.mode)
        result["execution_time_ms"] = round((time.time() - start_time) * 1000, 2)
        result["cache_hit"] = False
        await cache_manager.set(ckey, result, category="rag_chat")
        logger.info("Result /ask: %s", result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in /ask: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    logger.info("Entry /evaluate: %s", request)
    try:
        await _load_context_async(request.filename)
        from evaluator import run_full_evaluation

        result = await run_full_evaluation(
            llm=get_llm(),
            rag_chain=state.qa_chain,
            db=state.vector_db,
            uploaded_files=[request.filename],
            mode=request.mode,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Evaluation error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/analyze/paper")
async def analyze_paper_endpoint(request: AnalyzePaperRequest):
    try:
        await _load_context_async(request.filename)
        from research_analyzer import run_full_pipeline, get_document_text

        text = await asyncio.to_thread(get_document_text, state.vector_db)
        if not text:
            return {"status": "error", "message": "Could not extract text"}

        result = await run_full_pipeline(
            get_llm(), text, reference_text=request.reference_text
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Analysis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/set_file")
async def set_file(request: SetFileRequest):
    logger.info("Loading file context: %s", request.filename)
    file_path = os.path.join(DATASET_DIR, request.filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")

    await _load_context_async(request.filename)
    return {"message": "Success", "filename": request.filename}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        file_path = os.path.join(DATASET_DIR, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        if os.getenv("PRELOAD_ON_UPLOAD", "false").lower() in ("1", "true", "yes"):
            await _load_context_async(file.filename)

        return {"message": "Success", "filename": file.filename}
    except Exception as e:
        logger.error("Upload error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/files")
async def list_files():
    dataset_path = DATASET_DIR
    if not os.path.isdir(dataset_path):
        return {"files": []}
    files = [
        f
        for f in os.listdir(dataset_path)
        if os.path.isfile(os.path.join(dataset_path, f))
    ]
    return {"files": files}


@app.get("/history/{username}")
async def get_history(username: str):
    return {"history": []}


@app.post("/login")
async def login(creds: dict):
    return {"status": "success", "username": creds.get("username")}


def __getattr__(name: str):
    """Backward compatibility for `import app` in route modules."""
    if name == "vector_db":
        return state.vector_db
    if name == "qa_chain":
        return state.qa_chain
    if name == "_get_llm":
        return get_llm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
