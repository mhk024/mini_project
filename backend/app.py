"""
FastAPI backend — cold start stays light: no torch / embeddings / chroma at import time.
"""

from __future__ import annotations

import os
import sys
import time
import gc
import logging
import asyncio
from threading import Lock
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()

# Directory for uploaded and processed files. Render's slug is read‑only, so use a writable path.
DATASET_DIR = os.getenv("DATASET_DIR", os.path.join(os.getenv("TMPDIR", "/tmp"), "dataset"))

from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.cache_manager import cache_manager
from services.resource_manager import get_llm, release_heavy_models
from routes.academic import router as academic_router
from rag_pipeline import create_or_load_db
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


async def _load_context_async(filename: str, file_path: str | None = None):
    """Load vector DB + RAG chain on demand (thread pool for CPU/IO heavy work)."""
    load_start = time.time()
    if filename in state.loaded_contexts:
        ctx = state.loaded_contexts[filename]
        state.loaded_contexts.move_to_end(filename)
        state.vector_db = ctx["db"]
        state.qa_chain = ctx["qa"]
        logger.info(f"Context {filename} loaded from cache in {(time.time() - load_start)*1000:.2f} ms")
        return

    if not file_path:
        file_path = os.path.join(DATASET_DIR, filename)
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
            qa = await asyncio.to_thread(build_rag_chain, db, file_path)
        except Exception as e:
            logger.error("Failed to initialize LLM or RAG chain: %s", e)
            raise HTTPException(status_code=500, detail="LLM initialization error")
        
        state.loaded_contexts[filename] = {"db": db, "qa": qa}
        _evict_contexts_if_needed()
        state.vector_db = db
        state.qa_chain = qa
        # Log loading time and perform garbage collection
        load_elapsed = (time.time() - load_start) * 1000
        logger.info(f"Loaded context {filename} in {load_elapsed:.2f} ms")
        gc.collect()
        # Optional memory usage logging on Linux
        try:
            import resource
            mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            logger.info(f"Memory usage after loading: {mem_kb/1024:.2f} MB")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Research Assistant API starting (lazy-load mode)")
    for route in app.routes:
        if hasattr(route, "methods"):
            logger.info("Route: %s methods: %s", route.path, route.methods)
    yield
    logger.info("Research Assistant API shutting down")
        # Release heavy models to free memory on shutdown
    release_heavy_models()


app = FastAPI(lifespan=lifespan)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
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
    file_id: str | None = None
    mode: str = "student"
    session_id: str | None = None


class AnalyzePaperRequest(BaseModel):
    filename: str | None = None
    reference_text: str = ""


class EvaluateRequest(BaseModel):
    filename: str
    questions: list | None = None
    mode: str = "quick"


class SetFileRequest(BaseModel):
    filename: str


import json
import uuid
from utils import status as file_status


USERS_FILE = "users.json"
HISTORY_FILE = "history.json"
APP_STATE_FILE = "app_state.json"
REGISTRY_FILE_PATH = os.path.join(os.path.dirname(__file__), "uploads", "file_registry.json")
registry_lock = Lock()

def load_file_registry() -> dict:
    """Load the file registry from the JSON file. Return empty dict if missing/malformed."""
    logger.info("Loading file registry from %s", REGISTRY_FILE_PATH)
    if not os.path.exists(REGISTRY_FILE_PATH):
        logger.info("File registry does not exist at %s, returning empty dict", REGISTRY_FILE_PATH)
        return {}
    try:
        with registry_lock:
            with open(REGISTRY_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    logger.warning("File registry JSON is not a dictionary, returning empty dict")
                    return {}
                logger.info("Successfully loaded %d entries from file registry", len(data))
                return data
    except Exception as e:
        logger.error("Failed to load file registry from %s: %s. Returning fallback empty dict", REGISTRY_FILE_PATH, e)
        return {}

def save_file_registry(registry: dict) -> None:
    """Save the file registry to the JSON file in a thread-safe and robust manner."""
    logger.info("Saving file registry to %s with %d entries", REGISTRY_FILE_PATH, len(registry))
    try:
        os.makedirs(os.path.dirname(REGISTRY_FILE_PATH), exist_ok=True)
        tmp_path = REGISTRY_FILE_PATH + ".tmp"
        with registry_lock:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(registry, f, indent=2)
            os.replace(tmp_path, REGISTRY_FILE_PATH)
        logger.info("Successfully saved file registry to %s", REGISTRY_FILE_PATH)
    except Exception as e:
        logger.error("Failed to save file registry: %s", e)

def register_uploaded_file(file_id: str, filename: str, file_path: str) -> None:
    """Register an uploaded file in the global FILE_REGISTRY and persist immediately."""
    logger.info("Registering uploaded file: file_id=%s, filename=%s, path=%s", file_id, filename, file_path)
    FILE_REGISTRY[file_id] = {
        "filename": filename,
        "path": file_path,
        "uploaded": True,
        "indexed": False,
        "created_at": datetime.utcnow().isoformat()
    }
    save_file_registry(FILE_REGISTRY)

# Load registry on startup
FILE_REGISTRY = load_file_registry()

# Synchronize FILE_REGISTRY with file_status status records
for fid, info in FILE_REGISTRY.items():
    try:
        current_status = file_status.get_status(fid)
        if not current_status.get("filename"):
            file_status.set_status(fid, {
                "uploaded": info.get("uploaded", True),
                "processing": False,
                "indexed": info.get("indexed", False),
                "error": None,
                "filename": info.get("filename"),
            })
            logger.info("Synchronized file_id %s from registry to status cache", fid)
    except Exception as e:
        logger.error("Failed to sync file_id %s to file_status: %s", fid, e)


def load_json_file(file_path: str, default: dict) -> dict:
    if not os.path.exists(file_path):
        return default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(file_path: str, data: dict):
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving to {file_path}: {e}")

async def _get_active_document_text() -> str:
    # 1. Try to read active document text from loaded vector db context
    if state.vector_db is not None:
        from research_analyzer import get_document_text
        text = await asyncio.to_thread(get_document_text, state.vector_db)
        if text:
            return text
            
    # 2. Try to fall back to the most recently loaded context
    if state.loaded_contexts:
        last_filename = list(state.loaded_contexts.keys())[-1]
        ctx = state.loaded_contexts[last_filename]
        from research_analyzer import get_document_text
        text = await asyncio.to_thread(get_document_text, ctx["db"])
        if text:
            return text
            
    # 3. Try to fall back to the most recently uploaded file in DATASET_DIR
    if os.path.exists(DATASET_DIR):
        files = [
            f for f in os.listdir(DATASET_DIR)
            if os.path.isfile(os.path.join(DATASET_DIR, f)) and f.endswith((".pdf", ".txt"))
        ]
        if files:
            files.sort(key=lambda x: os.path.getmtime(os.path.join(DATASET_DIR, x)), reverse=True)
            last_file = files[0]
            await _load_context_async(last_file)
            from research_analyzer import get_document_text
            text = await asyncio.to_thread(get_document_text, state.vector_db)
            if text:
                return text
                
    raise HTTPException(status_code=400, detail="No active document found. Please upload/load a document first.")

# Global dictionary to track evaluation job statuses
eval_jobs = {}

async def run_evaluation_task(job_id: str, request: EvaluateRequest):
    try:
        await _load_context_async(request.filename)
        from evaluator import run_full_evaluation
        
        def progress_callback(step_name, current_progress, total, result=None):
            eval_jobs[job_id]["step"] = step_name
            eval_jobs[job_id]["progress"] = current_progress
            eval_jobs[job_id]["total"] = total
            if result:
                eval_jobs[job_id]["result"] = result
                eval_jobs[job_id]["status"] = "completed"

        await run_full_evaluation(
            llm=get_llm(),
            rag_chain=state.qa_chain,
            db=state.vector_db,
            uploaded_files=[request.filename],
            progress_cb=progress_callback,
            mode=request.mode
        )
    except Exception as e:
        logger.error("Background evaluation failed: %s", e)
        eval_jobs[job_id]["status"] = "error"
        eval_jobs[job_id]["error"] = str(e)


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

# -------------------------------------------------------------------
# Indexing routine for lazy processing of uploaded documents
# -------------------------------------------------------------------
async def index_file(file_id: str, filename: str, file_path: str):
    """Create Chroma index for the given file_path and update status.

    This runs in a background task triggered by the first query on an
    uploaded document. It respects the MAX_CHUNKS, CHUNK_SIZE and
    CHUNK_OVERLAP settings defined in config.py and uses the lazy
    embedding model loader from services.resource_manager.
    """
    try:
        logger.info("INDEX START file_id=%s filename=%s path=%s", file_id, filename, file_path)
        file_status.set_status(file_id, {
            "uploaded": True,
            "processing": True,
            "indexed": False,
            "error": None,
            "filename": filename,
        })

        # Create or load the vector DB – heavy work off the event loop
        logger.info("Calling create_or_load_db for file_id=%s", file_id)
        await asyncio.to_thread(create_or_load_db, file_path)
        logger.info("create_or_load_db finished for file_id=%s", file_id)

        # Update status to reflect successful indexing
        file_status.set_status(file_id, {
            "uploaded": True,
            "processing": False,
            "indexed": True,
            "error": None,
            "filename": filename,
        })
        # Update indexed status in persistent registry
        if file_id in FILE_REGISTRY:
            FILE_REGISTRY[file_id]["indexed"] = True
            save_file_registry(FILE_REGISTRY)
        logger.info("INDEX COMPLETE file_id=%s filename=%s", file_id, filename)
    except Exception as e:
        logger.exception("Indexing failed for file_id=%s filename=%s path=%s", file_id, filename, file_path)
        # Record failure without overwriting uploaded flag
        file_status.set_status(file_id, {
            "uploaded": True,
            "processing": False,
            "indexed": False,
            "error": str(e),
            "filename": filename,
        })


@app.get("/files/status/{file_id}")
async def get_file_status(file_id: str):
    return file_status.get_status(file_id)

@app.post("/ask")
async def ask(request: QuestionRequest):
    logger.info("Entry /ask: %s", request)
    start_time = time.time()
    answer_text = ""
    try:
        # Determine which identifier to use for document loading
        file_identifier = request.file_id or request.filename
        if not file_identifier:
            raise HTTPException(status_code=400, detail="No file identifier provided")

        if request.file_id:
            # resolve file_id from registry
            file_info = FILE_REGISTRY.get(request.file_id)
            if not file_info:
                logger.error("Missing registry entry for file_id: %s", request.file_id)
                raise HTTPException(
                    status_code=400,
                    detail=f"No registry entry for file_id={request.file_id}"
                )
            
            file_path = file_info["path"]
            filename = file_info["filename"]
            logger.info("Registry lookup success: file_id=%s resolves to filename=%s, path=%s", request.file_id, filename, file_path)

            if not os.path.exists(file_path):
                logger.error("Missing file on disk: %s", file_path)
                raise HTTPException(
                    status_code=404,
                    detail=f"Stored file missing: {file_path}"
                )

            # Ensure the document is indexed before answering
            status = file_status.get_status(request.file_id)
            if not status.get("indexed"):
                # Trigger background indexing if not already started
                if not status.get("processing"):
                    file_status.set_status(request.file_id, {"processing": True})
                    asyncio.create_task(index_file(request.file_id, filename, file_path))
                raise HTTPException(status_code=503, detail="Document is being processed")
            
            # Load context using the resolved path and filename
            await _load_context_async(filename, file_path=file_path)
        else:
            # Fallback to legacy filename handling
            await _load_context_async(request.filename)

        if not state.qa_chain:
            raise HTTPException(status_code=503, detail="Context not loaded")

        ckey = f"ask:{file_identifier}:{request.question}:{request.mode}"
        cached = await cache_manager.get(ckey, category="rag_chat")
        
        if cached:
            result = cached
            if "pipeline" not in result:
                result["pipeline"] = {
                    "original_query": request.question,
                    "enhanced_query": request.question,
                    "retrieved_docs": [],
                    "reranked_docs": []
                }
            if "summary" not in result:
                result["summary"] = {
                    "short": "",
                    "simplified": "",
                    "detailed": ""
                }
            elif isinstance(result["summary"], dict):
                result["summary"].setdefault("short", "")
                result["summary"].setdefault("simplified", "")
                result["summary"].setdefault("detailed", "")
            answer_text = result.get("answer", "")
        else:
            answer = await state.qa_chain(request.question, mode=request.mode)
            answer_text = answer.get("answer", "")
            
            # Extract key points
            sentences = [s.strip() for s in answer_text.replace("\n", " ").split('. ') if s]
            key_points = sentences[:5]
            
            # Build initial result structure
            result = {
                "answer": answer_text,
                "key_points": key_points,
                "summary": answer.get("summary", {
                    "short": (sentences[0] + ".") if sentences else "",
                    "simplified": "",
                    "detailed": answer_text
                }),
                "pipeline": answer.get("pipeline", {
                    "original_query": request.question,
                    "enhanced_query": request.question,
                    "retrieved_docs": [],
                    "reranked_docs": []
                }),
                "sources": answer.get("sources", []),
                "execution_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            await cache_manager.set(ckey, result, category="rag_chat")

        logger.info("Result /ask: %s", result)
        

        # Update metrics in app_state.json
        if request.username:
            app_state = load_json_file(APP_STATE_FILE, {"users": {}})
            users_state = app_state.get("users", {})
            if request.username in users_state:
                mode_key = "Student" if request.mode == "student" else "Research"
                mode_state = users_state[request.username].get(mode_key, {})
                metrics = mode_state.get("metrics", {})
                
                total = metrics.get("total_queries", 0) + 1
                total_time = metrics.get("total_response_time_ms", 0.0) + result.get("execution_time_ms", 0.0)
                hits = metrics.get("cache_hits", 0) + (1 if result.get("cache_hit") else 0)
                misses = metrics.get("cache_misses", 0) + (0 if result.get("cache_hit") else 1)
                
                metrics["total_queries"] = total
                metrics["total_response_time_ms"] = total_time
                metrics["avg_response_time_ms"] = round(total_time / total, 2)
                metrics["cache_hits"] = hits
                metrics["cache_misses"] = misses
                metrics["cache_hit_rate"] = round(hits / total, 4)
                
                mode_state["metrics"] = metrics
                users_state[request.username][mode_key] = mode_state
                app_state["users"] = users_state
                save_json_file(APP_STATE_FILE, app_state)
                
        # Save to history.json
        if request.username and request.session_id:
            history = load_json_file(HISTORY_FILE, {})
            if request.username not in history:
                history[request.username] = {}
                
            session_id = request.session_id
            if session_id not in history[request.username]:
                title = (request.question[:30] + "…") if len(request.question) > 30 else request.question
                history[request.username][session_id] = {
                    "title": title,
                    "messages": []
                }
                
            history[request.username][session_id]["messages"].append({
                "role": "user",
                "content": request.question,
                "filename": request.filename
            })
            history[request.username][session_id]["messages"].append({
                "role": "assistant",
                "content": result.get("answer", ""),
                "filename": request.filename
            })
            save_json_file(HISTORY_FILE, history)
            
        # Ensure answer key exists
        result.setdefault('answer', answer_text)

        return result
    except Exception as e:
        logger.error("Error in /ask: %s", e, exc_info=True)
        detail = e.detail if isinstance(e, HTTPException) else str(e)
        return {
            "answer": f"An error occurred while processing your request: {detail}",
            "key_points": ["Request could not be completed."],
            "summary": {
                "short": "Error occurred",
                "simplified": "Error occurred",
                "detailed": detail
            },
            "pipeline": {
                "original_query": request.question,
                "enhanced_query": request.question,
                "retrieved_docs": [],
                "reranked_docs": []
            },
            "sources": [],
            "execution_time_ms": round((time.time() - start_time) * 1000, 2)
        }


@app.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    logger.info("Entry /evaluate: %s", request)
    job_id = str(uuid.uuid4())
    eval_jobs[job_id] = {
        "status": "running",
        "step": "Initializing",
        "progress": 0,
        "total": 3 if request.mode == "quick" else 10,
        "result": None,
        "error": None
    }
    
    asyncio.create_task(run_evaluation_task(job_id, request))
    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in eval_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return eval_jobs[job_id]


@app.post("/analyze/paper")
async def analyze_paper_endpoint(request: AnalyzePaperRequest):
    try:
        filename = request.filename
        if not filename:
            if state.loaded_contexts:
                filename = list(state.loaded_contexts.keys())[-1]
            elif os.path.exists(DATASET_DIR):
                files = [
                    f for f in os.listdir(DATASET_DIR)
                    if os.path.isfile(os.path.join(DATASET_DIR, f)) and f.endswith((".pdf", ".txt"))
                ]
                if files:
                    files.sort(key=lambda x: os.path.getmtime(os.path.join(DATASET_DIR, x)), reverse=True)
                    filename = files[0]
                    
        if not filename:
            raise HTTPException(status_code=400, detail="No active document found. Please load or upload a document first.")
            
        await _load_context_async(filename)
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


@app.post("/analyze/quality")
async def analyze_quality():
    try:
        text = await _get_active_document_text()
        from research_analyzer import analyze_paper_async
        result = await analyze_paper_async(get_llm(), text)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"scores": result.get("scores", {})}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/plagiarism")
async def analyze_plagiarism():
    try:
        text = await _get_active_document_text()
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="No active document text found.")

        # Lightweight literature overlap using fetched academic papers
        from services.academic_intelligence import run_full_academic_analysis_async

        analysis = await run_full_academic_analysis_async(text)
        papers = analysis.get("similar_papers", []) or []

        overlap_info = analysis.get("overlap_analysis", {}) or {}
        avg_overlap = float(overlap_info.get("similarity_percent", 0.0) or 0.0) / 100.0
        if avg_overlap >= 0.45:
            level = "High conceptual similarity"
        elif avg_overlap >= 0.22:
            level = "Moderate thematic overlap"
        else:
            level = "Low overlap"

        novelty_value = analysis.get("novelty_score")
        novelty_text = f"{round(float(novelty_value) * 10, 1)}%" if novelty_value is not None else "0%"
        overlap_text = overlap_info.get("plagiarism_message") or "Low confidence — insufficient comparison papers"

        return {
            "plagiarism": {
                "plagiarism_risk": level,
                "novelty_score": novelty_text,
                "overlap_analysis": overlap_text,
                "similar_papers_summary": [p.get("title", "") for p in papers[:5] if p.get("title")],
                "missing_references": [m.get("title", "") for m in (analysis.get("missing_citations") or [])[:5] if m.get("title")],
                "improvements": [s.get("suggestion", "") for s in (analysis.get("suggestions") or [])[:5] if s.get("suggestion")],
            },
            "literature_overlap_insights": {
                "level": level,
                "average_overlap_score": round(avg_overlap, 3),
                "reference_paper_count": len(papers),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/trends")
async def analyze_trends():
    try:
        text = await _get_active_document_text()
        from services.academic_intelligence import run_full_academic_analysis_async
        try:
            acad_res = await run_full_academic_analysis_async(text)
            trends_summary = acad_res.get("trends", {}).get("warning_message", "") or "Highly aligned with modern research."
        except Exception:
            trends_summary = "Stable"
            
        prompt = f"Analyze this research paper excerpt and provide a high-quality trend analysis paragraph (3-4 sentences) outlining its relevance to recent breakthroughs (2024-2026), modern methodologies, and industry standards: {text[:2000]}"
        try:
            llm = get_llm()
            raw_res = await llm.ainvoke(prompt)
            analysis_text = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
        except Exception as e:
            analysis_text = f"The paper's theme aligns with contemporary advancements. Trend: {trends_summary}."
            
        return {"trend_analysis": analysis_text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze/suggestions")
async def analyze_suggestions():
    try:
        text = await _get_active_document_text()
        from research_analyzer import analyze_paper_async
        result = await analyze_paper_async(get_llm(), text)
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return {"suggestions": result.get("suggestions", [])}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

        # Generate a UUID for this upload and register
        file_id = str(uuid.uuid4())
        register_uploaded_file(file_id, file.filename, file_path)

        file_status.set_status(file_id, {
            "uploaded": True,
            "processing": True,
            "indexed": False,
            "error": None,
            "filename": file.filename,
        })

        # Start indexing immediately after upload (non-blocking background task)
        asyncio.create_task(index_file(file_id, file.filename, file_path))

        # Optional eager preload (kept for backward compatibility)
        if os.getenv("PRELOAD_ON_UPLOAD", "false").lower() in ("1", "true", "yes"):
            await _load_context_async(file.filename)

        return {
            "status": "ok",
            "message": "Success",
            "file_id": file_id,
            "filename": file.filename
        }
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

# New endpoint to retrieve processing status for a specific upload
@app.get("/files/status/{file_id}")
async def get_file_status(file_id: str):
    return file_status.get_status(file_id)


@app.get("/history/{username}")
async def get_history(username: str):
    history = load_json_file(HISTORY_FILE, {})
    return {"history": history.get(username, {})}


@app.delete("/history/{username}/{session_id}")
async def delete_history(username: str, session_id: str):
    history = load_json_file(HISTORY_FILE, {})
    if username in history and session_id in history[username]:
        del history[username][session_id]
        save_json_file(HISTORY_FILE, history)
        return {"status": "success"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/login")
async def login(creds: dict):
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    users = load_json_file(USERS_FILE, {})
    if username not in users or users[username].get("password") != password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"status": "success", "username": username}


@app.post("/signup")
async def signup(creds: dict):
    username = creds.get("username")
    password = creds.get("password")
    if not username or not password:
        raise HTTPException(status_code=400, detail="Username and password are required")
    
    users = load_json_file(USERS_FILE, {})
    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    users[username] = {"password": password}
    save_json_file(USERS_FILE, users)
    return {"status": "success", "username": username}


@app.get("/state/{username}")
async def get_state(username: str):
    app_state = load_json_file(APP_STATE_FILE, {"users": {}})
    users_state = app_state.get("users", {})
    
    # Initialize default state for user if not exists
    if username not in users_state:
        users_state[username] = {
            "Student": {
                "chat": [],
                "last_pipeline": {},
                "last_input": "",
                "metrics": {
                    "total_queries": 0,
                    "total_response_time_ms": 0.0,
                    "avg_response_time_ms": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_hit_rate": 0.0
                }
            },
            "Research": {
                "chat": [],
                "last_pipeline": {},
                "last_input": "",
                "res_quality": None,
                "res_plagiarism": None,
                "res_trends": None,
                "res_suggestions": None,
                "insights": {
                    "quality": None,
                    "plagiarism": None,
                    "trends": None,
                    "suggestions": None
                },
                "metrics": {
                    "total_queries": 0,
                    "total_response_time_ms": 0.0,
                    "avg_response_time_ms": 0.0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_hit_rate": 0.0
                }
            },
            "Evaluate": {
                "eval_result": None,
                "eval_job_id": None,
                "last_input": "",
                "runs": []
            }
        }
        app_state["users"] = users_state
        save_json_file(APP_STATE_FILE, app_state)
        
    return {"state": users_state[username]}


@app.post("/state/{username}")
async def save_state(username: str, payload: dict):
    mode = payload.get("mode")  # e.g., "student", "researcher", "eval"
    data = payload.get("data", {})
    
    app_state = load_json_file(APP_STATE_FILE, {"users": {}})
    users_state = app_state.get("users", {})
    
    if username not in users_state:
        users_state[username] = {}
        
    mode_map = {
        "student": "Student",
        "researcher": "Research",
        "eval": "Evaluate",
        "eval_engine": "EvalEngine"
    }
    
    back_mode = mode_map.get(mode)
    if not back_mode:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
        
    current_mode_state = users_state[username].get(back_mode, {})
    
    if back_mode == "Student":
        current_mode_state["last_input"] = data.get("last_input", "")
        current_mode_state["last_pipeline"] = data.get("pipeline_data", {})
        
        messages = data.get("messages", [])
        chat_list = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                chat_list.append({
                    "query": messages[i].get("content", ""),
                    "response": messages[i+1].get("content", "")
                })
        current_mode_state["chat"] = chat_list
        
    elif back_mode == "Research":
        current_mode_state["last_input"] = data.get("last_input", "")
        current_mode_state["last_pipeline"] = data.get("pipeline_data", {})
        
        current_mode_state["res_quality"] = data.get("res_quality")
        current_mode_state["res_plagiarism"] = data.get("res_plagiarism")
        current_mode_state["res_trends"] = data.get("res_trends")
        current_mode_state["res_suggestions"] = data.get("res_suggestions")
        current_mode_state["insights"] = {
            "quality": data.get("res_quality"),
            "plagiarism": data.get("res_plagiarism"),
            "trends": data.get("res_trends"),
            "suggestions": data.get("res_suggestions")
        }
        
        messages = data.get("messages", [])
        chat_list = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                chat_list.append({
                    "query": messages[i].get("content", ""),
                    "response": messages[i+1].get("content", "")
                })
        current_mode_state["chat"] = chat_list
        
    elif back_mode == "Evaluate":
        current_mode_state["eval_result"] = data.get("eval_result")
        current_mode_state["eval_job_id"] = data.get("eval_job_id")
        current_mode_state["last_input"] = data.get("last_input", "")
        current_mode_state["runs"] = data.get("runs", [])
        
    users_state[username][back_mode] = current_mode_state
    app_state["users"] = users_state
    save_json_file(APP_STATE_FILE, app_state)
    return {"status": "success"}


@app.post("/state/{username}/switch")
async def switch_mode(username: str, payload: dict):
    mode = payload.get("mode")
    app_state = load_json_file(APP_STATE_FILE, {"users": {}})
    users_state = app_state.get("users", {})
    if username in users_state:
        users_state[username]["current_mode"] = mode
        app_state["users"] = users_state
        save_json_file(APP_STATE_FILE, app_state)
    return {"status": "success"}


@app.post("/state/{username}/clear_mode/{mode}")
async def clear_mode(username: str, mode: str):
    app_state = load_json_file(APP_STATE_FILE, {"users": {}})
    users_state = app_state.get("users", {})
    
    mode_map = {
        "student": "Student",
        "researcher": "Research",
        "eval": "Evaluate",
        "eval_engine": "EvalEngine"
    }
    
    back_mode = mode_map.get(mode)
    if username in users_state and back_mode in users_state[username]:
        if back_mode == "Student":
            users_state[username]["Student"] = {
                "chat": [],
                "last_pipeline": {},
                "last_input": "",
                "metrics": users_state[username]["Student"].get("metrics", {})
            }
        elif back_mode == "Research":
            users_state[username]["Research"] = {
                "chat": [],
                "last_pipeline": {},
                "last_input": "",
                "res_quality": None,
                "res_plagiarism": None,
                "res_trends": None,
                "res_suggestions": None,
                "insights": {
                    "quality": None,
                    "plagiarism": None,
                    "trends": None,
                    "suggestions": None
                },
                "metrics": users_state[username]["Research"].get("metrics", {})
            }
        elif back_mode == "Evaluate":
            users_state[username]["Evaluate"] = {
                "eval_result": None,
                "eval_job_id": None,
                "last_input": "",
                "runs": []
            }
        app_state["users"] = users_state
        save_json_file(APP_STATE_FILE, app_state)
        
    return {"status": "success"}


@app.post("/state/{username}/clear_all")
async def clear_all(username: str):
    app_state = load_json_file(APP_STATE_FILE, {"users": {}})
    users_state = app_state.get("users", {})
    
    if username in users_state:
        users_state[username]["Student"] = {
            "chat": [],
            "last_pipeline": {},
            "last_input": "",
            "metrics": users_state[username]["Student"].get("metrics", {})
        }
        users_state[username]["Research"] = {
            "chat": [],
            "last_pipeline": {},
            "last_input": "",
            "res_quality": None,
            "res_plagiarism": None,
            "res_trends": None,
            "res_suggestions": None,
            "insights": {
                "quality": None,
                "plagiarism": None,
                "trends": None,
                "suggestions": None
            },
            "metrics": users_state[username]["Research"].get("metrics", {})
        }
        users_state[username]["Evaluate"] = {
            "eval_result": None,
            "eval_job_id": None,
            "last_input": "",
            "runs": []
        }
        app_state["users"] = users_state
        save_json_file(APP_STATE_FILE, app_state)
        
    return {"status": "success"}


@app.get("/dashboard/{username}")
async def get_dashboard(username: str):
    app_state = load_json_file(APP_STATE_FILE, {"users": {}})
    user_data = app_state.get("users", {}).get(username, {})
    
    dash_dict = {}
    for key in ["Student", "Research", "Evaluate", "EvalEngine"]:
        metrics = user_data.get(key, {}).get("metrics", {})
        dash_dict[key] = {
            "total_queries": metrics.get("total_queries", 0),
            "avg_response_time_ms": metrics.get("avg_response_time_ms", 0.0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0.0)
        }
    return dash_dict


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
