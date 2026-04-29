import os
import sys
import time
import uuid
import json
import re
import hashlib
import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_ollama import OllamaLLM
from rag_pipeline import create_or_load_db, build_rag_chain
from research_analyzer import run_full_pipeline, get_document_text
from services.cache_manager import cache_manager
from services.async_utils import run_with_timeout
from routes.academic import router as academic_router
import evaluator

# Environment fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Windows encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Globals
qa_chain = None
vector_db = None
analyzer_llm = None
loaded_contexts = {} # filename -> {"db": db, "qa": qa}

# =========================
# 📦 HELPERS
# =========================
def _get_llm():
    global analyzer_llm
    if analyzer_llm is None:
        analyzer_llm = OllamaLLM(model="phi", num_predict=2048)
    return analyzer_llm

async def _load_context_async(filename: str):
    global qa_chain, vector_db
    if filename in loaded_contexts:
        vector_db = loaded_contexts[filename]["db"]
        qa_chain  = loaded_contexts[filename]["qa"]
        return

    file_path = os.path.join("dataset", filename)
    if not os.path.exists(file_path):
        raise HTTPException(404, "File not found")
    
    # Run heavy DB creation in a thread to not block
    db = await asyncio.to_thread(create_or_load_db, file_path)
    qa = await asyncio.to_thread(build_rag_chain, db)
    
    loaded_contexts[filename] = {"db": db, "qa": qa}
    vector_db = db
    qa_chain = qa

# =========================
# 🚀 APP LIFECYCLE
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🟢 Research Assistant Pipeline Online")
    yield
    logger.info("🔴 Pipeline Offline")

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Include Routers
app.include_router(academic_router, prefix="/academic")

@app.on_event("startup")
async def list_routes():
    for route in app.routes:
        logger.info(f"Route: {route.path} methods: {route.methods}")

# =========================
# 📥 MODELS
# =========================
class QuestionRequest(BaseModel):
    username: str
    question: str
    filename: str = None
    mode: str = "student"

class AnalyzePaperRequest(BaseModel):
    filename: str
    reference_text: str = ""

class EvaluateRequest(BaseModel):
    filename: str
    questions: list = None
    mode: str = "quick"

class SetFileRequest(BaseModel):
    filename: str

# =========================
# 🏠 ENDPOINTS
# =========================

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "ollama": "running",
            "rag": "ready" if qa_chain else "idle",
            "cache": "active"
        }
    }

@app.post("/ask")
async def ask(request: QuestionRequest):
    global qa_chain
    start_time = time.time()
    
    try:
        if request.filename:
            await _load_context_async(request.filename)
        
        if not qa_chain:
            raise HTTPException(503, "Context not loaded")

        # Check Cache
        ckey = f"ask:{request.filename}:{request.question}:{request.mode}"
        cached = await cache_manager.get(ckey, category="rag_chat")
        if cached:
            cached["cache_hit"] = True
            return cached

        # Run RAG async
        result = await qa_chain(request.question, mode=request.mode)
        
        elapsed = round((time.time() - start_time) * 1000, 2)
        result["execution_time_ms"] = elapsed
        result["cache_hit"] = False
        
        await cache_manager.set(ckey, result, category="rag_chat")
        return result

    except Exception as e:
        logger.error(f"Error in /ask: {e}")
        raise HTTPException(500, str(e))

@app.post("/evaluate")
async def evaluate_endpoint(request: EvaluateRequest):
    try:
        await _load_context_async(request.filename)
        llm = _get_llm()
        
        # We need a list of files for dataset info
        uploaded_files = [request.filename]
        
        result = await evaluator.run_full_evaluation(
            llm=llm,
            rag_chain=qa_chain,
            db=vector_db,
            uploaded_files=uploaded_files,
            mode=request.mode
        )
        return result
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        raise HTTPException(500, str(e))

@app.post("/analyze/paper")
async def analyze_paper_endpoint(request: AnalyzePaperRequest):
    try:
        await _load_context_async(request.filename)
        text = await asyncio.to_thread(get_document_text, vector_db)
        
        if not text:
            return {"status": "error", "message": "Could not extract text"}

        # Run full pipeline (Parallel APIs + Parallel LLM)
        llm = _get_llm()
        result = await run_full_pipeline(llm, text, reference_text=request.reference_text)
        return result

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(500, str(e))

@app.post("/set_file")
async def set_file(request: SetFileRequest):
    logger.info(f"📥 Received request to load file: {request.filename}")
    try:
        # Validate file exists
        file_path = os.path.join("dataset", request.filename)
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found in dataset: {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")
            
        # Load context
        logger.info(f"⏳ Loading context for {request.filename}...")
        await _load_context_async(request.filename)
        logger.info(f"✅ Context loaded successfully for {request.filename}")
        
        return {"message": "Success", "filename": request.filename}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in /set_file for {request.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("dataset", exist_ok=True)
        file_path = os.path.join("dataset", file.filename)
        content = await file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
            
        # Background: preload context
        await _load_context_async(file.filename)
        return {"message": "Success", "filename": file.filename}
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(500, str(e))

@app.get("/files")
async def list_files():
    try:
        dataset_path = os.path.join("dataset")
        if not os.path.isdir(dataset_path):
            return {"files": []}
        files = [
            f for f in os.listdir(dataset_path)
            if os.path.isfile(os.path.join(dataset_path, f))
        ]
        logger.info(f"📄 Files endpoint returned {len(files)} items")
        return {"files": files}
    except Exception as e:
        logger.error(f"Files endpoint error: {e}")
        raise HTTPException(500, str(e))

@app.get("/history/{username}")
async def get_history(username: str):
    # Simplified history fetch from cache or state
    return {"history": []}

@app.post("/login")
async def login(creds: dict):
    return {"status": "success", "username": creds.get("username")}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)