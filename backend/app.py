import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"

import warnings
warnings.filterwarnings("ignore")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from rag_pipeline import create_or_load_db, build_rag_chain
import json

USERS_FILE = "users.json"
HISTORY_FILE = "history.json"

# =========================
# 📦 STORAGE HELPERS
# =========================
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f)


qa = None
db = None   # ✅ NEW: store DB

# =========================
# 🚀 APP LIFECYCLE
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🟢 Server started. Upload a document to initialize RAG.")
    yield
    print("🔴 Server shutting down.")

app = FastAPI(lifespan=lifespan)


# =========================
# 📥 REQUEST MODELS
# =========================
class QuestionRequest(BaseModel):
    username: str
    question: str
    filename: str = None
    session_id: str = None

class FilenameRequest(BaseModel):
    filename: str

class UserCredentials(BaseModel):
    username: str
    password: str


# =========================
# 🏠 ROUTES
# =========================
@app.get("/")
def home():
    return {"message": "RAG API is running"}


# =========================
# 💬 ASK QUESTION
# =========================
@app.post("/ask")
def ask(request: QuestionRequest):
    global qa

    if qa is None:
        raise HTTPException(status_code=503, detail="RAG not loaded.")

    try:
        print("🔥 Question:", request.question)

        # ✅ FIXED
        answer = qa(request.question)

        print("✅ Answer:", answer)

        # Save history
        history = load_history()
        
        # Legacy migration: if it's a list, convert to session dict
        user_data = history.get(request.username)
        if isinstance(user_data, list):
            history[request.username] = {
                "legacy_session": {
                    "title": "Legacy Chat",
                    "messages": user_data
                }
            }
        elif user_data is None:
            history[request.username] = {}

        user_history = history[request.username]
        sess_id = request.session_id or "default"

        if sess_id not in user_history:
            user_history[sess_id] = {
                "title": request.question[:30] + "..." if len(request.question) > 30 else request.question,
                "messages": []
            }
        
        user_history[sess_id]["messages"].append({
            "role": "user", 
            "content": request.question,
            "filename": request.filename
        })
        user_history[sess_id]["messages"].append({
            "role": "assistant", 
            "content": answer,
            "filename": request.filename
        })

        save_history(history)

        return {"answer": answer}

    except Exception as e:
        print("❌ ERROR in /ask:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 📂 UPLOAD FILE
# =========================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global qa, db

    try:
        print("🚀 Upload started")

        os.makedirs("dataset", exist_ok=True)

        file_path = f"dataset/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        print("📂 File uploaded:", file.filename)

        # Step 1: Create DB
        try:
            print("🔄 Creating DB...")
            db = create_or_load_db(file_path)
            print("✅ DB created")
        except Exception as e:
            print("❌ DB ERROR:", e)
            raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")

        # Step 2: Build RAG
        try:
            print("🔄 Building RAG chain...")
            qa = build_rag_chain(db)
            print("✅ RAG chain built")
        except Exception as e:
            print("❌ RAG ERROR:", e)
            raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

        return {"message": "File uploaded & processed successfully"}

    except Exception as e:
        print("❌ FINAL ERROR in /upload:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 🔁 SWITCH FILE
# =========================
@app.post("/set_file")
def set_file(request: FilenameRequest):
    global qa, db

    try:
        file_path = os.path.join("dataset", request.filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        print("📂 Switching file:", request.filename)

        # Step 1: Load DB
        try:
            print("🔄 Loading DB...")
            db = create_or_load_db(file_path)
            print("✅ DB loaded")
        except Exception as e:
            print("❌ DB ERROR:", e)
            raise HTTPException(status_code=500, detail=f"DB Error: {str(e)}")

        # Step 2: Build RAG
        try:
            print("🔄 Building RAG chain...")
            qa = build_rag_chain(db)
            print("✅ RAG ready")
        except Exception as e:
            print("❌ RAG ERROR:", e)
            qa = None
            raise HTTPException(status_code=500, detail=f"RAG Error: {str(e)}")

        return {"message": f"Context switched to {request.filename}"}

    except Exception as e:
        print("❌ FINAL ERROR in /set_file:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# 👤 AUTH
# =========================
@app.post("/signup")
def signup(user: UserCredentials):
    users = load_users()

    if user.username in users:
        raise HTTPException(status_code=400, detail="Username already exists")

    users[user.username] = {"password": user.password}
    save_users(users)

    return {"message": "User registered successfully"}


@app.post("/login")
def login(user: UserCredentials):
    users = load_users()

    if user.username not in users or users[user.username]["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"message": "Login successful"}


# =========================
# 📜 HISTORY
# =========================
@app.get("/history/{username}")
def get_history(username: str):
    history = load_history()
    user_data = history.get(username, {})
    
    # Handle legacy format where history was just an array
    if isinstance(user_data, list):
        if len(user_data) > 0:
            user_data = {
                "legacy_session": {
                    "title": "Legacy Chat",
                    "messages": user_data
                }
            }
        else:
            user_data = {}

    return {"history": user_data}


# =========================
# 📁 FILE LIST
# =========================
@app.get("/files")
def list_files():
    if not os.path.exists("dataset"):
        return {"files": []}

    files = os.listdir("dataset")
    return {"files": [f for f in files if f.endswith((".txt", ".pdf"))]}