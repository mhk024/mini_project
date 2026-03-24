from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from rag_pipeline import get_qa_chain
import os
import json

USERS_FILE = "users.json"

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

qa = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Skip preloading — qa starts as None, /upload sets it
    print("🟢 Server started. Upload a document to initialize RAG.")
    yield

app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str

class UserCredentials(BaseModel):
    username: str
    password: str

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    if qa is None:
        raise HTTPException(status_code=503, detail="RAG not loaded. Please upload a document first.")
    try:
        result = qa.invoke({"input": request.question})  # updated from qa.run()
        answer = result["answer"]
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global qa
    try:
        os.makedirs("dataset", exist_ok=True)  # create folder if missing
        save_path = f"dataset/{file.filename}"
        with open(save_path, "wb") as f:
            f.write(await file.read())
        qa = get_qa_chain(save_path)
        return {"message": "File uploaded & processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=401, detail="Invalid username or password")
    return {"message": "Login successful", "username": user.username}