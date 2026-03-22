from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from rag_pipeline import get_qa_chain

qa = None  # initialize

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global qa
    try:
        print("🔶 Loading RAG pipeline...")
        qa = get_qa_chain()
        print("✅ RAG Loaded!")
    except Exception as e:
        print("❌ ERROR LOADING RAG:", e)
    yield
    # Shutdown (add cleanup here if needed)

app = FastAPI(lifespan=lifespan)

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "RAG API is running"}

@app.post("/ask")
def ask(request: QuestionRequest):
    if qa is None:
        raise HTTPException(status_code=503, detail="RAG not loaded")
    try:
        answer = qa.run(request.question)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        # Read the file content
        content = await file.read()
        
        # Save the file to disk or process it with your RAG pipeline
        # Example:
        # with open(f"uploaded_{file.filename}", "wb") as f:
        #     f.write(content)
        
        # Here you would typically index the document into your vector database
        # For example: process_and_index_document(content)
        
        return {"filename": file.filename, "message": "File uploaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")