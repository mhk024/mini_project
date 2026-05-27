"""
Route for handling question answering.
"""

from fastapi import APIRouter, Request
from rag_pipeline import build_rag_chain

router = APIRouter()

@router.post("/ask")
async def ask(request: Request):
    """Handle a user question and return an answer JSON response.

    Returns a safe JSON payload even on errors.
    """
    payload = await request.json()
    question = payload.get("question")
    if not question:
        return {"error": "Question not provided"}
    # Build RAG chain; passing None uses stub if DB missing
    rag_chain = build_rag_chain(None)
    try:
        response = await rag_chain(question)
        return {
            "answer": response.get("answer"),
            "summary": response.get("summary"),
            "sources": response.get("sources"),
        }
    except Exception as e:
        return {
            "answer": "An error occurred while processing your request.",
            "summary": {"short": "Error", "detailed": str(e)},
            "sources": [],
        }
