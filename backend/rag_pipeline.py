import os
import shutil
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector DB
from langchain_community.vectorstores import Chroma

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Ollama LLM
from langchain_ollama import OllamaLLM

# Prompt + parser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Cross-encoder for re-ranking
from sentence_transformers import CrossEncoder

# =========================
# 📄 LOAD DOCUMENT
# =========================
def load_document(file_path):
    if file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")


# =========================
# 🧠 CREATE / LOAD VECTOR DB
# =========================
def create_or_load_db(file_path):
    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Run Chroma entirely in memory to avoid Windows file locks (WinError 32)
    db = Chroma.from_documents(
        docs,
        embeddings
    )

    return db


# =========================
# ✨ QUERY ENHANCEMENT (ML Step 1)
# =========================
def enhance_query(llm, original_query):
    """Use the LLM to rewrite the user query for better retrieval."""
    enhance_prompt = PromptTemplate.from_template(
        """You are a search query optimizer. Rewrite the following user question 
into a more detailed, specific search query that will help retrieve the most 
relevant documents. Keep it concise (1-2 sentences max). 
Only output the rewritten query, nothing else.

Original question: {question}

Optimized search query:"""
    )

    try:
        formatted = enhance_prompt.format(question=original_query)
        enhanced = llm.invoke(formatted).strip()
        # Fallback if LLM returns empty or too long
        if not enhanced or len(enhanced) > 500:
            return original_query
        return enhanced
    except Exception as e:
        print(f"⚠️ Query enhancement failed: {e}")
        return original_query


# =========================
# ⭐ CROSS-ENCODER RE-RANKING (ML Step 2)
# =========================
_cross_encoder = None

def get_cross_encoder():
    """Lazy-load the cross-encoder model (singleton)."""
    global _cross_encoder
    if _cross_encoder is None:
        print("🔄 Loading cross-encoder model...")
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Cross-encoder loaded")
    return _cross_encoder


def rerank_documents(query, documents):
    """Re-rank documents using cross-encoder and return scored results."""
    if not documents:
        return []
    
    cross_encoder = get_cross_encoder()
    
    # Prepare pairs for cross-encoder
    pairs = [(query, doc.page_content) for doc in documents]
    
    # Get cross-encoder scores
    scores = cross_encoder.predict(pairs)
    
    # Combine docs with scores and sort by score descending
    scored_docs = []
    for doc, score in zip(documents, scores):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        source_label = os.path.basename(source)
        if page is not None:
            source_label += f" (Page {page + 1})"
        
        scored_docs.append({
            "content": doc.page_content,
            "score": float(score),
            "source": source_label
        })
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    
    return scored_docs


# =========================
# 🤖 BUILD RAG CHAIN (with full pipeline output)
# =========================
def build_rag_chain(db):
    llm = OllamaLLM(model="phi")

    prompt = PromptTemplate.from_template(
        """You are an intelligent assistant.

Use ONLY the context below to answer.
If answer not found, say "I don't know".

Context:
{context}

Question:
{input}

Answer:"""
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        """Execute full RAG pipeline and return structured result."""
        
        # Step 1: Enhance the query
        enhanced_query = enhance_query(llm, question)
        
        # Step 2: Retrieve documents using enhanced query
        raw_docs = retriever.invoke(enhanced_query)
        
        # Build retrieved docs list with basic similarity info
        retrieved_documents = []
        for i, doc in enumerate(raw_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            source_label = os.path.basename(source)
            if page is not None:
                source_label += f" (Page {page + 1})"
            
            retrieved_documents.append({
                "content": doc.page_content,
                "score": round(1.0 - (i * 0.1), 2),  # Approximate ranking score
                "source": source_label
            })
        
        # Step 3: Re-rank documents with cross-encoder
        reranked_documents = rerank_documents(enhanced_query, raw_docs)
        
        # Step 4: Generate answer using top re-ranked content
        # Use top 4 re-ranked docs for context
        top_docs_content = "\n\n".join(
            doc["content"] for doc in reranked_documents[:4]
        )
        formatted = prompt.format(context=top_docs_content, input=question)
        answer = llm.invoke(formatted)
        
        return {
            "original_query": question,
            "enhanced_query": enhanced_query,
            "retrieved_documents": retrieved_documents,
            "reranked_documents": reranked_documents,
            "answer": answer
        }

    return rag_chain


def get_qa_chain(file_path):
    db = create_or_load_db(file_path)
    return build_rag_chain(db)