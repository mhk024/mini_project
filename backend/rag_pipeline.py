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
import shutil

def create_or_load_db(file_path, persist_dir="chroma_db"):
    documents = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Always delete old DB and recreate fresh
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=persist_dir
    )

    return db


# =========================
# 🤖 BUILD RAG CHAIN
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

    retriever = db.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def rag_chain(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        formatted = prompt.format(context=context, input=question)
        return llm.invoke(formatted)

    return rag_chain


def get_qa_chain(file_path):
    db = create_or_load_db(file_path)
    return build_rag_chain(db)