import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA  # this should work after reinstall
from langchain_community.llms import HuggingFaceHub
from langchain.llms import HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()

def load_document(file_path):
    if file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    else:
        raise ValueError("Unsupported file")

def get_qa_chain(file_path):
    # 1. Load document
    documents = load_document(file_path)

    # 2. Split text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings()

    # 4. Vector DB
    db = FAISS.from_documents(docs, embeddings)

    # 5. LLM
    llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # 6. QA Chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    return qa