import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

def load_document(file_path):
    if file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    else:
        raise ValueError("Unsupported file type")

def get_qa_chain(file_path):
    documents = load_document(file_path)
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    llm = HuggingFaceEndpoint(
        repo_id="google/flan-t5-base",
        task="text2text-generation",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
Context: {context}
Question: {input}
""")
    combine_chain = create_stuff_documents_chain(llm, prompt)
    qa = create_retrieval_chain(db.as_retriever(), combine_chain)
    return qa