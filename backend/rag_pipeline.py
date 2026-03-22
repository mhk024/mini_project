import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA  # this should work after reinstall
from langchain_community.llms import HuggingFaceHub


def get_qa_chain():
    # 1. Load document
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "dataset", "sample.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split text
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = HuggingFaceEmbeddings()

    # 4. Store in vector DB
    db = FAISS.from_documents(docs, embeddings)

    # 5. Load LLM (HuggingFace)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

    # 6. Create QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    return qa