# Research Assistant Mini Project

## Project Description
This project is an AI-powered Research Assistant application designed to help users perform academic research, document analysis, and intelligent query processing. The system provides an interactive interface for managing and analyzing research-related tasks efficiently.

---

## Technologies and Tools Used

### Backend
- Python
- FastAPI
- REST API

### Frontend
- Streamlit

### Tools & Platforms
- Git & GitHub
- VS Code
- Render Deployment(frontend)
- Railway Deployment(backend)

### Libraries
- OpenAI APIs
- Semantic Scholar Integration
- OpenAlex Integration

---

## Features and Functionalities

- Academic paper analysis
- AI-powered query assistance
- Research data processing
- Streamlit-based user interface
- Backend API integration
- Document and semantic search
- Cached research responses
- Evaluation and testing modules

---

## Installation / Execution Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/mhk024/mini_project.git
```

### Step 2: Move into Project Folder

```bash
cd mini_project
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run Backend Server

```bash
uvicorn backend.app:app --reload
```

### Step 5: Run Frontend

```bash
streamlit run frontend/app.py
```

---

## Team Members

- MAHAK RATHOD
- MAHAK SACHDEV
- MANVI AGRAWAL

---

## Project Structure

```text
mini_project/
│
├── README.md                     # Project documentation
├── report.docx                   # Mini project report
├── requirements.txt              # Python dependencies
├── render.yaml                   # Deployment configuration
├── .gitignore                    # Ignored files/folders
│
├── backend/                      # FastAPI Backend
│   │
│   ├── app.py                    # Main FastAPI application
│   ├── config.py                 # Backend configuration settings
│   ├── evaluation.py             # Evaluation/testing logic
│   ├── rag_pipeline.py           # Retrieval-Augmented Generation pipeline
│   ├── research_analyzer.py      # Research analysis module
│   ├── state.py                  # State management
│   ├── app_state.json            # Application state data
│   ├── history.json              # Stored history data
│   ├── users.json                # User information
│   │
│   ├── routes/                   # API route handlers
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── services/                 # Backend service modules
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── dataset/                  # Dataset and research files
│   │   └── ...
│   │
│   ├── chroma_db/                # Vector database storage
│   │   └── ...
│   │
│   └── __pycache__/              # Python cache files
│
├── frontend/                     # Streamlit Frontend
│   │
│   ├── app.py                    # Main Streamlit frontend app
│   ├── requirements.txt          # Frontend dependencies
│   └── __pycache__/              # Python cache files
│
├── scratch/                      # Testing and experimental scripts
│   │
│   ├── check_health.py           # Health check script
│   ├── test_eval.py              # Evaluation testing
│   └── test_rag.py               # RAG testing module
│
├── screenshots/                  # Project screenshots/output
│   ├── home.png
│   └── response.png
│
└── assets/                       # Additional project assets
    └── ...
```

---

## Screenshots / Output
home.png
response.png

---

## Conclusion

This mini project demonstrates the practical implementation of AI-assisted research workflows using Python, Flask, and Streamlit with modern development tools and APIs.
