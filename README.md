---
title: AI PDF Agent
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# AI PDF Agent

A production-ready Retrieval-Augmented Generation (RAG) system for querying and summarizing PDF documents. Uses LangGraph to perform self-reflection for high-quality responses.

## Features

- **Multi-Document Ingestion**: Upload and process multiple PDFs simultaneously.
- **Hybrid Retrieval**: `EnsembleRetriever` combining FAISS (dense) + BM25 (sparse) for high recall.
- **Self-Reflecting Agent**: LangGraph workflow generates, evaluates, and rewrites answers for quality.
- **Automatic Summarization**: Generates a high-level summary upon document upload.
- **Source Citations**: Returns exact source filename and page numbers with every answer.
- **Conversational Memory**: Maintains context for natural follow-up questions.

## Architecture

- **Frontend**: Streamlit (`app.py`) — interactive chat UI
- **Backend**: Flask API (`backend/server.py`) — orchestration and routing
- **LLM Engine**: Groq (`llama-3.1-8b-instant`) via LangChain
- **Vector DB**: FAISS (local `faiss_index/` directory)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace

## Deployment on Hugging Face Spaces

### 1. Create a New Space
- Go to [huggingface.co/spaces](https://huggingface.co/spaces) → **Create New Space**
- Select **Docker** as the SDK

### 2. Set the GROQ_API_KEY Secret
- Go to your Space **Settings → Repository Secrets**
- Add a secret named `GROQ_API_KEY` with your key from [console.groq.com](https://console.groq.com/keys)

### 3. Push the Code
```bash
git init
git remote add origin https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
git add .
git commit -m "Initial commit"
git push origin main
```
> **Important**: Make sure `.env` is in `.gitignore` so you never commit your API key!

The Space will build and expose the Streamlit UI on port 7860 automatically.

---

## Local Development

### Prerequisites
- Python 3.10+
- A valid [Groq API Key](https://console.groq.com/keys)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create .env file
echo 'GROQ_API_KEY="your-key-here"' > .env

# Terminal 1 - Start Flask backend
PYTHONPATH=. python3 backend/server.py

# Terminal 2 - Start Streamlit frontend
streamlit run app.py
```

### Docker (Local)
```bash
docker build -t pdf-agent .
docker run -p 7860:7860 -e GROQ_API_KEY="your-groq-api-key" pdf-agent
```
Access the app at [http://localhost:7860](http://localhost:7860)

## Directory Structure

```
├── app.py                     # Streamlit Frontend
├── backend/
│   └── server.py              # Flask REST Backend
├── agents/
│   └── pdf_agent.py           # LangGraph Agent
├── rag/
│   ├── loader.py              # PDF ingestion
│   ├── chunking.py            # Text splitting
│   ├── embeddings.py          # HuggingFace embedding model
│   └── vectorstore.py         # FAISS DB management
├── tools/
│   └── retrieval_tool.py      # Hybrid FAISS+BM25 retriever
├── requirements.txt
├── Dockerfile
└── README.md
```
