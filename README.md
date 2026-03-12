---
title: AI PDF Agent
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 📄 AI PDF Agent

<p align="center">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge"/>
<img src="https://img.shields.io/badge/LangGraph-Agent%20Workflow-blue?style=for-the-badge"/>
<img src="https://img.shields.io/badge/FAISS-Vector%20Database-orange?style=for-the-badge"/>
<img src="https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=for-the-badge&logo=huggingface"/>
<img src="https://img.shields.io/badge/Groq-LLM-black?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit"/>
<img src="https://img.shields.io/badge/Docker-Deployment-blue?style=for-the-badge&logo=docker"/>

</p>

A **production-grade Retrieval-Augmented Generation (RAG) system** that enables users to upload PDF documents, automatically summarize them, and ask context-aware questions with **source citations**.

The system uses **LangGraph agents, hybrid retrieval (BM25 + FAISS), and Groq-powered LLM inference** to generate accurate and explainable responses.

## Why This Project

Large Language Models cannot efficiently process long documents directly due to context limitations.

This project solves the problem using **Retrieval-Augmented Generation (RAG)**, which retrieves relevant document segments and supplies them as context to the language model.

This approach enables:

- accurate document question answering
- scalable document processing
- explainable answers with citations

---

# 🚀 Live Demo

Try the application directly on Hugging Face Spaces:

👉 https://huggingface.co/spaces/Soumya79/AI-Pdf-Agent

---

# ✨ Key Features

### 📚 Multi-PDF Document Ingestion
Upload and process **multiple PDF files simultaneously**.

### 🔎 Hybrid Retrieval
Custom retrieval system combining:

- **FAISS** for dense semantic vector search
- **BM25** for keyword-based sparse retrieval

This improves recall and retrieval accuracy.

### 🤖 Agent-Based Reasoning
LangGraph workflow orchestrates the reasoning pipeline:

- Retrieve relevant document chunks
- Generate answer using LLM
- Self-reflect and improve responses

### 📝 Automatic Document Summarization
Generates a concise summary immediately after PDF ingestion.

### 📑 Source Citations
Every answer includes:

- Source PDF filename
- Exact page number references

Ensuring **traceable and explainable responses**.

### 💬 Conversational Memory
Maintains context across questions for **natural follow-up interactions**.

---

# 🧠 System Architecture

```
User
 │
 ▼
Streamlit UI
 │
 ▼
PDF Upload
 │
 ▼
Document Processing Pipeline
 ├── PDF Loader
 ├── Text Chunking
 ├── Embedding Generation
 └── Vector Storage (FAISS)
 │
 ▼
Hybrid Retrieval System
 ├── FAISS Semantic Search
 └── BM25 Keyword Search
 │
 ▼
LangGraph Agent Workflow
 ├── Retrieve Context
 ├── Generate Answer
 └── Reflect & Improve
 │
 ▼
Groq LLM (LLaMA-3.1-8B)
 │
 ▼
Final Answer + Citations
```

Single-process architecture optimized for **Hugging Face Spaces deployment**.

---

# 🛠 Tech Stack

| Component | Technology |
|--------|--------|
| Frontend | Streamlit |
| Agent Framework | LangGraph |
| LLM Integration | LangChain |
| Vector Database | FAISS |
| Embeddings | HuggingFace Sentence Transformers |
| LLM Provider | Groq (LLaMA-3.1-8B-Instant) |
| Retrieval | Hybrid (BM25 + FAISS) |
| Deployment | Hugging Face Spaces + Docker |

---

# 📂 Project Structure

```
pdf-agent/

app.py
│
├── agents/
│   └── pdf_agent.py
│
├── rag/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   └── vectorstore.py
│
├── tools/
│   └── retrieval_tool.py
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# ⚙️ Local Development

## Prerequisites

- Python **3.10+**
- Groq API Key

Get one here:

https://console.groq.com/keys

---

## Installation

```bash
pip install -r requirements.txt
```

Create `.env` file:

```bash
echo 'GROQ_API_KEY="your-api-key-here"' > .env
```

Run the application:

```bash
streamlit run app.py
```

Access the app:

```
http://localhost:8501
```

---

# 🐳 Docker Deployment

Build the Docker image:

```bash
docker build -t pdf-agent .
```

Run container:

```bash
docker run -p 7860:7860 -e GROQ_API_KEY="your-groq-api-key" pdf-agent
```

Open:

```
http://localhost:7860
```

---

# ☁️ Hugging Face Spaces Deployment

### Step 1 — Create Space

Go to:

https://huggingface.co/spaces

Select:

```
SDK → Docker
```

---

### Step 2 — Add API Key

Navigate to:

```
Settings → Repository Secrets
```

Add:

```
GROQ_API_KEY = your_api_key
```

---

### Step 3 — Push Code

```bash
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME

git add .
git commit -m "Initial commit"

git push space main
```

Hugging Face will automatically build and deploy the application.

---

# 📸 Example Queries

```
Summarize the document
What methodology was used in the study?
Which datasets were used?
Explain the key findings of the paper
```

---

# 📈 Future Improvements

- Cross-Encoder Re-ranking for improved retrieval accuracy
- Multi-modal document support
- Research literature review generation
- Persistent vector database
- Knowledge graph integration

---

# 👨‍💻 Author

**Soumyadip Changder**

AI Engineer | Machine Learning | Agentic AI Systems

GitHub  
https://github.com/soumadipchangder

Hugging Face  
https://huggingface.co/Soumya79
