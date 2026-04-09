# RagBench

**Production-grade RAG Evaluation Platform for Academic Research Papers**

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

---

## Why RagBench?

Most RAG demos just show that retrieval works. RagBench measures *how well* it works.

| Feature | RagBench | Basic RAG demo |
|---|---|---|
| Multi-model parallel generation | ✅ | ❌ |
| RAG-aware metrics (precision, faithfulness) | ✅ | ❌ |
| Retrieval debugger (chunk scores visible) | ✅ | ❌ |
| Side-by-side model comparison | ✅ | ❌ |
| Production architecture (layered services) | ✅ | ❌ |

Built to answer a real question: **given the same retrieved context, which LLM produces the most faithful, relevant answer?**

---

## What it does

Single `/evaluate` endpoint runs the full pipeline:

```
Upload PDF → Chunk → Embed → ChromaDB
                                ↓
Query → Vector Search → Context → LLM Router → parallel generation
                                                      ↓
                                              Evaluation Engine
                                              (per model, per query)
```

Metrics computed per model per query:

- **Context Precision** — what fraction of retrieved chunks actually support the answer
- **Context Recall** — how much of the ground truth is covered by retrieved context
- **Faithfulness** — how grounded the answer is in retrieved chunks (not hallucinated)
- **Answer Relevancy** — semantic similarity between query and final answer
- **Agentic self-critique** — Gemini evaluates its own answer for hallucination and autonomously retries with a stricter prompt if needed

---

## Screenshots

### Upload
<img width="1460" height="419" alt="Pasted Graphic 1" src="https://github.com/user-attachments/assets/639a6d82-2c0e-4509-93a1-f8ee8c174d52" />


### Query
<img width="1470" height="824" alt="Pasted Graphic" src="https://github.com/user-attachments/assets/dda5dd7a-e18c-41a3-9661-042c570164f4" />


### Evaluation — multi-model comparison
<img width="1470" height="355" alt="Pasted Graphic 1" src="https://github.com/user-attachments/assets/24064537-0f67-452d-b129-b30fc061da80" />


> **Note:** Local model evaluation (phi3, mistral) requires 16GB+ RAM for concurrent multi-model generation. Screenshots were captured using Gemini due to hardware constraints during development. The full pipeline runs correctly with all models in a sufficiently resourced environment.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  Frontend  (React + Vite, port 3000)        │
└───────────────────┬─────────────────────────┘
                    │ HTTP
┌───────────────────▼─────────────────────────┐
│  FastAPI Gateway  (port 8000)               │
│  /ingest  /query  /evaluate  /health        │
└──────┬──────────────┬───────────────┬───────┘
       │              │               │
┌──────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│  Ingestion  │ │    RAG     │ │    LLM     │
│  PDF parse  │ │  Retriever │ │   Router   │
│  Chunk+Embed│ │  Context   │ │ phi3       │
│  ChromaDB   │ │  Builder   │ │ mistral    │
└─────────────┘ └────────────┘ │ gemini-2.0 │
                               └─────┬──────┘
                          ┌──────────▼───────┐
                          │  Evaluation      │
                          │  ctx precision   │
                          │  faithfulness    │
                          │  answer relevancy│
                          └──────────────────┘

Infrastructure:
  ChromaDB  → persistent HTTP server  (port 8001)
  Ollama    → Mac native              (port 11434)
  API       → Uvicorn / Docker        (port 8000)
  Frontend  → Vite / Docker           (port 3000)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Local LLMs | Ollama — phi3, mistral |
| Cloud LLM | Gemini 2.0 Flash |
| Evaluation | Custom RAG metrics, semantic similarity |
| Frontend | React 18, Vite, Axios |
| Deployment | Docker Compose |

---

## Setup

### Prerequisites

- Docker + Docker Compose (optional, for containerised setup)
- [Ollama](https://ollama.ai) installed and running natively
- Gemini API key — free tier at [aistudio.google.com](https://aistudio.google.com)

### 1 — Pull Ollama models

```bash
ollama pull phi3
ollama pull mistral
```

Verify:

```bash
ollama list
```

### 2 — Clone and configure

```bash
git clone https://github.com/VanshikaLud04/ragbench
cd ragbench
cp .env.example .env
```

Edit `.env`:

```dotenv
CHROMA_HOST=localhost
CHROMA_PORT=8001
OLLAMA_HOST=http://localhost:11434
GEMINI_API_KEY=your_key_here
```

> If running inside Docker, use `OLLAMA_HOST=http://host.docker.internal:11434` and `CHROMA_HOST=chromadb`.

### 3 — Start ChromaDB

```bash
chroma run --host localhost --port 8001 --path ./chroma_data
```

### 4 — Start the API

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd ragbench
uvicorn backend.main:app --reload
```

### 5 — Start the frontend

```bash
cd ragbench/frontend/frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |

### Docker (full stack)

```bash
docker compose up --build
```

---

## Project Structure

```
ragbench/
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── config.py          # Pydantic settings, env vars
│   │   └── database.py        # ChromaDB HttpClient
│   ├── api/
│   │   ├── routes/            # ingest, query, evaluate, health
│   │   └── schemas/           # request + response models
│   └── services/
│       ├── ingestion/         # PDF processing, chunking, embedding
│       ├── rag/               # vector store, retriever, context builder
│       ├── llm/               # router, ollama client, gemini client
│       └── evaluation/        # evaluator, metrics
├── frontend/
│   └── frontend/
│       └── src/
│           ├── pages/         # Upload, Query, Evaluate
│           └── components/    # Navbar, ChunkViewer, ComparisonTable
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/ingest/` | Upload and ingest a PDF |
| `POST` | `/query/` | RAG query with single model |
| `POST` | `/evaluate/` | Multi-model eval with full metrics |
| `GET` | `/health/` | Health check |

Full interactive docs at `http://localhost:8000/docs`

---

## Known Limitations

- Concurrent local model evaluation (phi3 + mistral simultaneously) requires 16GB+ RAM
- ChromaDB must be running as a separate HTTP server before starting the API
- Gemini model name must match the current `google-genai` SDK — currently `gemini-2.0-flash`

---

## Roadmap

- [ ] ARES / Ragas integration for standardised eval scores
- [ ] Experiment tracking — save and compare eval runs over time
- [ ] Support for more file types (DOCX, TXT, MD)
- [ ] OpenAI GPT-4o as additional model option
- [ ] Automated dataset generation from uploaded papers

---

## License

MIT
