

# RagBench

**Production-grade RAG Evaluation Platform for Academic Research Papers**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-FF6B35?style=flat)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

</div>

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

---

## Screenshots

### Upload
<!-- SS: upload a PDF, show "Ingested successfully — chunks: 47" success card -->
![Upload](docs/screenshots/upload.png)

### Query
<!-- SS: question typed, phi3 selected, answer card visible, chunk viewer expanded -->
![Query](docs/screenshots/query.png)

### Evaluation — multi-model comparison
<!-- SS: same query across phi3 + mistral + gemini, metrics table with green best-score highlights -->
![Evaluate](docs/screenshots/evaluate.png)

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
└─────────────┘ └────────────┘ │ gemini-1.5 │
                               └─────┬──────┘
                          ┌──────────▼───────┐
                          │  Evaluation      │
                          │  ctx precision   │
                          │  faithfulness    │
                          │  answer relevancy│
                          └──────────────────┘

Infrastructure:
  ChromaDB  → Docker container  (port 8001)
  Ollama    → Mac native        (port 11434, accessed via host.docker.internal)
  API       → Docker container  (port 8000)
  Frontend  → Docker container  (port 3000)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Pydantic v2, Uvicorn |
| Vector DB | ChromaDB (persistent, cosine similarity) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Local LLMs | Ollama — phi3, mistral (native on host) |
| Cloud LLM | Gemini 1.5 Flash |
| Evaluation | Custom RAG metrics, semantic similarity |
| Frontend | React 18, Vite, Axios |
| Deployment | Docker Compose |

---

## Setup

### Prerequisites

- Docker + Docker Compose
- [Ollama](https://ollama.ai) installed and running natively on your machine
- Gemini API key (free tier works fine)

### 1 — Pull Ollama models (on your Mac, not Docker)

```bash
ollama pull phi3
ollama pull mistral
```

Verify Ollama is running:

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
CHROMA_PORT=8001
OLLAMA_HOST=http://host.docker.internal:11434
GEMINI_API_KEY=your_key_here
```

### 3 — Start

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:3000 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |

### Local dev (without Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

---

## Project Structure

```
ragbench/
├── backend/
│   ├── main.py
│   ├── core/
│   │   ├── config.py          # Pydantic settings, env vars
│   │   └── database.py        # ChromaDB client
│   ├── api/
│   │   ├── routes/            # ingest, query, evaluate, health
│   │   └── schemas/           # request + response models
│   └── services/
│       ├── ingestion/         # PDF processing, chunking, embedding
│       ├── rag/               # vector store, retriever, context builder
│       ├── llm/               # router, ollama client, gemini client
│       └── evaluation/        # evaluator, metrics
├── frontend/
│   └── src/
│       ├── pages/             # Upload, Query, Evaluate
│       └── components/        # Navbar, ChunkViewer, ComparisonTable
├── tests/
│   ├── test_rag.py
│   └── test_evaluation.py
├── scripts/
│   └── pull_models.sh
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

## Roadmap

- [ ] ARES / Ragas integration for standardised eval scores
- [ ] LangGraph agentic retrieval (multi-hop queries)
- [ ] Experiment tracking — save and compare eval runs over time
- [ ] Support for more file types (DOCX, TXT, MD)
- [ ] OpenAI GPT-4o as additional model option
- [ ] Automated dataset generation from uploaded papers

---

## License

MIT
