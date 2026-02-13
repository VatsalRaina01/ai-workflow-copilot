# AI Workflow Copilot

A production-grade retrieval-augmented chatbot with **multi-agent orchestration**, **hybrid search (Semantic + BM25)**, **guardrails**, and a modern React frontend — built with Python, FastAPI, LangChain, and ChromaDB.

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **Multi-Agent Routing** | LLM-based intent detection routes queries to RAG, Summarization, or Code agents |
| 🔍 **Hybrid Search** | Semantic similarity + BM25 keyword matching with Reciprocal Rank Fusion |
| 📄 **Multi-Format Upload** | PDF, DOCX, TXT, and Markdown parsing with drag & drop |
| 🛡️ **Guardrails** | Prompt injection detection, output sanitization, rate limiting |
| 💾 **Persistent Storage** | ChromaDB persistent client — documents survive restarts |
| 📊 **RAG Evaluation** | Faithfulness, relevance, and quality scoring via LLM-as-judge |
| 🧠 **Conversation Memory** | Context-aware responses via chat history injection |
| 🎨 **Markdown Rendering** | Rich AI responses with syntax-highlighted code blocks & tables |
| 📎 **Source Citations** | Expandable source cards showing which documents were used |
| ⚡ **Streaming SSE** | Real-time response streaming with agent metadata |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend (React + Vite)                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Chat UI  │ │ Markdown │ │ Drag&Drop│ │ Sidebar  │           │
│  │ + Stream │ │ Renderer │ │  Upload  │ │ Sessions │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ REST API + SSE
┌───────────────────────────▼─────────────────────────────────────┐
│                     Backend (FastAPI)                             │
│                                                                   │
│  ┌──────────────── Guardrails Layer ──────────────────────────┐  │
│  │  Prompt Injection Detection │ Rate Limiter │ PII Redaction │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌──────────────── Agent Router (Intent Detection) ───────────┐  │
│  │  ┌───────────┐  ┌─────────────┐  ┌───────────┐            │  │
│  │  │ RAG Agent │  │ Summarize   │  │ Code      │            │  │
│  │  │ (Doc Q&A) │  │ Agent       │  │ Agent     │            │  │
│  │  └─────┬─────┘  └─────────────┘  └───────────┘            │  │
│  └────────┼───────────────────────────────────────────────────┘  │
│           │                                                       │
│  ┌────────▼─────────────┐  ┌────────────────────────────────┐   │
│  │   Hybrid Search      │  │     GitHub Models API          │   │
│  │  ┌────────┐ ┌─────┐  │  │  ┌─────────┐ ┌───────────┐    │   │
│  │  │Semantic│ │BM25 │  │  │  │gpt-4o-  │ │text-embed │    │   │
│  │  │(Cosine)│ │(KW) │  │  │  │mini     │ │3-small    │    │   │
│  │  └────────┘ └─────┘  │  │  └─────────┘ └───────────┘    │   │
│  │  Reciprocal Rank     │  └────────────────────────────────┘   │
│  │  Fusion (RRF)        │                                        │
│  └──────────────────────┘  ┌────────────────────────────────┐   │
│                             │     Document Parser            │   │
│  ┌──────────────────────┐  │  PDF │ DOCX │ TXT │ Markdown   │   │
│  │  ChromaDB            │  └────────────────────────────────┘   │
│  │  (Persistent)        │                                        │
│  └──────────────────────┘  ┌────────────────────────────────┐   │
│                             │     RAG Evaluation             │   │
│                             │  Faithfulness │ Relevance      │   │
│                             └────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- GitHub Personal Access Token (with `models:read` permission)

### 1. Set up Environment

```bash
# Set your GitHub token
export GITHUB_TOKEN=your_github_token_here
```

### 2. Run Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Backend docs: http://localhost:8000/api/v1/docs

### 3. Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

### 4. (Optional) Docker

```bash
docker-compose up --build
```

## 📁 Project Structure

```
ai-workflow-copilot/
├── backend/
│   ├── app/
│   │   ├── agents/               # Multi-agent system (OOP)
│   │   │   ├── base.py           # Abstract base class (ABC)
│   │   │   ├── rag_agent.py      # RAG with hybrid search + citations
│   │   │   ├── summarize_agent.py# Summarization agent
│   │   │   ├── code_agent.py     # Code analysis agent
│   │   │   └── router.py         # LLM intent detection + routing
│   │   ├── api/endpoints/
│   │   │   └── chat.py           # REST API (13 endpoints)
│   │   ├── core/
│   │   │   └── config.py         # Pydantic settings
│   │   ├── services/
│   │   │   ├── github_models.py  # LLM integration (retry logic)
│   │   │   ├── vector_store.py   # ChromaDB + BM25 hybrid search
│   │   │   ├── document_parser.py# PDF/DOCX/TXT parser
│   │   │   ├── guardrails.py     # Security layer
│   │   │   ├── evaluation.py     # RAG quality metrics
│   │   │   ├── llm_service.py    # LangChain memory integration
│   │   │   └── workflow_engine.py# Multi-step workflow executor
│   │   ├── models/
│   │   │   └── workflow.py       # SQLModel schemas
│   │   └── main.py               # App entry + lifecycle events
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   └── ChatPage.jsx      # Full chat UI with all features
│   │   ├── services/
│   │   │   └── api.js            # API client (streaming + metadata)
│   │   ├── App.jsx
│   │   └── index.css             # Glass-morphism dark theme
│   └── package.json
├── docker-compose.yml
└── README.md
```

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/chat` | POST | Send message with multi-agent routing |
| `/api/v1/chat/stream` | POST | Stream response via SSE with agent metadata |
| `/api/v1/documents` | POST | Upload document (PDF, DOCX, TXT, MD) |
| `/api/v1/documents/text` | POST | Add text content directly |
| `/api/v1/documents/count` | GET | Get indexed chunk count |
| `/api/v1/documents/{id}` | DELETE | Delete a specific document |
| `/api/v1/evaluate` | POST | Evaluate RAG response quality |
| `/api/v1/agents` | GET | List available agents |
| `/api/v1/clear` | POST | Clear conversation history |
| `/api/v1/health` | GET | Health check with system info |

## 🎯 How It Works

### Multi-Agent Routing
```
User Query → Intent Classification (LLM) → Agent Selection
                                              ├─ RAG Agent      (document Q&A)
                                              ├─ Summarize Agent (text condensation)
                                              └─ Code Agent     (code analysis)
```

### RAG Pipeline
1. **Upload** — Upload PDF, DOCX, TXT, or Markdown files
2. **Parse** — Multi-format document parser extracts text
3. **Chunk** — `RecursiveCharacterTextSplitter` creates semantic chunks
4. **Embed** — Chunks → vectors via `text-embedding-3-small`
5. **Store** — Vectors in ChromaDB (persistent), text in BM25 index
6. **Query** — User asks a question
7. **Hybrid Search** — Semantic cosine + BM25 keyword → RRF fusion
8. **Generate** — GPT-4o-mini generates answer with source citations
9. **Stream** — Response streams to UI in real-time with agent metadata

### Guardrails
- **Input**: Regex-based prompt injection detection + message length limits
- **Output**: PII redaction (email, phone, SSN, credit card)
- **Rate**: Token-bucket rate limiting per client

### Evaluation (LLM-as-Judge)
- **Faithfulness** — Is the answer grounded in sources? (0.0–1.0)
- **Relevance** — Are the retrieved sources relevant? (0.0–1.0)
- **Quality** — Overall response accuracy & clarity (0.0–1.0)

## 🧱 Design Principles

- **OOP** — Abstract base class (`BaseAgent`) → polymorphic agents
- **Modular Architecture** — Separate agents, services, API, and core layers
- **Data Validation** — Pydantic request/response models throughout
- **Scalability** — Async endpoints, streaming, persistent storage
- **Clean Code** — Type hints, docstrings, structured logging, error handling

## ⚙️ Configuration

All settings configurable via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | — | GitHub PAT for Models API |
| `CHAT_MODEL` | `gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `500` | Text chunk size |
| `CHUNK_OVERLAP` | `50` | Chunk overlap |
| `TOP_K_RESULTS` | `5` | Search results count |
| `BM25_WEIGHT` | `0.3` | BM25 vs semantic weight |
| `GUARDRAILS_ENABLED` | `true` | Enable/disable guardrails |
| `MAX_REQUESTS_PER_MINUTE` | `30` | Rate limit |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## 📜 License

MIT
