# 🤖 Agentic Copilot

A production-grade AI Agent platform built to learn and implement real-world AI infrastructure patterns.

**This is not a toy chatbot.** This is a modular, scalable AI agent system built like real AI infrastructure teams do—with clean architecture, proper abstractions, and production deployment patterns.

---

## 🎯 Vision

Build a complete AI Agent platform that demonstrates:

- **RAG (Retrieval-Augmented Generation)** with vector databases
- **ReAct agent loop** (manual implementation first, then LangGraph)
- **Tool registry and execution system**
- **Short-term and long-term memory**
- **Evaluation system** for measuring accuracy
- **Observability** (cost tracking, tracing, logging)
- **Production deployment patterns**

---

## 🏗️ Tech Stack

### Backend

- **Python 3.12**
- **FastAPI** - Modern async web framework
- **SQLAlchemy 2.0** - Type-safe ORM
- **Alembic** - Database migrations
- **Pydantic v2** - Request/response validation

### Database

- **PostgreSQL 16** - Primary database
- **pgvector** (coming in M2) - Vector similarity search
- **Redis 7** - Caching and job queue

### Infrastructure

- **Docker Compose** - Local development
- **Poetry/pip** - Python dependency management

---

## 🏛️ Architecture Principles

### 1. **Clean Separation of Concerns**

```
app/       → API layer (FastAPI routers, schemas)
core/      → Business logic (AI, chunking, embeddings, agents)
db/        → Persistence layer (future: repositories)
```

### 2. **Interface-Driven Design**

Everything behind interfaces for flexibility:

- `VectorStore` interface → `PgVectorStore`, `QdrantStore`
- `LLMClient` interface → `OpenAIClient`, `AnthropicClient`
- `EmbeddingsClient` interface → `OpenAIEmbeddings`
- `Tool` interface → `RetrieveTool`, `WebSearchTool`

### 3. **No Framework Lock-In**

Core AI logic is independent of FastAPI. Can be used by:

- API endpoints
- Background workers
- CLI scripts
- Jupyter notebooks

### 4. **Production-Ready Patterns**

- Proper error handling
- Type hints everywhere
- Database migrations
- Health checks
- Request validation
- Graceful shutdowns

---

## 📍 Milestone Roadmap

### ✅ **M0 – Infrastructure & Database**

**Status:** Complete

- Docker Compose setup (API + Postgres + Redis)
- Alembic migrations working
- `workspaces` table created
- Environment configuration

---

### ✅ **M1 – Document Ingestion & Chunking**

**Status:** Complete

**Goal:** System can store knowledge

**Delivered:**

- ✅ `documents` table (linked to workspaces)
- ✅ `chunks` table (linked to documents)
- ✅ Character-based chunking with overlap (`core/chunking.py`)
- ✅ `POST /workspaces/{id}/docs` - upload & chunk documents
- ✅ `GET /workspaces/{id}/docs` - list documents with pagination
- ✅ `GET /workspaces/{id}/docs/{id}` - get document + chunks
- ✅ `GET /workspaces/{id}/docs/{id}/chunks` - get all chunks

**Architecture:**

```
Document (1) ─→ (N) Chunks
   ↓
Workspace
```

---

### 🎯 **M2 – Embeddings + Vector Search (pgvector)**

**Status:** Next

**Goal:** Semantic retrieval system

**Deliverables:**

- Install and enable `pgvector` extension
- Add `embedding vector(1536)` column to `chunks`
- `EmbeddingsClient` interface in `core/embeddings/`
- Background job to embed chunks
- `/search?q=...` endpoint with cosine similarity
- Basic retrieval evaluation

---

### 📋 **M3 – RAG Answering with Citations**

**Status:** Planned

**Goal:** True Retrieval-Augmented Generation

**Deliverables:**

- `/chat` endpoint
- Retrieve top-k chunks via vector search
- Inject into LLM prompt
- Return answer + citations
- Prompt templates in `core/prompts/`

---

### 📋 **M4 – Minimal ReAct Agent (Manual)**

**Status:** Planned

**Goal:** Understand agents deeply (no frameworks)

**Deliverables:**

- Tool base interface
- Tool registry and schema validation
- Manual ReAct loop:
  - Think
  - Act (call tool)
  - Observe
  - Repeat
- Store tool calls in DB
- Max step control + safety guardrails

**Why manual first?**  
Understanding the loop before using LangGraph helps avoid black-box pitfalls.

---

### 📋 **M5 – Workflow Agent (LangGraph)**

**Status:** Planned

**Goal:** Production-safe orchestration

**Deliverables:**

- Convert ReAct into LangGraph state machine
- Explicit state transitions
- Retry policies
- Step budgets
- Deterministic behavior

---

### 📋 **M6 – Memory System**

**Status:** Planned

**Deliverables:**

- Short-term: conversation memory
- Long-term: semantic memory (vector DB)
- Project-level: facts and preferences

---

### 📋 **M7 – Evaluation System**

**Status:** Planned

**Deliverables:**

- Golden RAG dataset (JSONL)
- Evaluation runner
- Accuracy proxy metrics
- Latency tracking
- Tool success rate

---

### 📋 **M8 – Observability & Production**

**Status:** Planned

**Deliverables:**

- OpenTelemetry tracing
- Request ID propagation
- Cost tracking per request
- Production Docker config
- Deployment documentation

---

## 🚀 Getting Started

### Prerequisites

- **Docker** and **Docker Compose**
- **Git**
- (Optional) **Python 3.12** for local development

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd AgenticCopilot
```

### 2. Configure environment

Create `.env` in project root:

```env
# Database
POSTGRES_DB=agentic_copilot
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432

# API
API_HOST=0.0.0.0
API_PORT=8000
DATABASE_URL=postgresql+psycopg://postgres:postgres@postgres:5432/agentic_copilot

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
```

### 3. Start services

```bash
docker compose up -d --build
```

Services:

- **API:** http://localhost:8000
- **Docs:** http://localhost:8000/docs (Swagger UI)
- **PostgreSQL:** localhost:5432
- **Redis:** localhost:6379

### 4. Verify health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected"
}
```

---

## 📂 Project Structure

```
agentic-copilot/
├── README.md
├── docker-compose.yml
├── .env
│
├── apps/
│   └── api/                        # FastAPI application
│       ├── alembic/                # Database migrations
│       │   ├── versions/
│       │   └── env.py
│       ├── app/
│       │   ├── main.py             # FastAPI app entry point
│       │   ├── settings.py         # Configuration
│       │   ├── db.py               # Database session
│       │   ├── base.py             # SQLAlchemy Base
│       │   ├── models/             # SQLAlchemy models
│       │   │   ├── workspace.py
│       │   │   ├── document.py
│       │   │   └── chunk.py
│       │   ├── routers/            # API endpoints
│       │   │   ├── health.py
│       │   │   ├── workspaces.py
│       │   │   └── documents.py
│       │   └── schemas/            # Pydantic request/response
│       │       ├── workspaces.py
│       │       └── documents.py
│       ├── core/                   # AI business logic (M1+)
│       │   └── chunking.py         # Text chunking
│       ├── tests/                  # Tests
│       │   └── test_chunking.py
│       ├── Dockerfile
│       ├── requirements.txt
│       └── pytest.ini
│
├── core/                           # Shared core logic (future)
└── tests/                          # Integration tests (future)
```

### Key Design Decisions

**Why `core/` inside `apps/api/`?**  
In M1, chunking is API-specific. Later, we'll extract common logic to root-level `core/` for reuse across API + workers.

**Why separate `models/` and `schemas/`?**

- `models/` = SQLAlchemy (database layer)
- `schemas/` = Pydantic (API layer)

Clean separation prevents tight coupling.

---

## ✨ Current Features (M1)

### 1. Workspace Management

```bash
# Create workspace
curl -X POST http://localhost:8000/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name": "AI Research"}'

# List workspaces
curl http://localhost:8000/workspaces

# Get workspace
curl http://localhost:8000/workspaces/1
```

### 2. Document Ingestion

```bash
# Upload document (auto-chunks content)
curl -X POST http://localhost:8000/workspaces/1/docs \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "guide.txt",
    "content": "Your document content here...",
    "mime_type": "text/plain"
  }'

# List documents
curl http://localhost:8000/workspaces/1/docs

# Get document with chunks
curl http://localhost:8000/workspaces/1/docs/1

# Get only chunks
curl http://localhost:8000/workspaces/1/docs/1/chunks
```

### 3. Automatic Chunking

Documents are automatically chunked with:

- **Chunk size:** 1000 characters
- **Overlap:** 200 characters
- **Metadata:** `chunk_index`, `start_char`, `end_char`

Prepares data for vector embeddings in M2.

---

## 🧪 Testing

### Run all tests

```bash
docker compose exec api pytest -v
```

### Run specific test file

```bash
docker compose exec api pytest tests/test_chunking.py -v
```

### Run with coverage

```bash
docker compose exec api pytest --cov=app --cov=core tests/
```

---

## 🔧 Development Workflow

### Making code changes

Code changes are automatically reloaded (uvicorn `--reload` enabled).

### Creating a migration

```bash
docker compose exec api alembic revision --autogenerate -m "description"
```

### Applying migrations

```bash
docker compose exec api alembic upgrade head
```

### Rolling back migration

```bash
docker compose exec api alembic downgrade -1
```

### Viewing migration status

```bash
docker compose exec api alembic current
docker compose exec api alembic history
```

### Database access

```bash
docker compose exec postgres psql -U postgres -d agentic_copilot
```

### Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f api
```

### Rebuilding

```bash
docker compose down
docker compose up -d --build
```

### Clean slate (⚠️ deletes data)

```bash
docker compose down -v
docker compose up -d --build
docker compose exec api alembic upgrade head
```

---

## 🎓 Learning Goals

This project teaches:

1. **RAG Architecture** - Beyond basic tutorials
2. **Agent Design Patterns** - ReAct, tool use, memory
3. **Vector Databases** - pgvector, similarity search
4. **Production Practices** - migrations, testing, observability
5. **Clean Architecture** - Separation of concerns, interfaces
6. **Incremental Development** - Ship working software at each milestone

---

## 📚 Resources

### Core Concepts

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [Pydantic V2](https://docs.pydantic.dev/latest/)

### AI/ML

- [pgvector](https://github.com/pgvector/pgvector)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangGraph](https://langchain-ai.github.io/langgraph/)

---

## 🤝 Contributing

This is a learning project. Key principles:

- **Incremental progress** - One milestone at a time
- **Production patterns** - Build it right, not just quick
- **Clean code** - Type hints, docstrings, tests
- **No shortcuts** - Understand before abstracting

---

## 📝 License

MIT License - feel free to use this for learning and building.

---

## 🚀 Next Steps

**Current Status:** M1 Complete ✅

**Next Milestone:** M2 - Embeddings + Vector Search

To start M2:

1. Enable pgvector extension in PostgreSQL
2. Add vector column to chunks table
3. Create embeddings client interface
4. Implement similarity search endpoint
5. Build retrieval evaluation

---

## 💬 Questions?

This is a learning project documenting the journey of building production AI infrastructure.

Key philosophy: **Understand the primitives before using frameworks.**

Happy building! 🚀
