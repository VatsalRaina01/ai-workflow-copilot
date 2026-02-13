"""
FastAPI Application - AI Workflow Copilot

Main application entry point with structured logging,
middleware setup, and lifecycle management.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.endpoints import chat
import logging
import sys

# ─── Structured Logging Setup ─────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("ai_workflow_copilot")

# ─── FastAPI App ──────────────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Workflow Copilot - A retrieval-augmented chatbot with "
                "multi-agent orchestration, hybrid search, and guardrails.",
    version="3.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs"
)

# ─── CORS Middleware ──────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────

app.include_router(chat.router, prefix=f"{settings.API_V1_STR}", tags=["chat"])

# ─── Lifecycle Events ─────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info(f"  {settings.PROJECT_NAME} v3.0.0")
    logger.info(f"  Chat Model:      {settings.CHAT_MODEL}")
    logger.info(f"  Embedding Model: {settings.EMBEDDING_MODEL}")
    logger.info(f"  Guardrails:      {'Enabled' if settings.GUARDRAILS_ENABLED else 'Disabled'}")
    logger.info(f"  Hybrid Search:   BM25 weight={settings.BM25_WEIGHT}")
    logger.info(f"  Docs:            {settings.API_V1_STR}/docs")
    logger.info("=" * 60)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AI Workflow Copilot...")

# ─── Root Endpoints ───────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "AI Workflow Copilot API is running",
        "version": "3.0.0",
        "docs": f"{settings.API_V1_STR}/docs",
        "features": [
            "Multi-agent routing (RAG, Summarize, Code)",
            "Hybrid search (Semantic + BM25)",
            "PDF & DOCX support",
            "Prompt injection guardrails",
            "Streaming responses",
            "RAG evaluation metrics"
        ]
    }

@app.get(f"{settings.API_V1_STR}/health")
def health_check():
    from app.services.vector_store import vector_store
    return {
        "status": "healthy",
        "chat_model": settings.CHAT_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "indexed_chunks": vector_store.get_document_count(),
        "guardrails_enabled": settings.GUARDRAILS_ENABLED,
        "hybrid_search": True
    }
