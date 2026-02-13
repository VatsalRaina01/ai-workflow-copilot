"""
Chat API Endpoints

Provides chat, document management, evaluation, and session management endpoints.
Includes guardrails middleware for security.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from app.agents.router import agent_router
from app.services.vector_store import vector_store
from app.services.document_parser import document_parser
from app.services.guardrails import guardrails
from app.services.evaluation import evaluation_service
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── Request/Response Models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    has_context: bool
    sources: Optional[List[dict]] = None
    agent_used: Optional[str] = None
    intent: Optional[str] = None

class DocumentResponse(BaseModel):
    doc_id: str
    message: str
    chunks_created: int
    file_type: Optional[str] = None

class EvaluateRequest(BaseModel):
    question: str
    answer: str
    sources: Optional[List[dict]] = None


# ─── Chat Endpoints ───────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the AI assistant.
    
    The assistant uses multi-agent routing with intent detection:
    - RAG agent for document-based questions
    - Summarization agent for condensing text
    - Code agent for programming help
    
    Includes guardrails for prompt injection protection.
    """
    # Run guardrails
    is_valid, error_msg = guardrails.validate_request(request.message)
    if not is_valid:
        return ChatResponse(
            response=error_msg,
            has_context=False,
            agent_used="guardrails"
        )
    
    if request.stream:
        return await chat_stream(request)
    
    result = await agent_router.route(request.message)
    
    return ChatResponse(
        response=result["response"],
        has_context=result.get("has_context", False),
        sources=result.get("sources", []),
        agent_used=result.get("agent_used"),
        intent=result.get("intent")
    )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat response using Server-Sent Events with agent metadata."""
    # Run guardrails
    is_valid, error_msg = guardrails.validate_request(request.message)
    if not is_valid:
        def error_gen():
            yield f"data: {json.dumps({'chunk': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(error_gen(), media_type="text/event-stream")
    
    def generate():
        for chunk in agent_router.route_stream(request.message):
            # Check if it's metadata
            if chunk.startswith("__META__") and chunk.endswith("__META__"):
                meta_json = chunk.replace("__META__", "")
                yield f"data: {json.dumps({'metadata': json.loads(meta_json)})}\n\n"
            else:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


# ─── Document Endpoints ───────────────────────────────────────────────

@router.post("/documents", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for RAG indexing.
    
    Supports: .txt, .md, .pdf, .docx
    """
    # Validate file type
    if not document_parser.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(sorted(document_parser.SUPPORTED_EXTENSIONS))}"
        )
    
    # Read and parse file
    content_bytes = await file.read()
    
    try:
        text_content = document_parser.parse(content_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    if not text_content.strip():
        raise HTTPException(status_code=400, detail="Document appears to be empty")
    
    # Get file extension
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    
    # Add to vector store
    doc_id = vector_store.add_document(
        content=text_content,
        metadata={"filename": file.filename, "file_type": file_ext}
    )
    
    logger.info(f"Document uploaded: {file.filename} ({file_ext}), doc_id: {doc_id}")
    
    return DocumentResponse(
        doc_id=doc_id,
        message=f"Document '{file.filename}' uploaded and indexed successfully",
        chunks_created=vector_store.get_document_count(),
        file_type=file_ext
    )

@router.post("/documents/text", response_model=DocumentResponse)
async def add_text_document(content: str, title: str = "Untitled"):
    """Add text content directly as a document."""
    doc_id = vector_store.add_document(
        content=content,
        metadata={"title": title, "filename": title, "file_type": ".txt"}
    )
    
    return DocumentResponse(
        doc_id=doc_id,
        message=f"Text document '{title}' indexed successfully",
        chunks_created=vector_store.get_document_count(),
        file_type=".txt"
    )

@router.get("/documents/count")
async def get_document_count():
    """Get the number of document chunks in the vector store."""
    return {"count": vector_store.get_document_count()}

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document by ID."""
    success = vector_store.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {doc_id} deleted successfully"}


# ─── Evaluation Endpoints ─────────────────────────────────────────────

@router.post("/evaluate")
async def evaluate_response(request: EvaluateRequest):
    """
    Evaluate a RAG response for quality metrics.
    
    Returns faithfulness, relevance, and answer quality scores.
    """
    result = evaluation_service.evaluate(
        question=request.question,
        answer=request.answer,
        sources=request.sources or []
    )
    return result


# ─── Session & Agent Endpoints ─────────────────────────────────────────

@router.post("/clear")
async def clear_conversation():
    """Clear the conversation history."""
    agent_router.clear_history()
    return {"message": "Conversation history cleared"}

@router.get("/agents")
async def get_agents():
    """Get information about available agents."""
    return {"agents": agent_router.get_available_agents()}
