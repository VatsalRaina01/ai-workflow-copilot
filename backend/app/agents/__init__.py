"""
Agent Module - Modular agent system with polymorphic dispatch

Available agents:
- RAGAgent: Document Q&A with hybrid search
- SummarizeAgent: Text summarization
- CodeAgent: Code analysis and debugging
"""
from app.agents.rag_agent import rag_agent
from app.agents.summarize_agent import summarize_agent
from app.agents.code_agent import code_agent
from app.agents.router import agent_router
