"""
RAG Agent — answers questions using uploaded documents.
"""
from typing import Dict, Any, Generator
from app.agents.base import BaseAgent
from app.services.vector_store import vector_store
from app.services.github_models import github_models


class RAGAgent(BaseAgent):
    """Retrieves relevant docs via hybrid search and generates grounded answers."""
    
    def __init__(self):
        super().__init__(
            name="RAG Assistant",
            description="I answer questions using information from uploaded documents. "
                        "I provide accurate, context-based responses with source citations.",
            agent_type="rag"
        )
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search docs → build context → generate answer."""
        retrieved_docs = vector_store.hybrid_search(query)
        context_str = self._build_context(retrieved_docs)
        history = context.get("conversation_history", []) if context else []
        messages = self._build_messages(query, context_str, history)
        response = github_models.chat_completion(messages)
        
        return {
            "response": response,
            "sources": self._format_sources(retrieved_docs),
            "has_context": len(retrieved_docs) > 0,
            "agent_used": self.agent_type
        }
    
    def process_stream(self, query: str, context: Dict[str, Any] = None) -> Generator[str, None, None]:
        """Same as process() but streams the response chunk by chunk."""
        retrieved_docs = vector_store.hybrid_search(query)
        context_str = self._build_context(retrieved_docs)
        history = context.get("conversation_history", []) if context else []
        messages = self._build_messages(query, context_str, history)
        
        for chunk in github_models.chat_completion_stream(messages):
            yield chunk
    
    def _build_context(self, docs: list) -> str:
        """Turn retrieved chunks into a labeled context string."""
        if not docs:
            return ""
        
        parts = []
        for i, doc in enumerate(docs, 1):
            name = doc.get("metadata", {}).get("filename", f"Document {i}")
            score = doc.get("score", 0)
            parts.append(f"[Source {i}: {name} (relevance: {score:.2f})]\n{doc['content']}")
        
        return "\n\n---\n\n".join(parts)
    
    def _format_sources(self, docs: list) -> list:
        """Trim sources down for the frontend citation cards."""
        sources = []
        for i, doc in enumerate(docs, 1):
            content = doc["content"]
            sources.append({
                "index": i,
                "content": content[:200] + "..." if len(content) > 200 else content,
                "filename": doc.get("metadata", {}).get("filename", "Unknown"),
                "chunk_index": doc.get("metadata", {}).get("chunk_index", 0),
                "score": round(doc.get("score", 0), 3)
            })
        return sources
    
    def _build_messages(self, query: str, context: str, history: list = None) -> list:
        """Assemble the system prompt + history + user query for the LLM."""
        system_prompt = self.get_system_prompt()
        
        if context:
            system_prompt += (
                "\n\nUse the reference documents below to answer accurately. "
                "Cite sources as [Source N]. If the docs don't help, say so.\n\n"
                "REFERENCE DOCUMENTS:\n" + context
            )
        else:
            system_prompt += (
                "\n\nNo documents uploaded yet. Give a helpful general answer "
                "and suggest the user upload docs for better results."
            )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # keep last 6 messages for conversational context
        if history:
            messages.extend(history[-6:])
        
        messages.append({"role": "user", "content": query})
        return messages


# single shared instance
rag_agent = RAGAgent()
