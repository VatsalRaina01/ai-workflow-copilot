"""
Summarization Agent - Generates concise summaries

Handles document summarization, text condensation, and key point extraction.
Uses the vector store to retrieve uploaded document content for summarization.
"""
from typing import Dict, Any, Generator
from app.agents.base import BaseAgent
from app.services.vector_store import vector_store
from app.services.github_models import github_models


class SummarizeAgent(BaseAgent):
    """Agent specialized in text summarization."""
    
    def __init__(self):
        super().__init__(
            name="Summarization Expert",
            description="I create concise, structured summaries of documents and text. "
                        "I extract key points, main ideas, and organize information clearly.",
            agent_type="summarize"
        )
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Summarize the given text or document content."""
        doc_content = self._retrieve_document_content(query)
        messages = self._build_messages(query, context, doc_content)
        response = github_models.chat_completion(messages, temperature=0.3, max_tokens=2000)
        
        return {
            "response": response,
            "agent_used": self.agent_type,
            "has_context": bool(doc_content),
            "sources": []
        }
    
    def process_stream(self, query: str, context: Dict[str, Any] = None) -> Generator[str, None, None]:
        """Stream summarization response."""
        doc_content = self._retrieve_document_content(query)
        messages = self._build_messages(query, context, doc_content)
        for chunk in github_models.chat_completion_stream(messages, temperature=0.3, max_tokens=2000):
            yield chunk
    
    def _retrieve_document_content(self, query: str) -> str:
        """Retrieve uploaded document content from the vector store."""
        try:
            # Get all stored documents for summarization (use a broad query)
            docs = vector_store.hybrid_search(query, top_k=10)
            if not docs:
                return ""
            
            parts = []
            for i, doc in enumerate(docs, 1):
                name = doc.get("metadata", {}).get("filename", f"Document {i}")
                parts.append(f"[Source {i}: {name}]\n{doc['content']}")
            
            return "\n\n---\n\n".join(parts)
        except Exception:
            return ""
    
    def _build_messages(self, query: str, context: Dict[str, Any] = None, doc_content: str = "") -> list:
        """Build messages for summarization."""
        system_prompt = self.get_system_prompt() + """

When summarizing, follow this structure:
1. **Overview**: A 1-2 sentence high-level summary
2. **Key Points**: Bullet-point list of the most important information
3. **Details**: Any critical details, numbers, or specifics worth noting
4. **Conclusion**: A brief takeaway or implication

Keep summaries concise but comprehensive. Use markdown formatting."""

        if doc_content:
            system_prompt += (
                "\n\nThe user has uploaded documents. Use the content below to generate your summary.\n\n"
                "DOCUMENT CONTENT:\n" + doc_content
            )
        else:
            system_prompt += (
                "\n\nNo documents have been uploaded yet. If the user asks to summarize "
                "without providing text, let them know they can upload a document first."
            )

        conversation_history = context.get("conversation_history", []) if context else []
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        for msg in conversation_history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": query})
        return messages


# Singleton instance
summarize_agent = SummarizeAgent()
