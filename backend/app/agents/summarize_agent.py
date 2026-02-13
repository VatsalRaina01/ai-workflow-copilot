"""
Summarization Agent - Generates concise summaries

Handles document summarization, text condensation, and key point extraction.
"""
from typing import Dict, Any, Generator
from app.agents.base import BaseAgent
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
        messages = self._build_messages(query, context)
        response = github_models.chat_completion(messages, temperature=0.3)
        
        return {
            "response": response,
            "agent_used": self.agent_type,
            "has_context": bool(context),
            "sources": []
        }
    
    def process_stream(self, query: str, context: Dict[str, Any] = None) -> Generator[str, None, None]:
        """Stream summarization response."""
        messages = self._build_messages(query, context)
        for chunk in github_models.chat_completion_stream(messages, temperature=0.3):
            yield chunk
    
    def _build_messages(self, query: str, context: Dict[str, Any] = None) -> list:
        """Build messages for summarization."""
        system_prompt = self.get_system_prompt() + """

When summarizing, follow this structure:
1. **Overview**: A 1-2 sentence high-level summary
2. **Key Points**: Bullet-point list of the most important information
3. **Details**: Any critical details, numbers, or specifics worth noting
4. **Conclusion**: A brief takeaway or implication

Keep summaries concise but comprehensive. Use markdown formatting."""

        conversation_history = context.get("conversation_history", []) if context else []
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history for context
        for msg in conversation_history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": query})
        return messages


# Singleton instance
summarize_agent = SummarizeAgent()
