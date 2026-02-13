"""
Code Analysis Agent - Analyzes code, explains errors, suggests improvements

Handles code-related queries with language-aware analysis.
"""
from typing import Dict, Any, Generator
from app.agents.base import BaseAgent
from app.services.github_models import github_models


class CodeAgent(BaseAgent):
    """Agent specialized in code analysis and assistance."""
    
    def __init__(self):
        super().__init__(
            name="Code Analyst",
            description="I analyze code snippets, explain errors, suggest improvements, "
                        "and help with programming questions across multiple languages.",
            agent_type="code"
        )
    
    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze code or answer programming questions."""
        messages = self._build_messages(query, context)
        response = github_models.chat_completion(messages, temperature=0.2, max_tokens=1500)
        
        return {
            "response": response,
            "agent_used": self.agent_type,
            "has_context": bool(context),
            "sources": []
        }
    
    def process_stream(self, query: str, context: Dict[str, Any] = None) -> Generator[str, None, None]:
        """Stream code analysis response."""
        messages = self._build_messages(query, context)
        for chunk in github_models.chat_completion_stream(messages, temperature=0.2, max_tokens=1500):
            yield chunk
    
    def _build_messages(self, query: str, context: Dict[str, Any] = None) -> list:
        """Build messages for code analysis."""
        system_prompt = self.get_system_prompt() + """

When analyzing code:
1. **Identify the language** and framework being used
2. **Explain what the code does** in plain English
3. **Point out issues** — bugs, anti-patterns, security concerns
4. **Suggest improvements** with corrected code examples
5. **Use markdown code blocks** with proper language tags for all code

When helping with errors:
- Explain WHY the error occurs, not just how to fix it
- Provide the corrected code with comments explaining changes
- Suggest preventive measures

Always use clear formatting with code blocks and explanations."""

        conversation_history = context.get("conversation_history", []) if context else []
        
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in conversation_history[-6:]:
            messages.append(msg)
        
        messages.append({"role": "user", "content": query})
        return messages


# Singleton instance
code_agent = CodeAgent()
