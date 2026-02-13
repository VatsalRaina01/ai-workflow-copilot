"""
Agent Router - Orchestrates agent selection and prompt chaining

Routes queries to appropriate agents using LLM-based intent detection
and manages multi-step workflows with conversation memory.
"""
from typing import Dict, Any, List, Generator
from app.agents.rag_agent import rag_agent
from app.agents.summarize_agent import summarize_agent
from app.agents.code_agent import code_agent
from app.services.github_models import github_models
import json
import logging

logger = logging.getLogger(__name__)


class AgentRouter:
    """
    Routes queries to appropriate agents using intent classification.
    
    Workflow:
    1. Classify query intent via LLM
    2. Route to appropriate agent (RAG, Summarize, Code, or General)
    3. Inject conversation history for contextual continuity
    4. Chain responses if needed for multi-step tasks
    """
    
    INTENT_SYSTEM_PROMPT = """You are an intent classifier. Classify the user's message into exactly one category.

Categories:
- "rag": Questions about uploaded documents, seeking information from files, document-related queries
- "summarize": Requests to summarize text, condense information, create overviews, generate bullet points
- "code": Code-related questions, debugging, code review, programming help, error analysis
- "general": General conversation, greetings, chitchat, questions not related to documents or code

Respond with ONLY a JSON object: {"intent": "<category>", "confidence": <0.0-1.0>}
Do not include any other text."""
    
    def __init__(self):
        self.agents = {
            "rag": rag_agent,
            "summarize": summarize_agent,
            "code": code_agent,
        }
        self.conversation_history: List[Dict[str, str]] = []
    
    def _classify_intent(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to classify the intent of a query.
        
        Returns:
            Dict with 'intent' (str) and 'confidence' (float)
        """
        try:
            messages = [
                {"role": "system", "content": self.INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]
            response = github_models.chat_completion(messages, temperature=0.0, max_tokens=50)
            result = json.loads(response.strip())
            return {
                "intent": result.get("intent", "rag"),
                "confidence": result.get("confidence", 0.5)
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Intent classification failed: {e}, defaulting to 'rag'")
            return {"intent": "rag", "confidence": 0.5}
    
    async def route(self, query: str) -> Dict[str, Any]:
        """
        Route a query to the appropriate agent based on intent.
        """
        # Classify intent
        intent_result = self._classify_intent(query)
        intent = intent_result["intent"]
        confidence = intent_result["confidence"]
        
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Build context with conversation history
        context = {"conversation_history": self.conversation_history.copy()}
        
        # Route to the appropriate agent
        agent = self.agents.get(intent, rag_agent)
        
        # For general queries without documents, use RAG agent (it handles no-context gracefully)
        if intent == "general":
            agent = rag_agent
        
        result = await agent.process(query, context)
        
        # Enrich result with routing metadata
        result["intent"] = intent
        result["intent_confidence"] = confidence
        if "agent_used" not in result:
            result["agent_used"] = agent.agent_type
        
        # Add response to history
        self.conversation_history.append({
            "role": "assistant", 
            "content": result["response"]
        })
        
        # Keep history manageable (last 20 messages)
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
        
        return result
    
    def route_stream(self, query: str) -> Generator[str, None, None]:
        """
        Route a query and stream the response.
        
        Yields:
            - First yields agent metadata as JSON
            - Then yields response chunks
        """
        # Classify intent
        intent_result = self._classify_intent(query)
        intent = intent_result["intent"]
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Build context
        context = {"conversation_history": self.conversation_history.copy()}
        
        # Select agent
        agent = self.agents.get(intent, rag_agent)
        if intent == "general":
            agent = rag_agent
        
        # Yield metadata first
        metadata = json.dumps({
            "type": "metadata",
            "agent_used": agent.agent_type,
            "agent_name": agent.name,
            "intent": intent,
            "intent_confidence": intent_result["confidence"]
        })
        yield f"__META__{metadata}__META__"
        
        # Stream from selected agent
        full_response = ""
        for chunk in agent.process_stream(query, context):
            full_response += chunk
            yield chunk
        
        # Add complete response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": full_response
        })
        
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    async def chain_prompts(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Execute a chain of prompts sequentially.
        Each prompt can use context from previous steps.
        """
        results = []
        context = {}
        
        for i, prompt in enumerate(prompts):
            enriched_prompt = prompt
            if context:
                enriched_prompt = f"Previous context: {context.get('last_response', '')}\n\nCurrent task: {prompt}"
            
            result = await self.route(enriched_prompt)
            results.append(result)
            
            context["last_response"] = result["response"]
            context["step"] = i + 1
        
        return results
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_available_agents(self) -> List[Dict[str, str]]:
        """Return info about all available agents."""
        return [agent.get_info() for agent in self.agents.values()]


# Singleton instance
agent_router = AgentRouter()
