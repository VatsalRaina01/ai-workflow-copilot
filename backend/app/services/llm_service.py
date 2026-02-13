from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class LLMService:
    """LangChain-integrated LLM service with conversation memory."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=settings.GITHUB_TOKEN,
            openai_api_base="https://models.inference.ai.azure.com",
            temperature=0.7
        )
        # Conversation memory (keep last 10 exchanges)
        self.memory = ConversationBufferWindowMemory(
            k=10,
            return_messages=True,
            memory_key="history"
        )

    async def generate_response(
        self, 
        prompt: str, 
        system_context: str = "You are a helpful AI assistant."
    ) -> str:
        """Generate response with conversation memory."""
        messages = [
            SystemMessage(content=system_context),
        ]
        
        # Add conversation history
        history = self.memory.load_memory_variables({})
        if history.get("history"):
            messages.extend(history["history"])
        
        messages.append(HumanMessage(content=prompt))
        
        response = await self.llm.ainvoke(messages)
        
        # Save to memory
        self.memory.save_context(
            {"input": prompt},
            {"output": response.content}
        )
        
        return response.content

    async def process_step(self, step_type: str, input_data: str) -> str:
        """Process a workflow step with type-specific handling."""
        step_handlers = {
            "generation": lambda: self.generate_response(input_data),
            "summary": lambda: self.generate_response(
                f"Summarize the following text:\n\n{input_data}",
                "You are a concise summarizer. Create bullet-point summaries."
            ),
            "analysis": lambda: self.generate_response(
                f"Analyze the following data:\n\n{input_data}",
                "You are a data analyst. Provide structured analysis."
            ),
            "code_review": lambda: self.generate_response(
                f"Review this code:\n\n{input_data}",
                "You are a senior code reviewer. Find bugs, suggest improvements."
            ),
        }
        
        handler = step_handlers.get(step_type, lambda: self.generate_response(input_data))
        return await handler()
    
    def clear_memory(self):
        """Clear conversation memory."""
        self.memory.clear()


llm_service = LLMService()
