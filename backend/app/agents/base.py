"""
Base Agent - Abstract class for all agents

Defines the interface for modular agent system with polymorphic dispatch.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Generator


class BaseAgent(ABC):
    """Base class for all agents in the system.
    
    Uses OOP principles:
    - Abstraction: Abstract interface for all agents
    - Polymorphism: Each agent implements process() differently
    - Encapsulation: Agent-specific logic is self-contained
    """
    
    def __init__(self, name: str, description: str, agent_type: str = "general"):
        self.name = name
        self.description = description
        self.agent_type = agent_type
    
    @abstractmethod
    async def process(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a query and return a response.
        
        Args:
            query: The user's query
            context: Optional context from previous steps
            
        Returns:
            Dict containing the response, metadata, and agent_used
        """
        pass
    
    @abstractmethod
    def process_stream(self, query: str, context: Dict[str, Any] = None) -> Generator[str, None, None]:
        """
        Process a query with streaming response.
        
        Args:
            query: The user's query
            context: Optional context
            
        Yields:
            Response chunks as strings
        """
        pass
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        return f"You are {self.name}. {self.description}"
    
    def get_info(self) -> Dict[str, str]:
        """Return agent metadata for UI display."""
        return {
            "name": self.name,
            "type": self.agent_type,
            "description": self.description
        }
