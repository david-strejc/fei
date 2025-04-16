"""
Base provider interface for Fei.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union


class LLMProviderInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def setup(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Set up the provider with API key and model.
        
        Args:
            api_key: Optional API key for the provider
            model: Optional model name to use
        """
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        """
        Get the default model for this provider.
        
        Returns:
            The default model name
        """
        pass
    
    @abstractmethod
    def complete(self, 
                messages: List[Dict[str, str]], 
                tools: Optional[List[Dict[str, Any]]] = None,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                stream: bool = False) -> Dict[str, Any]:
        """
        Generate a completion from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional list of tools to make available to the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            
        Returns:
            Response from the LLM provider
        """
        pass
    
    @abstractmethod
    def extract_response_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the text content from a response.
        
        Args:
            response: Response from the LLM provider
            
        Returns:
            The text content of the response
        """
        pass
    
    @abstractmethod
    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a response.
        
        Args:
            response: Response from the LLM provider
            
        Returns:
            List of tool calls
        """
        pass


class ProviderFactory:
    """Factory for creating provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type) -> None:
        """
        Register a provider class.
        
        Args:
            provider_name: Name of the provider
            provider_class: Provider class
        """
        cls._providers[provider_name] = provider_class
    
    @classmethod
    def create_provider(cls, provider_name: str) -> LLMProviderInterface:
        """
        Create a provider instance.
        
        Args:
            provider_name: Name of the provider
            
        Returns:
            Provider instance
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider_name not in cls._providers:
            raise ValueError(f"Provider '{provider_name}' not registered")
        
        return cls._providers[provider_name]()
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered providers.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())
