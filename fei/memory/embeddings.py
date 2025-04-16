"""
Embedding providers for Fei memory system.

This module provides embedding providers for generating vector
embeddings of text for semantic search.
"""

import asyncio
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

from fei.utils.logging import get_logger
from fei.utils.config import Config

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""
    
    @abstractmethod
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        pass


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using LiteLLM."""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LiteLLM embedding provider.
        
        Args:
            config: Configuration object
            provider: Provider to use (openai, cohere, etc.)
            model: Model to use
            api_key: API key (overrides config)
        """
        self.config = config or Config()
        
        # Set up provider
        self.provider = provider or self.config.get("embedding_provider", "openai")
        
        # Set up model
        self.model = model or self._get_default_model()
        
        # Set up API key
        self.api_key = api_key or self._get_api_key()
        
        # Set up dimension based on model
        self._dimension = self._get_dimension()
    
    def _get_default_model(self) -> str:
        """
        Get the default model for the provider.
        
        Returns:
            Default model name
        """
        # Provider-specific default models
        defaults = {
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
            "google": "textembedding-gecko"
        }
        
        return (
            self.config.get(f"{self.provider}.embedding_model") or
            self.config.get("embedding_model") or
            defaults.get(self.provider, "text-embedding-3-small")
        )
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get the API key for the provider.
        
        Returns:
            API key or None
        """
        # Provider-specific API key environment variables
        key_map = {
            "openai": "OPENAI_API_KEY",
            "cohere": "COHERE_API_KEY",
            "google": "GOOGLE_API_KEY"
        }
        
        key_env_var = key_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
        
        return (
            self.config.get(f"{self.provider}.api_key") or
            self.config.get("api_key") or
            self.config.get(key_env_var)
        )
    
    def _get_dimension(self) -> int:
        """
        Get the dimension for the current model.
        
        Returns:
            Embedding dimension
        """
        # Model-specific dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "embed-english-v3.0": 1024,
            "textembedding-gecko": 768
        }
        
        return dimensions.get(self.model, 1536)
    
    @property
    def dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension
        """
        return self._dimension
    
    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Import here to avoid circular imports
            from litellm import embedding as litellm_embedding
            
            # Set up parameters
            params = {
                "model": f"{self.provider}/{self.model}",
                "input": text,
                "api_key": self.api_key
            }
            
            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: litellm_embedding(**params)
            )
            
            # Extract embedding
            embedding_data = response.data[0].embedding
            
            return np.array(embedding_data)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Import here to avoid circular imports
            from litellm import embedding as litellm_embedding
            
            # Set up parameters
            params = {
                "model": f"{self.provider}/{self.model}",
                "input": texts,
                "api_key": self.api_key
            }
            
            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: litellm_embedding(**params)
            )
            
            # Extract embeddings
            embeddings = [np.array(item.embedding) for item in response.data]
            
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise


class EmbeddingProviderFactory:
    """Factory for creating embedding provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, provider_name: str, provider_class: type) -> None:
        """
        Register an embedding provider class.
        
        Args:
            provider_name: Name of the provider
            provider_class: Provider class
        """
        cls._providers[provider_name] = provider_class
    
    @classmethod
    def create_provider(
        cls,
        provider_name: str,
        config: Optional[Config] = None,
        **kwargs
    ) -> EmbeddingProvider:
        """
        Create an embedding provider instance.
        
        Args:
            provider_name: Name of the provider
            config: Optional configuration
            **kwargs: Additional arguments for the provider
            
        Returns:
            Embedding provider instance
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider_name not in cls._providers:
            raise ValueError(f"Embedding provider '{provider_name}' not registered")
        
        return cls._providers[provider_name](config=config, **kwargs)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered embedding providers.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())


# Register providers
EmbeddingProviderFactory.register_provider("litellm", LiteLLMEmbeddingProvider)
