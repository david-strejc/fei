"""
Provider manager for Fei.
"""

from typing import Dict, Optional, Any

from fei.providers.base import LLMProviderInterface, ProviderFactory
from fei.utils.config import Config


class ProviderManager:
    """Manager for LLM providers."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the provider manager.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.current_provider = None
    
    def get_provider(self, provider_name: Optional[str] = None) -> LLMProviderInterface:
        """
        Get a provider instance.
        
        Args:
            provider_name: Optional provider name, defaults to config value
            
        Returns:
            Provider instance
        """
        provider_name = provider_name or self.config.get("provider", "litellm")
        
        if self.current_provider is None or self.current_provider.__class__.__name__ != f"{provider_name.capitalize()}Provider":
            self.current_provider = ProviderFactory.create_provider(provider_name)
            self.current_provider.setup()
        
        return self.current_provider
    
    def list_providers(self) -> Dict[str, str]:
        """
        List all available providers with their default models.
        
        Returns:
            Dictionary of provider names and default models
        """
        result = {}
        for provider_name in ProviderFactory.list_providers():
            provider = ProviderFactory.create_provider(provider_name)
            result[provider_name] = provider.get_default_model()
        
        return result
