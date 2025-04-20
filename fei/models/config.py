"""
Configuration for model selection and switching.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
import os

from fei.utils.config import Config

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for model selection and switching."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the model configuration.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self._load_default_models()
    
    def _load_default_models(self) -> None:
        """Load default models if not already configured."""
        if "models" not in self.config.data:
            # Define default models
            default_models = {
                "anthropic": {
                    "claude-3-opus-20240229": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_generation", "code_understanding",
                            "reasoning", "planning", "tool_use",
                            "math", "science"
                        ],
                        "context_length": 200000,
                        "cost_per_input_token": 0.00015,
                        "cost_per_output_token": 0.00075,
                        "reliability": 0.98,
                        "metadata": {
                            "description": "Claude 3 Opus - Anthropic's most powerful model"
                        }
                    },
                    "claude-3-sonnet-20240229": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_generation", "code_understanding",
                            "reasoning", "planning", "tool_use"
                        ],
                        "context_length": 200000,
                        "cost_per_input_token": 0.00003,
                        "cost_per_output_token": 0.00015,
                        "reliability": 0.95,
                        "metadata": {
                            "description": "Claude 3 Sonnet - Balanced performance and cost"
                        }
                    },
                    "claude-3-haiku-20240307": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_understanding", "reasoning"
                        ],
                        "context_length": 200000,
                        "cost_per_input_token": 0.00000025,
                        "cost_per_output_token": 0.00000125,
                        "reliability": 0.92,
                        "metadata": {
                            "description": "Claude 3 Haiku - Fast and cost-effective"
                        }
                    }
                },
                "openai": {
                    "gpt-4o": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_generation", "code_understanding",
                            "reasoning", "planning", "tool_use",
                            "math", "science"
                        ],
                        "context_length": 128000,
                        "cost_per_input_token": 0.00005,
                        "cost_per_output_token": 0.00015,
                        "reliability": 0.97,
                        "metadata": {
                            "description": "GPT-4o - OpenAI's most capable model"
                        }
                    },
                    "gpt-4-turbo": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_generation", "code_understanding",
                            "reasoning", "planning", "tool_use"
                        ],
                        "context_length": 128000,
                        "cost_per_input_token": 0.00001,
                        "cost_per_output_token": 0.00003,
                        "reliability": 0.95,
                        "metadata": {
                            "description": "GPT-4 Turbo - Balanced performance and cost"
                        }
                    },
                    "gpt-3.5-turbo": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_understanding"
                        ],
                        "context_length": 16000,
                        "cost_per_input_token": 0.0000005,
                        "cost_per_output_token": 0.0000015,
                        "reliability": 0.9,
                        "metadata": {
                            "description": "GPT-3.5 Turbo - Fast and cost-effective"
                        }
                    }
                },
                "google": {
                    "gemini-1.5-pro": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_generation", "code_understanding",
                            "reasoning", "planning", "tool_use"
                        ],
                        "context_length": 1000000,
                        "cost_per_input_token": 0.00000375,
                        "cost_per_output_token": 0.00001125,
                        "reliability": 0.94,
                        "metadata": {
                            "description": "Gemini 1.5 Pro - Google's most capable model"
                        }
                    },
                    "gemini-1.0-pro": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_understanding", "reasoning"
                        ],
                        "context_length": 32000,
                        "cost_per_input_token": 0.0000025,
                        "cost_per_output_token": 0.0000075,
                        "reliability": 0.92,
                        "metadata": {
                            "description": "Gemini 1.0 Pro - Balanced performance and cost"
                        }
                    }
                },
                "groq": {
                    "llama3-70b-8192": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_generation", "code_understanding",
                            "reasoning"
                        ],
                        "context_length": 8192,
                        "cost_per_input_token": 0.0000007,
                        "cost_per_output_token": 0.0000009,
                        "reliability": 0.93,
                        "metadata": {
                            "description": "Llama 3 70B - Fast inference on Groq"
                        }
                    },
                    "mixtral-8x7b-32768": {
                        "capabilities": [
                            "text_generation", "chat", "function_calling",
                            "code_understanding", "reasoning"
                        ],
                        "context_length": 32768,
                        "cost_per_input_token": 0.0000006,
                        "cost_per_output_token": 0.0000006,
                        "reliability": 0.91,
                        "metadata": {
                            "description": "Mixtral 8x7B - Fast and cost-effective on Groq"
                        }
                    }
                }
            }
            
            # Save default models to config
            self.config.set("models", default_models)
            self.config.save()
            logger.info("Loaded default model configurations")
    
    def get_selector_config(self) -> Dict[str, Any]:
        """
        Get the model selector configuration.
        
        Returns:
            Model selector configuration
        """
        return self.config.get("model_selector", {})
    
    def set_selector_config(self, selector_config: Dict[str, Any]) -> None:
        """
        Set the model selector configuration.
        
        Args:
            selector_config: Model selector configuration
        """
        self.config.set("model_selector", selector_config)
        self.config.save()
    
    def get_fallback_config(self) -> Dict[str, Any]:
        """
        Get the fallback strategy configuration.
        
        Returns:
            Fallback strategy configuration
        """
        return self.config.get("fallback_strategy", {})
    
    def set_fallback_config(self, fallback_config: Dict[str, Any]) -> None:
        """
        Set the fallback strategy configuration.
        
        Args:
            fallback_config: Fallback strategy configuration
        """
        self.config.set("fallback_strategy", fallback_config)
        self.config.save()
    
    def get_default_model(self, provider: Optional[str] = None) -> Tuple[str, str]:
        """
        Get the default model.
        
        Args:
            provider: Optional provider name
            
        Returns:
            Tuple of (provider_name, model_id)
            
        Raises:
            ValueError: If no default model is found
        """
        # Check if a specific provider is requested
        if provider:
            provider_models = self.config.get("models", {}).get(provider, {})
            if not provider_models:
                raise ValueError(f"No models found for provider {provider}")
            
            # Get the default model for the provider
            default_model = self.config.get("default_models", {}).get(provider)
            if default_model:
                return provider, default_model
            
            # If no default model is specified, use the first model
            return provider, next(iter(provider_models.keys()))
        
        # Get the default provider
        default_provider = self.config.get("provider")
        if not default_provider:
            # If no default provider, use the first provider with models
            providers = self.config.get("models", {})
            if not providers:
                raise ValueError("No models found in configuration")
            
            default_provider = next(iter(providers.keys()))
        
        # Get the default model for the provider
        default_model = self.config.get("default_models", {}).get(default_provider)
        if default_model:
            return default_provider, default_model
        
        # If no default model is specified, use the first model for the provider
        provider_models = self.config.get("models", {}).get(default_provider, {})
        if not provider_models:
            raise ValueError(f"No models found for provider {default_provider}")
        
        return default_provider, next(iter(provider_models.keys()))
    
    def set_default_model(self, provider: str, model_id: str) -> None:
        """
        Set the default model for a provider.
        
        Args:
            provider: Provider name
            model_id: Model ID
            
        Raises:
            ValueError: If the model is not found
        """
        # Check if the model exists
        provider_models = self.config.get("models", {}).get(provider, {})
        if model_id not in provider_models:
            raise ValueError(f"Model {model_id} not found for provider {provider}")
        
        # Set the default model
        default_models = self.config.get("default_models", {})
        default_models[provider] = model_id
        self.config.set("default_models", default_models)
        
        # If this is the first default model, set the default provider
        if "provider" not in self.config.data:
            self.config.set("provider", provider)
        
        self.config.save()
    
    def set_default_provider(self, provider: str) -> None:
        """
        Set the default provider.
        
        Args:
            provider: Provider name
            
        Raises:
            ValueError: If the provider is not found
        """
        # Check if the provider exists
        if provider not in self.config.get("models", {}):
            raise ValueError(f"Provider {provider} not found")
        
        # Set the default provider
        self.config.set("provider", provider)
        self.config.save()
