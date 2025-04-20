"""
LiteLLM provider implementation.
"""

import os
from typing import Dict, List, Optional, Any, Union
import litellm

from fei.providers.base import LLMProviderInterface, ProviderFactory
from fei.utils.config import Config


class LiteLLMProvider(LLMProviderInterface):
    """LiteLLM provider implementation."""

    def __init__(self):
        """Initialize the LiteLLM provider."""
        self.config = Config()
        self.provider = None
        self.api_key = None
        self.model = None
        self.provider_key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        self.default_models = {
            "anthropic": "claude-3-opus-20240229",
            "openai": "gpt-4-turbo",
            "groq": "llama3-70b-8192",
            "google": "gemini-1.5-pro",
        }

    def setup(self, api_key: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None) -> None:
        """
        Set up the provider with API key and model.

        Args:
            api_key: Optional API key for the provider
            model: Optional model name to use
            provider: Optional provider name to use
        """
        # Set provider from arguments, environment, or default
        if provider:
            self.provider = provider
        elif self.provider is None:
            # Try to get from environment variable first
            env_provider = os.environ.get("PROVIDER")
            if env_provider:
                self.provider = env_provider.lower()
            else:
                # Fall back to config or default
                self.provider = self.config.get("provider", "anthropic")

        # Ensure provider is valid
        if self.provider not in self.provider_key_map and self.provider != "gemini":
            # Handle gemini as an alias for google
            if self.provider == "gemini":
                self.provider = "google"
            else:
                # Fall back to default provider
                self.provider = "anthropic"

        self.api_key = self._setup_api_key(api_key)
        self.model = self._setup_model(model)

    def _setup_api_key(self, api_key: Optional[str] = None) -> str:
        """
        Set up the API key for the provider.

        Args:
            api_key: Optional API key for the provider

        Returns:
            The API key

        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        if api_key:
            return api_key

        # Try to get from config using the provider-specific key format
        config_key_name = f"{self.provider}.api_key"
        config_key = self.config.get(config_key_name)
        if config_key:
            return config_key

        # Try to get from environment
        env_var = self.provider_key_map.get(self.provider)
        if env_var and env_var in os.environ:
            return os.environ[env_var]

        raise ValueError(f"API key for provider '{self.provider}' not found")

    def _setup_model(self, model: Optional[str] = None) -> str:
        """
        Set up the model for the provider.

        Args:
            model: Optional model name to use

        Returns:
            The model name
        """
        # Priority: explicit parameter > environment variable > config > default

        # 1. Check explicit parameter
        if model:
            return model

        # 2. Check environment variable
        env_model = os.environ.get("MODEL")
        if env_model:
            return env_model

        # 3. Try to get from config
        config_model = self.config.get("model")
        if config_model:
            return config_model

        # 4. Use provider-specific config
        provider_model_key = f"{self.provider}.model"
        provider_model = self.config.get(provider_model_key)
        if provider_model:
            return provider_model

        # 5. Use default model for provider
        return self.get_default_model()

    def get_default_model(self) -> str:
        """
        Get the default model for this provider.

        Returns:
            The default model name
        """
        return self.default_models.get(self.provider, "claude-3-opus-20240229")

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
        # Handle gemini provider as an alias for google
        provider = self.provider
        if provider == "gemini":
            provider = "google"

        # Format the model name correctly for LiteLLM
        model = f"{provider}/{self.model}"

        # Set API key in environment
        env_var = self.provider_key_map.get(provider, "ANTHROPIC_API_KEY")
        os.environ[env_var] = self.api_key

        # Special handling for Google/Gemini models
        if provider == "google" and "GEMINI_API_KEY" not in os.environ:
            os.environ["GEMINI_API_KEY"] = self.api_key

        # Call LiteLLM
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            stream=stream
        )

        return response

    def extract_response_content(self, response: Dict[str, Any]) -> str:
        """
        Extract the text content from a response.

        Args:
            response: Response from the LLM provider

        Returns:
            The text content of the response
        """
        if not response or "choices" not in response:
            return ""

        choice = response["choices"][0]
        if "message" not in choice:
            return ""

        message = choice["message"]
        return message.get("content", "")

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a response.

        Args:
            response: Response from the LLM provider

        Returns:
            List of tool calls
        """
        if not response or "choices" not in response:
            return []

        choice = response["choices"][0]
        if "message" not in choice:
            return []

        message = choice["message"]
        if "tool_calls" not in message:
            return []

        return message["tool_calls"]


# Register the provider
ProviderFactory.register_provider("litellm", LiteLLMProvider)
