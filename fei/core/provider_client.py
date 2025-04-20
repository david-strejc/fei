"""
Provider client for Fei code assistant.

This module contains the ProviderClient class, which handles
communication with LLM providers.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Tuple

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.core.mediator import Mediator

logger = get_logger(__name__)


class ProviderClient:
    """
    Handles communication with LLM providers.

    This class is responsible for:
    1. Managing provider configuration and API keys
    2. Sending messages to LLM providers
    3. Extracting responses and tool calls
    """

    def __init__(
        self,
        config: Config,
        mediator: Optional[Mediator] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize the provider client.

        Args:
            config: Configuration object
            mediator: Mediator for component communication
            provider: Provider to use (anthropic, openai, etc.)
            api_key: API key (overrides config)
            model: Model to use (overrides config)
        """
        self.config = config
        self.mediator = mediator

        # Provider-specific settings
        self.provider_key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY",
            "gemini": "GOOGLE_API_KEY"  # Gemini uses the Google API key
        }

        # Default models for each provider
        self.default_models = {
            "anthropic": "claude-3-7-sonnet-20250219",
            "openai": "gpt-4o",
            "groq": "llama3-70b-8192",
            "google": "gemini-1.5-pro-latest",
            "gemini": "gemini-2.5-flash-preview-04-17"  # Default Gemini model
        }

        # Set up provider
        self.provider = provider or self.config.get("provider", "anthropic")

        # Set up API key
        self.api_key = self._setup_api_key(api_key)

        # Set up model
        self.model = model or self._setup_model()

    def _setup_api_key(self, api_key: Optional[str] = None) -> str:
        """
        Set up API key with proper fallbacks.

        Args:
            api_key: API key (overrides config)

        Returns:
            API key to use
        """
        # Priority: explicit parameter > provider-specific config > generic config
        key_env_var = self.provider_key_map.get(self.provider, f"{self.provider.upper()}_API_KEY")

        return (
            api_key or
            self.config.get(f"{self.provider}.api_key") or
            self.config.get("api_key") or
            self.config.get(key_env_var)
        )

    def _setup_model(self) -> str:
        """
        Set up model with proper fallbacks.

        Returns:
            Model name to use
        """
        return (
            self.config.get(f"{self.provider}.model") or
            self.config.get("model", self.default_models.get(self.provider, self.default_models["anthropic"]))
        )

    async def send_message(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Send a message to the LLM provider.

        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            tools: Optional list of tools

        Returns:
            Tuple of (response content, tool calls)
        """
        try:
            # Import here to avoid circular imports
            from litellm import completion as litellm_completion

            # Handle gemini provider as an alias for google
            provider = self.provider
            if provider == "gemini":
                provider = "google"

            # Determine model identifier (avoid double prefix if model already includes provider)
            # For gemini models, we need special handling
            if self.provider == "gemini" or (provider == "google" and "gemini" in self.model):
                # Extract the model name without the gemini- prefix
                if self.model.startswith("gemini/"):
                    model_id = self.model
                elif self.model.startswith("gemini-"):
                    # For models like gemini-2.5-flash-preview-04-17, we need to keep the full name
                    model_id = self.model
                else:
                    model_id = f"gemini/{self.model}"
            else:
                model_id = self.model if '/' in self.model else f"{provider}/{self.model}"
            # Set up parameters
            params = {
                "model": model_id,
                "messages": messages,
                "api_key": self.api_key,
                "max_tokens": 4000
            }

            # Add system prompt if provided
            if system_prompt:
                # Insert system message at the beginning
                params["messages"] = [{"role": "system", "content": system_prompt}] + params["messages"]

            # Add tools if provided
            if tools:
                params["tools"] = tools

            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: litellm_completion(**params)
            )

            # Extract tool calls and content
            tool_calls = self._extract_tool_calls(response)
            content = self._extract_response_content(response)

            return content, tool_calls

        except Exception as e:
            logger.error(f"Error in send_message: {e}", exc_info=True)
            error_msg = f"I encountered an error while processing your request: {str(e)}"
            return error_msg, []

    async def send_continuation(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Send a continuation message to get final response after tool execution.

        Args:
            messages: List of conversation messages
            tools: Optional list of tools

        Returns:
            Response content
        """
        try:
            # Import here to avoid circular imports
            from litellm import completion as litellm_completion

            # Handle gemini provider as an alias for google
            provider = self.provider
            if provider == "gemini":
                provider = "google"

            # Determine model identifier for continuation
            # For gemini models, we need special handling
            if self.provider == "gemini" or (provider == "google" and "gemini" in self.model):
                # Extract the model name without the gemini- prefix
                if self.model.startswith("gemini/"):
                    cont_model_id = self.model
                elif self.model.startswith("gemini-"):
                    # For models like gemini-2.5-flash-preview-04-17, we need to keep the full name
                    cont_model_id = self.model
                else:
                    cont_model_id = f"gemini/{self.model}"
            else:
                cont_model_id = self.model if '/' in self.model else f"{provider}/{self.model}"
            # Set up parameters
            continue_params = {
                "model": cont_model_id,
                "max_tokens": 4000,
                "messages": messages,
                "api_key": self.api_key
            }

            # Make sure to include tools in the continuation
            if tools:
                continue_params["tools"] = tools

            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            continue_response = await loop.run_in_executor(
                None,
                lambda: litellm_completion(**continue_params)
            )

            # Extract text response safely
            answer = self._extract_response_content(continue_response)

            return answer

        except Exception as e:
            logger.error(f"Error in continuation: {e}", exc_info=True)
            error_msg = f"I tried to use tools to answer your question, but encountered an error: {str(e)}"
            return error_msg

    def _extract_tool_calls(self, response: Any) -> List[Dict[str, Any]]:
        """
        Extract tool calls from a response.

        Args:
            response: Response from the LLM provider

        Returns:
            List of tool calls
        """
        try:
            # Check if response has tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Extract tool calls
                return [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": tc.function.arguments
                    }
                    for tc in response.tool_calls
                ]

            # Check for Anthropic-style tool calls
            if hasattr(response, 'content') and isinstance(response.content, list):
                tool_calls = []

                for content_block in response.content:
                    if content_block.type == 'tool_use':
                        tool_calls.append({
                            "id": content_block.id,
                            "name": content_block.name,
                            "input": content_block.input
                        })

                return tool_calls

            return []
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}", exc_info=True)
            return []

    def _extract_response_content(self, response: Any) -> str:
        """
        Extract content from a response.

        Args:
            response: Response from the LLM provider

        Returns:
            Response content
        """
        try:
            # Handle different response formats
            if hasattr(response, 'choices') and response.choices:
                # OpenAI format
                message = response.choices[0].message

                if hasattr(message, 'content') and message.content is not None:
                    return message.content

                # Handle content list (Anthropic via LiteLLM)
                if hasattr(message, 'content') and isinstance(message.content, list):
                    text_parts = []
                    for content_block in message.content:
                        if content_block.get('type') == 'text':
                            text_parts.append(content_block.get('text', ''))
                    return ''.join(text_parts)

            # Direct Anthropic format
            if hasattr(response, 'content') and isinstance(response.content, list):
                text_parts = []
                for content_block in response.content:
                    if content_block.type == 'text':
                        text_parts.append(content_block.text)
                return ''.join(text_parts)

            # Simple content attribute
            if hasattr(response, 'content') and isinstance(response.content, str):
                return response.content

            # Fallback to string representation
            return str(response)
        except Exception as e:
            logger.error(f"Error extracting response content: {e}", exc_info=True)
            return f"Error extracting response: {str(e)}"
