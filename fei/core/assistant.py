#!/usr/bin/env python3
"""
Main assistant class for Fei code assistant.

This module contains the Assistant class, which is the main entry point
for interacting with the Fei code assistant.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Awaitable, Tuple

from litellm import completion as litellm_completion
from litellm.exceptions import ServiceUnavailableError, APIError, InvalidRequestError, RateLimitError

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.core.orchestrator import AssistantOrchestrator
from fei.core.provider_client import ProviderClient
from fei.core.tool_executor import ToolExecutor
from fei.core.conversation_manager import ConversationManager
from fei.core.model_optimizer import ModelOptimizer
from fei.core.mcp import MCPManager
from fei.core.mediator import Mediator
from fei.tools.unified_registry import UnifiedToolRegistry

logger = get_logger(__name__)

# Type alias for the UI function that asks the user for consent
# It should take a prompt string and return True (approved) or False (denied)
ConsentUiProvider = Callable[[str], Awaitable[bool]]

class ProviderManager:
    """Manages provider configuration and API key handling"""

    def __init__(
        self,
        config: Config,
        provider: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize provider manager

        Args:
            config: Configuration object
            provider: Provider to use (anthropic, openai, etc.)
            api_key: API key (overrides config)
        """
        self.config = config

        # Set up provider
        self.provider = provider or self.config.get("provider", "anthropic")

        # Provider-specific settings
        self.provider_key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "google": "GOOGLE_API_KEY" # Added Google
        }

        # Default model mapping by provider
        self.default_models = {
            "anthropic": "claude-3-7-sonnet-20250219",
            "openai": "gpt-4o",
            "groq": "groq/llama3-70b-8192",
            "google": "gemini/gemini-2.5-pro" # Added Google with requested model
        }

        # Set up API key based on provider
        self.api_key = self._setup_api_key(api_key)

        # Set up model
        self.model = self._setup_model()

    def _setup_api_key(self, api_key: Optional[str] = None) -> str:
        """
        Set up API key with proper fallbacks

        Args:
            api_key: API key provided directly

        Returns:
            API key to use

        Raises:
            ValueError: If no API key is available
        """
        if api_key:
            return api_key

        env_key = self.provider_key_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
        config_key = f"{self.provider}.api_key"

        # Try to get key from config or environment
        key = (
            self.config.get(config_key) or
            os.environ.get(env_key) or
            os.environ.get("LLM_API_KEY")
        )

        if not key:
            raise ValueError(f"{self.provider} API key is required. Set it in config, environment, or provide explicitly.")

        # Set environment variable for litellm
        os.environ[env_key] = key

        return key

    def _setup_model(self) -> str:
        """
        Set up model with proper fallbacks

        Returns:
            Model name to use
        """
        return (
            self.config.get(f"{self.provider}.model") or
            self.config.get("model", self.default_models.get(self.provider, self.default_models["anthropic"]))
        )


class ToolManager:
    """Manages tool registration, formatting, and execution"""

    def __init__(
        self,
        provider: str,
        tool_registry: Optional[Any] = None,
        mcp_manager: Optional[MCPManager] = None
    ):
        """
        Initialize tool manager

        Args:
            provider: Provider name for proper formatting
            tool_registry: Tool registry instance
            mcp_manager: MCP manager instance
        """
        self.provider = provider
        self.tool_registry = tool_registry
        self.mcp_manager = mcp_manager

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools, using provider-specific format

        Returns:
            List of tool definitions in provider-specific format
        """
        if not self.tool_registry:
            return []

        # For Anthropic, use the format that works with LiteLLM
        if self.provider == "anthropic":
            from fei.tools.definitions import ANTHROPIC_TOOL_DEFINITIONS
            tools = []

            # Include tools from registry in format that works with LiteLLM
            for tool in ANTHROPIC_TOOL_DEFINITIONS:
                # Format according to LiteLLM's expected format for Anthropic
                tool_copy = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                tools.append(tool_copy)

            return tools
        else:
            # Standard format for other providers
            tools = self.tool_registry.get_tools()

            # Ensure all tools have a type for OpenAI compatibility
            for tool in tools:
                if "type" not in tool:
                    tool["type"] = "function"

            return tools

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool safely

        Args:
            tool_name: Tool name
            tool_args: Tool arguments

        Returns:
            Tool result or error information
        """
        if not self.tool_registry:
            return {"error": "No tool registry available"}

        # Ensure logger is at DEBUG level for this critical section
        tool_logger = get_logger(__name__) # Get logger for this module
        original_level = tool_logger.level
        if tool_logger.level > logging.DEBUG:
             tool_logger.setLevel(logging.DEBUG)
             tool_logger.debug("Temporarily set assistant logger to DEBUG for tool execution.")

        tool_logger.debug(f"Attempting to execute tool: {tool_name}")
        tool_logger.debug(f"Arguments: {json.dumps(tool_args, indent=2)}")

        # Run tool execution in an executor to avoid blocking
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.tool_registry.execute_tool(tool_name, tool_args)
            )
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return {"error": f"Tool execution error: {str(e)}"}

        # Ensure result is serializable
        try:
            json.dumps(result)
        except (TypeError, OverflowError) as e:
            tool_logger.warning(f"Tool result not JSON serializable: {e}")
            result_data = {"error": f"Tool returned non-serializable result: {e}", "partial_result": str(result)[:1000]}
        else:
             result_data = result

        tool_logger.debug(f"Tool {tool_name} execution result: {json.dumps(result_data, indent=2)}")

        # Restore original logger level if changed
        if tool_logger.level != original_level:
            tool_logger.setLevel(original_level)
            tool_logger.debug("Restored original assistant logger level.")

        return result_data

# Nested ConversationManager removed; using fei.core.conversation_manager.ConversationManager




class Assistant:
    """
    Main assistant class for Fei code assistant.

    This class serves as the main entry point for interacting with
    the Fei code assistant. It delegates to specialized components
    for specific functionality.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tool_registry: Optional[UnifiedToolRegistry] = None,
        mcp_manager: Optional[MCPManager] = None,
        consent_ui_provider: Optional[ConsentUiProvider] = None
    ):
        """
        Initialize the assistant.

        Args:
            config: Configuration object
            api_key: API key (overrides config)
            model: Model to use (overrides config)
            provider: Provider to use (anthropic, openai, etc.)
            tool_registry: Tool registry instance
            mcp_manager: MCP manager instance
            consent_ui_provider: Function to ask user for consent via UI
        """
        self.config = config or Config()

        # Create mediator
        self.mediator = Mediator()

        # Create provider client
        self.provider_client = ProviderClient(
            self.config,
            self.mediator,
            provider,
            api_key,
            model
        )

        # Create components
        self.model_optimizer = ModelOptimizer(self.config)
        # Initialize MCP manager (pass consent handler only)
        self.mcp_manager = mcp_manager or MCPManager(
            self.config,
            consent_handler=self._handle_mcp_consent
        )
        self.tool_registry = tool_registry or UnifiedToolRegistry(self.mediator)
        self.tool_executor = ToolExecutor(
            self.provider_client.provider,
            self.mcp_manager,
            self.mediator,
            self.tool_registry
        )
        self.conversation_manager = ConversationManager(
            self.provider_client.provider,
            self.mediator,
            self.config,
            model=self.provider_client.model
        )

        # Create orchestrator
        self.orchestrator = AssistantOrchestrator(
            self.config,
            self.provider_client,
            self.tool_executor,
            self.conversation_manager,
            self.model_optimizer,
            self.mcp_manager,
            self.mediator,
            consent_ui_provider
        )

        # Register components with mediator
        self._register_with_mediator()

    def _register_with_mediator(self) -> None:
        """Register components with the mediator."""
        self.mediator.register_component("assistant", self)
        self.mediator.register_component("provider_client", self.provider_client)
        self.mediator.register_component("tool_executor", self.tool_executor)
        self.mediator.register_component("conversation_manager", self.conversation_manager)
        self.mediator.register_component("model_optimizer", self.model_optimizer)
        self.mediator.register_component("mcp_manager", self.mcp_manager)
        self.mediator.register_component("tool_registry", self.tool_registry)
        self.mediator.register_component("orchestrator", self.orchestrator)

    @property
    def conversation(self) -> List[Dict[str, Any]]:
        """Get current conversation."""
        return self.conversation_manager.get_messages()

    @property
    def provider(self) -> str:
        """Get current provider."""
        return self.provider_client.provider

    @property
    def model(self) -> str:
        """Get current model."""
        return self.provider_client.model

    @model.setter
    def model(self, value: str) -> None:
        """Set current model."""
        self.provider_client.model = value

    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments

        Returns:
            Tool result
        """
        return await self.tool_executor.execute_tool(tool_name, tool_args)

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools.

        Returns:
            List of tool definitions
        """
        return self.tool_executor.get_tools()

    async def process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process tool calls from LLM

        Args:
            tool_calls: List of tool calls

        Returns:
            List of tool results
        """
        results = []

        for tool_call in tool_calls:
            # Ensure required fields exist
            if not all(key in tool_call for key in ["name", "id", "input"]):
                logger.warning(f"Skipping invalid tool call: {tool_call}")
                continue

            tool_name = tool_call["name"]
            tool_id = tool_call["id"]

            # Safe parsing of tool arguments
            try:
                tool_args = json.loads(tool_call["input"]) if isinstance(tool_call["input"], str) else tool_call["input"]
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing tool arguments: {e}")
                results.append({
                    "type": "tool_result",
                    "tool_call_id": tool_id,
                    "tool_use_id": tool_id,
                    "name": tool_name,
                    "content": json.dumps({"error": f"Invalid JSON in tool arguments: {str(e)}"})
                })
                continue
            # --- Memory Tool Special Handling ---
            original_model = self.model
            if tool_name.startswith("memory_") or tool_name.startswith("memdir_"):
                # Temporarily switch to Gemini Flash for memory operations
                flash_model_id = "gemini/gemini-1.5-flash" # Found via search
                logger.debug(f"Switching to {flash_model_id} for memory tool: {tool_name}")
                self.model = flash_model_id
                # Note: This assumes the API key works for Flash.
                # A more robust solution might involve separate ProviderManagers.

            try:
                result = await self.execute_tool(tool_name, tool_args)
            finally:
                # Switch back to the original model
                if self.model != original_model:
                    logger.debug(f"Switching back to original model: {original_model}")
                    self.model = original_model
            # --- End Memory Tool Special Handling ---

            # Format compatible with LiteLLM and both Anthropic/OpenAI
            results.append({
                "type": "tool_result",
                "tool_call_id": tool_id,  # OpenAI format
                "tool_use_id": tool_id,   # Anthropic format
                "name": tool_name,        # Additional info for compatibility
                "content": json.dumps(result)
            })

        return results

    async def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a message to the assistant and get a response.

        Args:
            message: User message
            system_prompt: Optional system prompt

        Returns:
            Assistant response
        """
        return await self.orchestrator.chat(message, system_prompt)

    async def _send_message_to_llm(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
            tools: Optional[List[Dict[str, Any]]] = None,
            use_optimized_model: bool = True # Flag to control optimization
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Send a message to the LLM and extract tool calls, potentially using an optimized model.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            tools: Optional tools to include

        Returns:
            Tuple of (response_content, tool_calls)

        Raises:
            Various exceptions based on error type
        """
        # Determine which model to use for this specific call
        model_to_use = self.model
        if use_optimized_model:
            # Note: This optimization happens *before* the memory tool specific override
            model_to_use = self.model_optimizer.get_optimal_model(self.provider, self.model)
            if model_to_use != self.model:
                 logger.debug(f"Using optimized model for LLM call: {model_to_use}")

        # Set up parameters
        params = {
            "model": model_to_use, # Use the determined model
            "max_tokens": 4000,
            "messages": messages,
            "api_key": self.provider_manager.api_key # Explicitly pass API key
        }

        if system_prompt:
            params["system"] = system_prompt

        if tools:
            params["tools"] = tools

        try:
            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: litellm_completion(**params)
            )

            # Store message ID
            self.conversation_manager.last_message_id = response.id

            # Extract tool calls and content
            tool_calls = self._extract_tool_calls(response)

            # Extract content safely with proper error handling
            content = self._extract_response_content(response)

            return content, tool_calls

        except (ServiceUnavailableError, RateLimitError) as e:
            logger.error(f"Service error with {self.provider}: {e}")
            raise RuntimeError(f"Service error with {self.provider}: {e}")
        except InvalidRequestError as e:
            logger.error(f"Invalid request to {self.provider}: {e}")
            raise ValueError(f"Invalid request to {self.provider}: {e}")
        except APIError as e:
            logger.error(f"API error with {self.provider}: {e}")
            raise RuntimeError(f"API error with {self.provider}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error communicating with {self.provider}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Unexpected error: {str(e)}")

    def _extract_tool_calls(self, response) -> List[Dict[str, Any]]:
        """
        Extract tool calls from LLM response

        Args:
            response: LLM response object

        Returns:
            List of normalized tool calls
        """
        tool_calls = []
        logger.debug(f"Attempting to extract tool calls from raw response: {response!r}") # Log raw response

        try:
            # Try different formats based on provider
            response_tool_calls = None

            # OpenAI format
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    response_tool_calls = message.tool_calls

            # Anthropic format
            if not response_tool_calls and hasattr(response, 'tool_calls'):
                response_tool_calls = response.tool_calls

            # Extract and normalize tool calls
            if response_tool_calls:
                for tool_call in response_tool_calls:
                    # Handle different response formats
                    if hasattr(tool_call, 'function'):
                        # OpenAI format
                        tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "input": tool_call.function.arguments
                        })
                    elif hasattr(tool_call, 'name'):
                        # Anthropic format
                        tool_calls.append({
                            "id": tool_call.id,
                            "name": tool_call.name,
                            "input": tool_call.input
                        })
            logger.debug(f"Successfully extracted tool calls: {tool_calls}")
        except Exception as e:
            logger.error(f"Error extracting tool calls: {e}", exc_info=True)
            # Don't raise here - just return an empty list

        if not tool_calls:
            logger.debug("No tool calls found in the response.")

        return tool_calls

    def _extract_response_content(self, response) -> str:
        """
        Extract content from LLM response with proper error handling

        Args:
            response: LLM response object

        Returns:
            Response content string
        """
        try:
            if hasattr(response, 'choices') and response.choices and len(response.choices) > 0:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content is not None:
                    return message.content

            # Fallback with generic message
            return "I'll help you with that."

        except Exception as e:
            logger.error(f"Error extracting response content: {e}", exc_info=True)
            return "I'll help you with that."

    async def _send_continuation(self, tools: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Send a continuation message after tool execution

        Args:
            tools: Optional tools to include
            Final response after tool execution
        """
        # Determine which model to use for the continuation call
        model_to_use = self.model_optimizer.get_optimal_model(self.provider, self.model)
        if model_to_use != self.model:
            logger.debug(f"Using optimized model for continuation call: {model_to_use}")

        try:
            # Set continuation parameters
            continue_params = {
                "model": model_to_use, # Use the determined model
                "max_tokens": 4000,
                "messages": self.conversation,
                "api_key": self.provider_manager.api_key # Explicitly pass API key
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

            # Add final assistant response to conversation
            self.conversation_manager.add_assistant_message(answer)

            return answer

        except Exception as e:
            logger.error(f"Error in continuation: {e}", exc_info=True)
            error_msg = f"I tried to use tools to answer your question, but encountered an error: {str(e)}"
            self.conversation_manager.add_assistant_message(error_msg)
            return error_msg

    async def _handle_mcp_consent(
        self,
        server_id: str,
        service: str,
        method: str,
        params: Dict[str, Any]
    ) -> bool:
        """
        Handle MCP consent requests.

        Args:
            server_id: The ID of the MCP server requesting action
            service: The name of the service being called
            method: The name of the method being called
            params: The parameters for the method call

        Returns:
            True if consent is granted, False otherwise
        """
        # Delegate to orchestrator
        return await self.orchestrator._handle_mcp_consent(
            server_id,
            service,
            method,
            params
        )

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_manager.reset()

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler_func: Callable,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> None:
        """
        Register a new tool.

        Args:
            name: Tool name
            description: Tool description
            input_schema: Tool input schema
            handler_func: Tool handler function
            tags: Optional tags
            category: Optional category
        """
        self.tool_registry.register_tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler_func=handler_func,
            tags=tags,
            category=category
        )

    def register_class_methods(
        self,
        cls: Any,
        prefix: str = "",
        tags: Optional[List[str]] = None,
        category: Optional[str] = None
    ) -> None:
        """
        Register all methods of a class as tools.

        Args:
            cls: Class instance
            prefix: Prefix for tool names
            tags: Tags to apply to all registered tools
            category: Category to place tools in
        """
        self.tool_registry.register_class_methods(
            cls=cls,
            prefix=prefix,
            tags=tags,
            category=category
        )
