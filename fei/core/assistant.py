#!/usr/bin/env python3
"""
Core assistant implementation for Fei

This module provides the main assistant class that handles communication
with LLM providers via LiteLLM and manages tool execution.
"""

import os
import json
import asyncio
import logging # Add missing import
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import lru_cache

from litellm import completion as litellm_completion
from litellm.exceptions import ServiceUnavailableError, APIError, InvalidRequestError, RateLimitError

from fei.tools.registry import ToolRegistry
from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.core.mcp import MCPManager

logger = get_logger(__name__)

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
        tool_registry: Optional[ToolRegistry] = None,
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


class ConversationManager:
    """Manages conversation state and messaging formatting"""
    
    def __init__(self, provider: str):
        """
        Initialize conversation manager
        
        Args:
            provider: Provider name for proper formatting
        """
        self.provider = provider
        self.conversation = []
        self.last_message_id = None
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation
        
        Args:
            message: User message content
        """
        self.conversation.append({"role": "user", "content": message})
    
    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add an assistant message to the conversation
        
        Args:
            content: Assistant message content
            tool_calls: Optional tool calls for the message
        """
        assistant_message = {"role": "assistant", "content": content}
        
        if tool_calls:
            if isinstance(assistant_message["content"], str):
                assistant_message["content"] = [{"type": "text", "text": assistant_message["content"]}]
                
            # Add tool_calls field to match OpenAI format    
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["input"] if isinstance(tc["input"], str) else json.dumps(tc["input"])
                    }
                } for tc in tool_calls
            ]
            
        self.conversation.append(assistant_message)
    
    def add_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        """
        Add tool results to the conversation in provider-specific format
        
        Args:
            tool_results: Tool execution results
        """
        if self.provider == "anthropic":
            # Anthropic tool results format
            for result in tool_results:
                tool_result_message = {
                    "role": "tool", 
                    "tool_call_id": result["tool_use_id"],
                    "name": result["name"],
                    "content": result["content"]
                }
                # Add each tool result as a separate message
                self.conversation.append(tool_result_message)
        else:
            # Format for other providers
            for result in tool_results:
                # Add each result as a separate tool message for better visibility
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["content"]
                })
            # Removed the combined 'user' role message for tool results,
            # as it seems incompatible with Gemini continuation calls via LiteLLM.
            # The individual 'tool' role messages added above should suffice.
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get current conversation messages
        
        Returns:
            List of conversation messages
        """
        return self.conversation
    
    def reset(self) -> None:
        """Reset the conversation history"""
        self.conversation = []
        self.last_message_id = None


class Assistant:
    """Main assistant class for Fei code assistant"""
    
    def __init__(
        self, 
        config: Optional[Config] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        tool_registry: Optional[ToolRegistry] = None,
        mcp_manager: Optional[MCPManager] = None
    ):
        """
        Initialize the assistant
        
        Args:
            config: Configuration object
            api_key: API key (overrides config)
            model: Model to use (overrides config)
            provider: Provider to use (anthropic, openai, etc.)
            tool_registry: Tool registry instance
            mcp_manager: MCP manager instance
        """
        self.config = config or Config()
        
        # Initialize component managers
        self.provider_manager = ProviderManager(self.config, provider, api_key)
        self.provider = self.provider_manager.provider
        self.model = model or self.provider_manager.model
        
        # Initialize MCP manager first as it might be needed by tools
        self.mcp_manager = mcp_manager or MCPManager(self.config)
        
        # Initialize tool manager
        self.tool_manager = ToolManager(self.provider, tool_registry, self.mcp_manager)
        self.tool_registry = tool_registry
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(self.provider)
    
    @property
    def conversation(self) -> List[Dict[str, Any]]:
        """Get current conversation for backward compatibility"""
        return self.conversation_manager.get_messages()
    
    @property
    def last_message_id(self) -> Optional[str]:
        """Get last message ID for backward compatibility"""
        return self.conversation_manager.last_message_id
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools, using provider-specific format
        
        Returns:
            List of tool definitions in provider-specific format
        """
        return self.tool_manager.get_tools()
    
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool
        
        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            
        Returns:
            Tool result
        """
        return await self.tool_manager.execute_tool(tool_name, tool_args)
    
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
            
            result = await self.execute_tool(tool_name, tool_args)
            
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
        Send a message to the LLM and process any tool calls
        
        Args:
            message: User message
            system_prompt: Optional system prompt to override default
            
        Returns:
            Assistant response
        """
        # Add user message to conversation
        self.conversation_manager.add_user_message(message)
        
        logger.info(f"Sending message to {self.provider} model {self.model}")
        
        # Get tools
        tools = self.get_tools()
        
        # Send message to LLM
        initial_response, tool_calls = await self._send_message_to_llm(
            self.conversation,
            system_prompt,
            tools
        )
        
        # If there are tool calls, execute them and continue conversation
        if tool_calls:
            logger.info(f"Processing {len(tool_calls)} tool calls")
            
            # Add assistant message with tool calls
            self.conversation_manager.add_assistant_message(initial_response, tool_calls)
            
            # Execute tools
            tool_results = await self.process_tool_calls(tool_calls)
            
            # Add tool results to conversation
            self.conversation_manager.add_tool_results(tool_results)
            
            logger.info("Continuing conversation with tool results")
            
            # Send continuation message to get final response after tool execution
            answer = await self._send_continuation(tools)
            
        else:
            # No tool calls, just add the response to conversation
            self.conversation_manager.add_assistant_message(initial_response)
            answer = initial_response
        
        return answer
    
    async def _send_message_to_llm(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Send a message to the LLM and extract tool calls
        
        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            tools: Optional tools to include
            
        Returns:
            Tuple of (response_content, tool_calls)
            
        Raises:
            Various exceptions based on error type
        """
        # Set up parameters
        params = {
            "model": self.model,
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
            
        Returns:
            Final response after tool execution
        """
        try:
            # Set continuation parameters
            continue_params = {
                "model": self.model,
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
    
    def reset_conversation(self) -> None:
        """Reset the conversation history"""
        self.conversation_manager.reset()
