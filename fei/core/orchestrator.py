"""
Assistant orchestrator for Fei code assistant.

This module contains the AssistantOrchestrator class, which coordinates
the high-level flow of the assistant, delegating to specialized components.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Union, Callable, Awaitable

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.core.provider_client import ProviderClient
from fei.core.tool_executor import ToolExecutor
from fei.core.conversation_manager import ConversationManager
from fei.core.model_optimizer import ModelOptimizer
from fei.core.mcp import MCPManager, MCPConsentDeniedError
from fei.core.mediator import Mediator

logger = get_logger(__name__)

# Type alias for the UI function that asks the user for consent
ConsentUiProvider = Callable[[str], Awaitable[bool]]


class AssistantOrchestrator:
    """
    Orchestrates the high-level flow of the assistant.

    This class is responsible for:
    1. Coordinating the conversation flow
    2. Delegating to specialized components
    3. Managing the overall assistant state
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        provider_client: Optional[ProviderClient] = None,
        tool_executor: Optional[ToolExecutor] = None,
        conversation_manager: Optional[ConversationManager] = None,
        model_optimizer: Optional[ModelOptimizer] = None,
        mcp_manager: Optional[MCPManager] = None,
        mediator: Optional[Mediator] = None,
        consent_ui_provider: Optional[ConsentUiProvider] = None
    ):
        """
        Initialize the assistant orchestrator.

        Args:
            config: Configuration object
            provider_client: Provider client for LLM communication
            tool_executor: Tool executor for executing tools
            conversation_manager: Conversation manager for managing conversation state
            model_optimizer: Model optimizer for selecting optimal models
            mcp_manager: MCP manager for managing model-controlled processes
            mediator: Mediator for component communication
            consent_ui_provider: Function to ask user for consent via UI
        """
        self.config = config or Config()
        self.consent_ui_provider = consent_ui_provider

        # Initialize mediator first
        self.mediator = mediator or Mediator()

        # Initialize components with dependency injection
        self.provider_client = provider_client or ProviderClient(self.config, self.mediator)
        self.model_optimizer = model_optimizer or ModelOptimizer(self.config)

        # Initialize MCP manager with consent handler
        self.mcp_manager = mcp_manager or MCPManager(
            self.config,
            consent_handler=self._handle_mcp_consent,
            mediator=self.mediator
        )

        # Initialize tool executor with MCP manager
        self.tool_executor = tool_executor or ToolExecutor(
            self.provider_client.provider,
            self.mcp_manager,
            self.mediator
        )

        # Initialize conversation manager
        self.conversation_manager = conversation_manager or ConversationManager(
            self.provider_client.provider,
            self.mediator,
            self.config,
            model=self.provider_client.model
        )

        # Register components with mediator
        self._register_with_mediator()

    def _register_with_mediator(self) -> None:
        """Register components with the mediator."""
        self.mediator.register_component("orchestrator", self)
        self.mediator.register_component("provider_client", self.provider_client)
        self.mediator.register_component("tool_executor", self.tool_executor)
        self.mediator.register_component("conversation_manager", self.conversation_manager)
        self.mediator.register_component("model_optimizer", self.model_optimizer)
        self.mediator.register_component("mcp_manager", self.mcp_manager)

    @property
    def conversation(self) -> List[Dict[str, Any]]:
        """Get current conversation for backward compatibility."""
        return self.conversation_manager.get_messages()

    @property
    def provider(self) -> str:
        """Get current provider for backward compatibility."""
        return self.provider_client.provider

    @property
    def model(self) -> str:
        """Get current model for backward compatibility."""
        return self.provider_client.model

    @model.setter
    def model(self, value: str) -> None:
        """Set current model for backward compatibility."""
        self.provider_client.model = value

    async def chat(self, message: str, system_prompt: Optional[str] = None) -> str:
        """
        Send a message to the LLM and process any tool calls.

        Args:
            message: User message
            system_prompt: Optional system prompt to override default

        Returns:
            Assistant response
        """
        # Add user message to conversation
        # This may return a response if compaction occurred and generated a response
        compaction_response = self.conversation_manager.add_user_message(message)

        # If we got a response from compaction, use it directly
        if compaction_response:
            # Add the compaction-generated response to conversation
            self.conversation_manager.add_assistant_message(compaction_response)
            return compaction_response

        # Determine the model to use (potentially optimized)
        # Determine the optimal model to use (await async evaluation)
        effective_model = await self.model_optimizer.get_optimal_model(
            self.provider_client.provider,
            self.provider_client.model
        )
        logger.info(f"Using model: {effective_model}")

        # Temporarily set the optimized model
        original_model = self.provider_client.model
        self.provider_client.model = effective_model

        try:
            # Get tools
            tools = self.tool_executor.get_tools()

            # Send message to LLM
            initial_response, tool_calls = await self.provider_client.send_message(
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
                tool_results = await self._process_tool_calls(tool_calls)

                # Add tool results to conversation
                self.conversation_manager.add_tool_results(tool_results)

                logger.info("Continuing conversation with tool results")

                # Send continuation message to get final response after tool execution
                answer = await self.provider_client.send_continuation(
                    self.conversation,
                    tools
                )

            else:
                # No tool calls, just add the response to conversation
                self.conversation_manager.add_assistant_message(initial_response)
                answer = initial_response

            return answer
        finally:
            # Restore original model
            self.provider_client.model = original_model

    async def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process tool calls from LLM.

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

            # Parse tool arguments
            tool_args = tool_call["input"]
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON in tool arguments: {tool_args}")
                    continue

            # Execute tool with special handling for memory tools
            result = await self._execute_tool_with_special_handling(tool_name, tool_args)

            # Format compatible with LiteLLM and both Anthropic/OpenAI
            results.append({
                "type": "tool_result",
                "tool_call_id": tool_id,  # OpenAI format
                "tool_use_id": tool_id,   # Anthropic format
                "name": tool_name,        # Additional info for compatibility
                "content": json.dumps(result)
            })

        return results

    async def _execute_tool_with_special_handling(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with special handling for certain tool types.

        Args:
            tool_name: Tool name
            tool_args: Tool arguments

        Returns:
            Tool result
        """
        # Special handling for memory tools
        original_model = self.provider_client.model
        try:
            if tool_name.startswith("memory_") or tool_name.startswith("memdir_"):
                # Temporarily switch to Gemini Flash for memory operations
                flash_model_id = "gemini/gemini-1.5-flash"
                logger.debug(f"Switching to {flash_model_id} for memory tool: {tool_name}")
                self.provider_client.model = flash_model_id

            # Execute the tool
            return await self.tool_executor.execute_tool(tool_name, tool_args)
        finally:
            # Restore original model
            if self.provider_client.model != original_model:
                logger.debug(f"Switching back to original model: {original_model}")
                self.provider_client.model = original_model

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
        if not self.consent_ui_provider:
            logger.warning("No consent UI provider available, denying MCP action")
            return False

        # Format a user-friendly prompt
        prompt = (
            f"The assistant wants to use an external service:\n"
            f"- Server: {server_id}\n"
            f"- Service: {service}\n"
            f"- Method: {method}\n"
            f"- Parameters: {json.dumps(params, indent=2)}\n\n"
            f"Do you approve this action?"
        )

        try:
            # Call the UI provider to get user consent
            approved = await self.consent_ui_provider(prompt)
            if approved:
                logger.info(f"User approved MCP action: {server_id}.{service}.{method}")
                return True
            else:
                logger.info(f"User denied MCP action: {server_id}.{service}.{method}")
                return False
        except Exception as e:
            logger.error(f"Error getting user consent via UI for {server_id}.{service}.{method}: {e}", exc_info=True)
            # Deny on error for safety
            return False

    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_manager.reset()
