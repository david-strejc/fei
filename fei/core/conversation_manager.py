"""
Conversation manager for Fei code assistant.

This module contains the ConversationManager class, which handles
conversation state and message formatting.
"""

import json
from typing import Dict, List, Optional, Any, Union

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.core.mediator import Mediator
from fei.core.prompt_compactor import PromptCompactor
from fei.core.auto_summarizer import AutoSummarizer

logger = get_logger(__name__)


class ConversationManager:
    """
    Manages conversation state and message formatting.

    This class is responsible for:
    1. Maintaining conversation history
    2. Formatting messages for different providers
    3. Adding user, assistant, and tool messages
    4. Optionally compacting conversation history
    """

    def __init__(self,
                 provider: str,
                 mediator: Optional[Mediator] = None,
                 config: Optional[Config] = None,
                 enable_compaction: Optional[bool] = None,
                 enable_auto_summarization: Optional[bool] = None,
                 model: Optional[str] = None):
        """
        Initialize conversation manager.

        Args:
            provider: Provider name for proper formatting
            mediator: Mediator for component communication
            config: Configuration object
            enable_compaction: Whether to enable conversation compaction
            enable_auto_summarization: Whether to enable auto summarization
            model: Model name for token counting
        """
        self.provider = provider
        self.mediator = mediator
        self.config = config or Config()
        self.model = model
        self.conversation = []
        self.last_message_id = None

        # Initialize prompt compactor if enabled
        self.enable_compaction = enable_compaction
        if self.enable_compaction is None:
            self.enable_compaction = self.config.get_bool("compactor.enabled", False)

        self.prompt_compactor = None
        if self.enable_compaction:
            self.prompt_compactor = PromptCompactor(
                config=self.config,
                provider=self.provider,
                model=self.model
            )
            logger.debug("Initialized PromptCompactor for conversation")

        # Initialize auto summarizer if enabled
        self.enable_auto_summarization = enable_auto_summarization
        if self.enable_auto_summarization is None:
            self.enable_auto_summarization = self.config.get_bool("auto_summarizer.enabled", False)

        self.auto_summarizer = None
        if self.enable_auto_summarization:
            self.auto_summarizer = AutoSummarizer(
                config=self.config,
                provider=self.provider,
                model=self.model
            )
            logger.debug("Initialized AutoSummarizer for conversation")

    def add_user_message(self, message: str) -> Optional[str]:
        """
        Add a user message to the conversation.

        Args:
            message: User message content

        Returns:
            Optional AI response if compaction occurred and generated a response
        """
        user_message = {"role": "user", "content": message}
        self.conversation.append(user_message)

        # Add to prompt compactor if enabled, passing the current message for potential response
        response = None
        if self.prompt_compactor:
            response = self.prompt_compactor.add_message(user_message, message)

        # Add to auto summarizer if enabled
        if self.auto_summarizer:
            self.auto_summarizer.add_message(user_message)
            # Check if we need to update conversation with summarized messages
            if self._should_update_from_summarizer():
                self._update_from_summarizer()

        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("user_message_added", {
                "message": message,
                "conversation_length": len(self.conversation)
            })

        return response

    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add an assistant message to the conversation.

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

        # Add to prompt compactor if enabled
        if self.prompt_compactor:
            self.prompt_compactor.add_message(assistant_message)

        # Add to auto summarizer if enabled
        if self.auto_summarizer:
            self.auto_summarizer.add_message(assistant_message)
            # Check if we need to update conversation with summarized messages
            if self._should_update_from_summarizer():
                self._update_from_summarizer()

        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("assistant_message_added", {
                "content": content,
                "has_tool_calls": bool(tool_calls),
                "tool_call_count": len(tool_calls) if tool_calls else 0,
                "conversation_length": len(self.conversation)
            })

    def add_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        """
        Add tool results to the conversation.

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
                tool_message = tool_result_message
                self.conversation.append(tool_message)

                # Add to prompt compactor if enabled
                if self.prompt_compactor:
                    self.prompt_compactor.add_message(tool_message)

                # Add to auto summarizer if enabled
                if self.auto_summarizer:
                    self.auto_summarizer.add_message(tool_message)
        else:
            # Format for other providers
            for result in tool_results:
                # Add each result as a separate tool message for better visibility
                tool_message = {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["content"]
                }
                self.conversation.append(tool_message)

                # Add to prompt compactor if enabled
                if self.prompt_compactor:
                    self.prompt_compactor.add_message(tool_message)

                # Add to auto summarizer if enabled
                if self.auto_summarizer:
                    self.auto_summarizer.add_message(tool_message)

        # Check if we need to update conversation with summarized messages
        if self.auto_summarizer and self._should_update_from_summarizer():
            self._update_from_summarizer()

        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("tool_results_added", {
                "result_count": len(tool_results),
                "conversation_length": len(self.conversation)
            })

    def get_messages(self, use_compaction: Optional[bool] = None, use_auto_summarization: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Get current conversation messages.

        Args:
            use_compaction: Whether to use compaction (defaults to self.enable_compaction)
            use_auto_summarization: Whether to use auto summarization (defaults to self.enable_auto_summarization)

        Returns:
            List of conversation messages, potentially compacted or summarized
        """
        # Determine whether to use compaction and auto summarization
        should_compact = self.enable_compaction if use_compaction is None else use_compaction
        should_auto_summarize = self.enable_auto_summarization if use_auto_summarization is None else use_auto_summarization

        # Check if we should update from auto summarizer
        if should_auto_summarize and self.auto_summarizer and self._should_update_from_summarizer():
            self._update_from_summarizer()

        # Return compacted messages if enabled and available
        if should_compact and self.prompt_compactor:
            return self.prompt_compactor.get_compacted_messages(self.conversation)

        # Otherwise return full conversation
        return self.conversation

    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation = []
        self.last_message_id = None

        # Reset prompt compactor if enabled
        if self.prompt_compactor:
            self.prompt_compactor.reset()

        # Reset auto summarizer if enabled
        if self.auto_summarizer:
            self.auto_summarizer.clear()

        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("conversation_reset", {})

    def set_compaction(self, enabled: bool) -> None:
        """
        Enable or disable conversation compaction

        Args:
            enabled: Whether to enable compaction
        """
        # Update compaction setting
        self.enable_compaction = enabled

        # Initialize or reset prompt compactor
        if enabled and not self.prompt_compactor:
            self.prompt_compactor = PromptCompactor(
                config=self.config,
                provider=self.provider,
                model=self.model
            )
            logger.debug("Initialized PromptCompactor for conversation")
        elif not enabled and self.prompt_compactor:
            self.prompt_compactor = None
            logger.debug("Disabled PromptCompactor for conversation")

    def set_auto_summarization(self, enabled: bool) -> None:
        """
        Enable or disable auto summarization

        Args:
            enabled: Whether to enable auto summarization
        """
        # Update auto summarization setting
        self.enable_auto_summarization = enabled

        # Initialize or reset auto summarizer
        if enabled and not self.auto_summarizer:
            self.auto_summarizer = AutoSummarizer(
                config=self.config,
                provider=self.provider,
                model=self.model
            )
            logger.debug("Initialized AutoSummarizer for conversation")
        elif not enabled and self.auto_summarizer:
            self.auto_summarizer = None
            logger.debug("Disabled AutoSummarizer for conversation")

    def get_token_count(self) -> int:
        """
        Get approximate token count for current conversation

        Returns:
            Approximate token count
        """
        if self.auto_summarizer:
            # Use token count from auto summarizer
            return self.auto_summarizer.get_token_count()
        elif self.prompt_compactor:
            # Use token counter from prompt compactor
            token_counter = self.prompt_compactor.token_counter
            return token_counter.count_tokens_in_messages(self.conversation)
        else:
            # Rough estimation based on character count
            total_chars = sum(len(str(msg.get("content", ""))) for msg in self.conversation)
            return int(total_chars * 0.25)  # Rough estimate of 4 chars per token

    def _should_update_from_summarizer(self) -> bool:
        """
        Check if we should update the conversation from the auto summarizer

        Returns:
            Whether to update from auto summarizer
        """
        if not self.auto_summarizer:
            return False

        # Check if the token count exceeds the threshold
        token_count = self.auto_summarizer.get_token_count()
        token_threshold = self.auto_summarizer.token_threshold

        logger.debug(f"Current token count: {token_count}, threshold: {token_threshold}")

        return token_count > token_threshold

    def _update_from_summarizer(self) -> None:
        """
        Update the conversation with summarized messages from the auto summarizer
        """
        if not self.auto_summarizer:
            return

        # Get summarized messages
        summarized_messages = self.auto_summarizer.get_messages()

        # Update conversation with summarized messages
        self.conversation = summarized_messages

        logger.info(f"Updated conversation with summarized messages. New token count: {self.get_token_count()}")
