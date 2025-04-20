"""
AutoSummarizer for managing conversation context by token count.

This module provides a mechanism to automatically summarize conversations
when they reach a configurable token threshold.
"""

import asyncio
import threading
import time
import logging
from typing import Dict, Any, List, Tuple, Optional

from fei.utils.config import Config
from fei.utils.token_counter import TokenCounter
from fei.core.summarization_prompts import get_summary_prompt

logger = logging.getLogger(__name__)


class AutoSummarizer:
    """
    AutoSummarizer manages conversation context by summarizing when token count exceeds threshold.

    This class monitors the token count of a conversation and triggers summarization
    when the count exceeds a configurable threshold. The summarization is performed
    by the LLM, and the conversation continues with the summarized context.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        provider: str = "anthropic",
        model: Optional[str] = None,
        summarize_provider: Optional[str] = None,
        summarize_model: Optional[str] = None,
        token_threshold: Optional[int] = None,
        tail_k: Optional[int] = None,
        summary_style: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize auto summarizer

        Args:
            config: Configuration object
            provider: Provider name for token counting
            model: Model name for token counting
            summarize_provider: Provider to use for summarization
            summarize_model: Model to use for summarization
            token_threshold: Token count threshold to trigger summarization
            tail_k: Number of recent messages to keep verbatim
            summary_style: Style for summaries (bullet, narrative, technical)
            enabled: Whether auto summarization is enabled
        """
        self.config = config or Config()
        self.provider = provider
        self.model = model

        # Load configuration
        self.enabled = enabled if enabled is not None else self.config.get_bool("auto_summarizer.enabled", False)
        self.tail_k = tail_k or self.config.get_int("auto_summarizer.tail_k", 3)
        self.token_threshold = token_threshold or self.config.get_int("auto_summarizer.token_threshold", 30000)
        self.summary_style = summary_style or self.config.get_string("auto_summarizer.summary_style", "bullet")

        # Set up summarization model - default to using the same model as the main conversation
        # This ensures we don't have authentication issues with different providers
        self.summarize_provider = summarize_provider or self.config.get_string(
            "auto_summarizer.summarize_provider", provider
        )
        self.summarize_model = summarize_model or self.config.get_string(
            "auto_summarizer.summarize_model", model
        )

        # Log the summarization model being used
        logger.info(f"Using {self.summarize_provider}/{self.summarize_model} for summarization")

        # Initialize token counter
        self.token_counter = TokenCounter(provider, model)

        # Initialize state
        self.lock = threading.Lock()
        self.raw_log = []  # [(role, content), ...]
        self.summary = ""
        self.last_summarization_time = time.time()
        self.current_token_count = 0

    def add_message(self, message: Dict[str, Any]) -> None:
        """
        Add a message to the conversation and check if summarization is needed

        Args:
            message: Message to add
        """
        if not self.enabled:
            return

        with self.lock:
            # Extract role and content
            role = message.get("role", "unknown")

            # Handle content that could be a list or string
            content = message.get("content", "")
            if isinstance(content, list):
                # For messages with multiple parts, join text parts
                content_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        content_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        content_parts.append(part)
                content = "\n".join(content_parts)

            # Add to raw log
            self.raw_log.append((role, content))

            # Update token count
            message_tokens = self.token_counter.count_tokens_in_text(content)
            message_tokens += self.token_counter.count_tokens_in_text(role)
            self.current_token_count += message_tokens

            logger.debug(f"Added message with {message_tokens} tokens. Current total: {self.current_token_count}")

            # Check if summarization is needed
            if self._needs_summarization():
                logger.info(f"Token threshold of {self.token_threshold} exceeded. Starting summarization.")
                self._summarize()

    def _needs_summarization(self) -> bool:
        """
        Check if summarization is needed based on token count

        Returns:
            Whether summarization is needed
        """
        # Don't summarize if we have fewer messages than tail_k
        if len(self.raw_log) <= self.tail_k:
            logger.debug(f"Not enough messages for summarization: {len(self.raw_log)} <= {self.tail_k}")
            return False

        # Check if we've exceeded the token threshold
        logger.debug(f"Current token count: {self.current_token_count}, threshold: {self.token_threshold}")
        if self.current_token_count > self.token_threshold:
            logger.info(f"Summarization needed: {self.current_token_count} tokens exceed threshold of {self.token_threshold}")
            return True

        return False

    def _summarize(self) -> None:
        """
        Summarize the oldest messages and replace them with a summary
        """
        # Keep the most recent tail_k messages
        if len(self.raw_log) <= self.tail_k:
            return

        # Split messages - keep the most recent tail_k messages intact
        head = self.raw_log[:-self.tail_k]  # Messages to summarize
        tail = self.raw_log[-self.tail_k:]  # Messages to keep

        logger.info(f"Starting summarization: summarizing {len(head)} messages, keeping {len(tail)} recent messages")

        # Calculate token counts before summarization for logging
        head_tokens = sum(self.token_counter.count_tokens_in_text(content) for _, content in head)
        logger.info(f"Messages to be summarized contain approximately {head_tokens} tokens")

        try:
            # Create a new event loop for the thread
            loop = asyncio.new_event_loop()
            result_container = [None]  # Use a container to store the result

            def run_async_summarization():
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.summarize_async(head))
                    result_container[0] = result  # Store the result
                except Exception as e:
                    logger.error(f"Error in summarization thread: {e}", exc_info=True)
                finally:
                    loop.close()

            # Create and start thread
            thread = threading.Thread(target=run_async_summarization)
            thread.start()
            thread.join(timeout=30)  # Wait up to 30 seconds

            if thread.is_alive():
                logger.warning("Summarization timed out, using fallback")
                new_summary = f"[Summarization timed out for {len(head)} messages]"
            else:
                # Get the result from the container
                if result_container[0] is None:
                    logger.error("Summarization failed to produce a result")
                    new_summary = f"[Summarization failed for {len(head)} messages]"
                else:
                    # Unpack the result
                    new_summary = result_container[0]

            # Create a summary message in the format expected by the LLM
            summary_message = ("system", f"Previous conversation summary: {new_summary}")

            # Update raw log - replace head with summary message and keep tail
            self.raw_log = [summary_message] + tail

            # Recalculate token count
            self.current_token_count = 0
            for role, content in self.raw_log:
                self.current_token_count += self.token_counter.count_tokens_in_text(content)
                self.current_token_count += self.token_counter.count_tokens_in_text(role)

            # Add overhead for message formatting
            self.current_token_count += len(self.raw_log) * 4

            # Calculate token counts after summarization for logging
            summary_tokens = self.token_counter.count_tokens_in_text(new_summary)
            logger.info(f"Summary contains {summary_tokens} tokens (reduced from {head_tokens} tokens)")
            logger.info(f"New total token count: {self.current_token_count}")

            # Update last summarization time
            self.last_summarization_time = time.time()

            logger.info(f"Summarization complete: {len(head)} messages summarized into 1 summary message, {len(tail)} recent messages kept")

        except Exception as e:
            logger.error(f"Error during summarization: {e}", exc_info=True)
            # Don't update state on error

    async def summarize_async(self, messages: List[Tuple[str, str]]) -> str:
        """
        Summarize messages asynchronously

        Args:
            messages: List of (role, content) tuples to summarize

        Returns:
            Summary text
        """
        try:
            # Import here to avoid circular imports
            from litellm import completion as litellm_completion

            # Get the appropriate summary prompt based on style
            prompt = get_summary_prompt(messages, self.summary_style)

            # Handle model name format based on provider
            if self.summarize_provider == "google":
                model_name = self.summarize_model
            else:
                model_name = f"{self.summarize_provider}/{self.summarize_model}"

            logger.debug(f"Summarizing with model: {model_name}")

            params = {
                "model": model_name,
                "messages": prompt,
                "temperature": 0.2,
                "max_tokens": 1000
            }

            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: litellm_completion(**params)
            )

            # Extract content
            content = response.choices[0].message.content.strip()
            logger.debug(f"Received summarization response: {content[:100]}...")

            return content

        except Exception as e:
            logger.error(f"Error summarizing messages: {e}", exc_info=True)
            # Return a simple fallback summary
            return f"[Failed to generate summary: {str(e)}]"

    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get the current messages for the conversation

        Returns:
            List of message dictionaries
        """
        with self.lock:
            return [{"role": role, "content": content} for role, content in self.raw_log]

    def get_token_count(self) -> int:
        """
        Get the current token count

        Returns:
            Current token count
        """
        with self.lock:
            return self.current_token_count

    def clear(self) -> None:
        """
        Clear the conversation history
        """
        with self.lock:
            self.raw_log = []
            self.summary = ""
            self.current_token_count = 0
            self.last_summarization_time = time.time()
