#!/usr/bin/env python3
"""
Prompt Compactor for Fei code assistant

This module provides the PromptCompactor class, which manages conversation
context by summarizing older messages to keep the context window lean.
"""

import os
import json
import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable, Awaitable
from collections import deque

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.utils.token_counter import TokenCounter

logger = get_logger(__name__)


class PromptCompactor:
    """
    Manages conversation context by summarizing older messages

    This class is responsible for:
    1. Tracking conversation history
    2. Deciding when to compact the conversation
    3. Summarizing older messages
    4. Providing a compact context for LLM calls
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        provider: str = "anthropic",
        model: Optional[str] = None,
        summarize_provider: Optional[str] = None,
        summarize_model: Optional[str] = None,
        tail_k: Optional[int] = None,
        exchanges_before_compact: Optional[int] = None,
        summary_style: Optional[str] = None,
        enabled: Optional[bool] = None
    ):
        """
        Initialize prompt compactor

        Args:
            config: Configuration object
            provider: Provider name for token counting
            model: Model name for token counting
            summarize_provider: Provider to use for summarization
            summarize_model: Model to use for summarization
            tail_k: Number of recent messages to keep verbatim
            exchanges_before_compact: Number of full exchanges before compaction
            summary_style: Style for summaries (bullet, narrative, etc.)
            enabled: Whether compaction is enabled
        """
        self.config = config or Config()
        self.provider = provider
        self.model = model

        # Load configuration
        self.enabled = enabled if enabled is not None else self.config.get_bool("compactor.enabled", False)
        self.tail_k = tail_k or self.config.get_int("compactor.tail_k", 3)
        self.exchanges_before_compact = exchanges_before_compact or self.config.get_int("compactor.exchanges_before_compact", 3)
        self.summary_style = summary_style or self.config.get_string("compactor.summary_style", "bullet")

        # Set up summarization model - default to using the same model as the main conversation
        # This ensures we don't have authentication issues with different providers
        self.summarize_provider = summarize_provider or self.config.get_string(
            "compactor.summarize_provider", provider
        )
        self.summarize_model = summarize_model or self.config.get_string(
            "compactor.summarize_model", model or "gpt-3.5-turbo"
        )

        # Log the summarization model being used
        logger.info(f"Using {self.summarize_provider}/{self.summarize_model} for summarization")

        # Initialize token counter
        self.token_counter = TokenCounter(provider, model)

        # Initialize state
        self.lock = threading.Lock()
        self.raw_log = []  # [(role, content), ...]
        self.summary = ""
        self.last_compaction_time = time.time()
        self.exchange_count = 0  # Count of full exchanges (user+assistant)

        logger.debug(f"Initialized PromptCompactor (enabled={self.enabled})")

    def add_message(self, message: Dict[str, Any], current_message: Optional[str] = None) -> Optional[str]:
        """
        Add a message to the conversation

        Args:
            message: Message to add
            current_message: Optional current user message to respond to

        Returns:
            Optional response to current message if compaction occurred
        """
        if not self.enabled:
            return None

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

            # Track exchanges - increment when we get an assistant message
            # This counts complete exchanges (user + assistant)
            if role == "assistant":
                self.exchange_count += 1
                logger.debug(f"Exchange count incremented to {self.exchange_count}")

            # Check if compaction is needed
            if self._needs_compaction():
                response = self._compact(current_message)
                # Reset exchange count after compaction
                self.exchange_count = 0
                logger.debug("Exchange count reset after compaction")
                return response

            return None

    def context_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get messages ready for LLM call

        Returns:
            List of messages with summary as system message
        """
        if not self.enabled:
            return []

        with self.lock:
            # Get the most recent messages
            tail = self.raw_log[-self.tail_k:] if len(self.raw_log) > self.tail_k else self.raw_log

            # Create messages list
            messages = [{"role": "system", "content": self.summary}]
            messages.extend([{"role": r, "content": c} for r, c in tail])

            return messages

    def get_compacted_messages(self, original_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply compaction to a list of messages

        Args:
            original_messages: Original messages

        Returns:
            Compacted messages
        """
        if not self.enabled or len(original_messages) <= self.tail_k:
            return original_messages

        with self.lock:
            # Find system message if any
            system_message = None
            non_system_messages = []

            for msg in original_messages:
                if msg.get("role") == "system":
                    system_message = msg
                else:
                    non_system_messages.append(msg)

            # Get the most recent non-system messages
            tail = non_system_messages[-self.tail_k:] if len(non_system_messages) > self.tail_k else non_system_messages

            # Create compacted messages
            if system_message:
                # Combine original system message with summary
                combined_system_content = f"{system_message.get('content', '')}\n\n{self.summary}"
                compacted_messages = [{"role": "system", "content": combined_system_content}]
            else:
                # Use summary as system message
                compacted_messages = [{"role": "system", "content": self.summary}]

            # Add tail messages
            compacted_messages.extend(tail)

            return compacted_messages

    def reset(self) -> None:
        """Reset the compactor state"""
        with self.lock:
            self.raw_log = []
            self.summary = "The dialogue has not been summarized yet."
            self.last_compaction_time = time.time()

    async def summarize_async(self, messages: List[Tuple[str, str]], current_message: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """
        Summarize messages asynchronously

        Args:
            messages: List of (role, content) tuples to summarize
            current_message: Optional current user message to respond to

        Returns:
            Tuple of (summary_text, response_to_current_message)
        """
        try:
            # Import here to avoid circular imports
            from litellm import completion as litellm_completion

            # Format messages for summarization
            chunk_txt = "\n".join(f"{r.upper()}: {c}" for r, c in messages)

            # Create prompt based on whether we have a current message
            if current_message:
                # Combined summarization and response prompt
                prompt = [
                    {"role": "system", "content": (
                        "You are an AI assistant helping with code and technical tasks. "
                        "First, create a VERY CONCISE summary of the previous conversation history. "
                        "Focus only on the most important information, decisions, code snippets, and key context. "
                        f"Use {self.summary_style} style and be extremely brief - aim for 100 words or less. "
                        "Then, respond to the user's latest message. "
                        "Format your response as:\n\n"
                        "SUMMARY: [your concise summary of previous conversation]\n\n"
                        "RESPONSE: [your response to the current message]"
                    )},
                    {"role": "user", "content": f"Previous conversation history:\n{chunk_txt}\n\nCurrent message: {current_message}"}
                ]
            else:
                # Just summarization prompt
                prompt = [
                    {"role": "system", "content": (
                        "You are a summarization engine for a software-agent chat. "
                        f"Create an EXTREMELY CONCISE summary in {self.summary_style} style. "
                        "Focus only on the most important information, decisions, code snippets, and key context. "
                        "Be extremely brief - aim for 100 words or less. "
                        "The summary will replace all these messages in the conversation history, so include only what's essential."
                    )},
                    {"role": "user", "content": chunk_txt}
                ]

            # Set up parameters
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
                "max_tokens": 2000  # Increased for combined summary+response
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

            # Parse the response if we have a current message
            if current_message:
                # Split the response into summary and response parts
                parts = content.split("RESPONSE:")
                if len(parts) == 2:
                    summary_part = parts[0].replace("SUMMARY:", "").strip()
                    response_part = parts[1].strip()
                    logger.info(f"Successfully generated summary ({len(summary_part)} chars) and response")
                    return summary_part, response_part
                else:
                    # Fallback if format wasn't followed
                    logger.warning("Response format not followed correctly, using heuristic parsing")
                    # Try to find a reasonable split point
                    lines = content.split("\n")
                    summary_lines = []
                    response_lines = []
                    in_summary = True

                    for line in lines:
                        if in_summary and (line.strip() == "" or "response" in line.lower()):
                            in_summary = False
                            continue

                        if in_summary:
                            summary_lines.append(line)
                        else:
                            response_lines.append(line)

                    summary = "\n".join(summary_lines).strip()
                    response = "\n".join(response_lines).strip()

                    # If we couldn't split properly, use the whole thing as response
                    # and generate a generic summary
                    if not summary or not response:
                        logger.warning("Could not parse summary and response, using fallback")
                        return "Previous conversation summarized for context management.", content

                    logger.info(f"Parsed summary ({len(summary)} chars) and response using heuristics")
                    return summary, response
            else:
                # Just return the summary with no response
                logger.info(f"Generated summary only ({len(content)} chars)")
                return content, None

        except Exception as e:
            logger.error(f"Error summarizing messages: {e}", exc_info=True)
            # Return a simple fallback summary and no response
            return f"[Failed to generate summary: {str(e)}]", None

    def _needs_compaction(self) -> bool:
        """
        Check if compaction is needed based on the number of exchanges

        Returns:
            Whether compaction is needed
        """
        # Don't compact if we have fewer messages than tail_k
        if len(self.raw_log) <= self.tail_k:
            return False

        # Check if we've reached the configured number of exchanges
        if self.exchange_count >= self.exchanges_before_compact:
            logger.info(f"Compaction needed: {self.exchange_count} exchanges completed (threshold: {self.exchanges_before_compact})")
            return True

        return False

    def _compact(self, current_message: Optional[str] = None) -> Optional[str]:
        """
        Summarize all but the most recent messages and replace them with a summary

        Args:
            current_message: Optional current user message to respond to

        Returns:
            Optional response to current message if provided
        """
        # Keep the most recent tail_k messages (default: 3 messages = 1.5 exchanges)
        if len(self.raw_log) <= self.tail_k:
            return None

        # Split messages - keep the most recent tail_k messages intact
        head = self.raw_log[:-self.tail_k]  # Messages to summarize
        tail = self.raw_log[-self.tail_k:]  # Messages to keep

        logger.info(f"Starting compaction: summarizing {len(head)} messages, keeping {len(tail)} recent messages")
        if current_message:
            logger.info(f"Compaction will also generate a response to: {current_message[:50]}...")

        # Calculate token counts before compaction for logging
        head_tokens = sum(self.token_counter.count_tokens_in_text(content) for _, content in head)
        logger.info(f"Messages to be summarized contain approximately {head_tokens} tokens")

        try:
            # Create a new event loop for the thread
            loop = asyncio.new_event_loop()
            result_container = [None]  # Use a container to store the result

            def run_async_summarization():
                try:
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(self.summarize_async(head, current_message))
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
                response = None
            else:
                # Get the result from the container
                if result_container[0] is None:
                    logger.error("Summarization failed to produce a result")
                    new_summary = f"[Summarization failed for {len(head)} messages]"
                    response = None
                else:
                    # Unpack the result
                    new_summary, response = result_container[0]
                    if response:
                        logger.info(f"Compaction generated a response of {len(response)} chars")

            # Replace the old messages with the summary
            # Create a summary message in the format expected by the LLM
            summary_message = ("system", f"Previous conversation summary: {new_summary}")

            # Update raw log - replace head with summary message and keep tail
            self.raw_log = [summary_message] + tail

            # Calculate token counts after compaction for logging
            summary_tokens = self.token_counter.count_tokens_in_text(new_summary)
            logger.info(f"Summary contains {summary_tokens} tokens (reduced from {head_tokens} tokens)")

            # Update last compaction time
            self.last_compaction_time = time.time()

            logger.info(f"Compaction complete: {len(head)} messages summarized into 1 summary message, {len(tail)} recent messages kept")

            # Return the response if we have one
            return response

        except Exception as e:
            logger.error(f"Error during compaction: {e}", exc_info=True)
            # Don't update state on error
            return None

    @staticmethod
    def _merge_summaries(old: str, new: str) -> str:
        """
        Merge old and new summaries

        Args:
            old: Old summary
            new: New summary

        Returns:
            Merged summary
        """
        # Skip if old summary is the initial message
        if old == "The dialogue has not been summarized yet.":
            return new

        # Add timestamp to new summary
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        return f"{old}\n\n[Update {timestamp}] {new}"
