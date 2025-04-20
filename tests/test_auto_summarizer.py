#!/usr/bin/env python3
"""
Test script for the AutoSummarizer.

This script tests the AutoSummarizer by simulating a conversation with large messages
and verifying that summarization occurs when the token threshold is reached.
"""

import os
import sys
import asyncio
import argparse
import time
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.core.assistant import Assistant
from fei.utils.config import Config
from fei.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def generate_large_message(size_tokens: int = 5000) -> str:
    """
    Generate a large message with approximately the specified number of tokens.

    Args:
        size_tokens: Approximate number of tokens to generate

    Returns:
        A large message
    """
    # Assuming 4 characters per token on average
    chars_per_token = 4
    total_chars = size_tokens * chars_per_token

    # Generate a message with the specified number of characters
    # Use a more varied text to make summarization more meaningful
    paragraphs = [
        "This is a test message to demonstrate auto summarization. The auto summarizer is designed to monitor the token count of a conversation and trigger summarization when it exceeds a configurable threshold. This helps manage the context window size and reduce API costs.",
        "The implementation includes a standalone summarization module with different styles (bullet, narrative, technical). The summarization logic is decoupled from the main conversation flow, making it easy to customize and extend.",
        "When summarization is triggered, older messages are replaced with a concise summary, while the most recent messages are kept intact. This significantly reduces the token count while preserving important context.",
        "The auto summarizer is particularly useful for models with smaller context windows or when trying to reduce API costs by minimizing token usage. It's also helpful for long-running conversations that would otherwise exceed the model's context window.",
        "The implementation includes proper error handling and logging, making it robust and reliable. It's also designed to be configurable, allowing users to customize the token threshold, the number of recent messages to keep, and the summarization style.",
        "Testing the auto summarizer involves generating large messages to trigger summarization and verifying that the token count is reduced while preserving important information. This helps ensure that the implementation works as expected in real-world scenarios."
    ]

    # Repeat paragraphs until we reach the desired token count
    result = ""
    while len(result) < total_chars:
        for paragraph in paragraphs:
            result += paragraph + "\n\n"
            if len(result) >= total_chars:
                break

    return result[:total_chars]


async def run_test(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    token_threshold: int = 30000,
    message_size_tokens: int = 5000,
    num_messages: int = 8,
    comparison: bool = False,
    debug: bool = False
) -> None:
    """
    Run the auto summarizer test.

    Args:
        provider: Provider to use
        model: Model to use
        api_key: API key
        token_threshold: Token threshold for auto summarization
        message_size_tokens: Size of each message in tokens
        num_messages: Number of messages to send
        comparison: Whether to run a comparison test with and without auto summarization
        debug: Whether to enable debug logging
    """
    if comparison:
        print("\n================================================================================")
        print(f"Running comparison test with and without auto summarization")
        print(f"Provider: {provider}")
        print("================================================================================\n")

        # Run test with auto summarization
        print("\nRunning test WITH auto summarization...")
        await run_single_test(
            provider=provider,
            model=model,
            api_key=api_key,
            token_threshold=token_threshold,
            message_size_tokens=message_size_tokens,
            num_messages=num_messages,
            enable_auto_summarization=True,
            debug=debug
        )

        # Run test without auto summarization
        print("\nRunning test WITHOUT auto summarization...")
        await run_single_test(
            provider=provider,
            model=model,
            api_key=api_key,
            token_threshold=token_threshold,
            message_size_tokens=message_size_tokens,
            num_messages=num_messages,
            enable_auto_summarization=False,
            debug=debug
        )
    else:
        # Run a single test
        await run_single_test(
            provider=provider,
            model=model,
            api_key=api_key,
            token_threshold=token_threshold,
            message_size_tokens=message_size_tokens,
            num_messages=num_messages,
            enable_auto_summarization=True,
            debug=debug
        )


async def run_single_test(
    provider: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    token_threshold: int = 30000,
    message_size_tokens: int = 5000,
    num_messages: int = 8,
    enable_auto_summarization: bool = True,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Run a single auto summarizer test.

    Args:
        provider: Provider to use
        model: Model to use
        api_key: API key
        token_threshold: Token threshold for auto summarization
        message_size_tokens: Size of each message in tokens
        num_messages: Number of messages to send
        enable_auto_summarization: Whether to enable auto summarization
        debug: Whether to enable debug logging

    Returns:
        Dictionary with test results
    """
    # Create config
    config = Config()

    # Set auto summarization settings
    config.set("auto_summarizer.enabled", enable_auto_summarization)
    config.set("auto_summarizer.token_threshold", token_threshold)
    config.set("auto_summarizer.tail_k", 1)  # Keep only the most recent message for testing

    # Ensure we use the same model for summarization as the main conversation
    config.set("auto_summarizer.summarize_provider", provider)
    if model:
        config.set("auto_summarizer.summarize_model", model)

    # Create assistant
    assistant = Assistant(
        config=config,
        api_key=api_key,
        model=model,
        provider=provider
    )

    print("\n================================================================================")
    print(f"Testing Auto Summarizer with {provider}/{assistant.provider_client.model}")
    print(f"Auto Summarization: {'Enabled' if enable_auto_summarization else 'Disabled'}")
    if enable_auto_summarization:
        print(f"  Token Threshold: {token_threshold}")
        print(f"  Message Size: ~{message_size_tokens} tokens")
    print("================================================================================\n")

    # Track metrics
    start_time = time.time()
    response_times = []
    token_counts = []
    summarizations = 0

    # Generate large messages
    large_messages = []
    for i in range(num_messages):
        base_message = f"Message {i+1}/{num_messages}: "
        large_message = base_message + generate_large_message(message_size_tokens - len(base_message))
        large_messages.append(large_message)

    # Run chat loop with large messages
    try:
        for i, prompt in enumerate(large_messages):
            print(f"\nPrompt {i+1}/{len(large_messages)}: {prompt[:50]}...")

            # Get token count before
            token_count_before = assistant.conversation_manager.get_token_count()
            token_counts.append(token_count_before)

            # Process prompt
            prompt_start_time = asyncio.get_event_loop().time()
            response = await assistant.chat(prompt)
            prompt_elapsed_time = asyncio.get_event_loop().time() - prompt_start_time
            response_times.append(prompt_elapsed_time)

            # Print response
            print(f"\nResponse: {response[:200]}..." if len(response) > 200 else f"\nResponse: {response}")

            # Get token count after
            token_count_after = assistant.conversation_manager.get_token_count()

            # Check if summarization occurred
            if token_count_after < token_count_before:
                summarizations += 1
                print(f"\n*** Summarization occurred! ***")
                print(f"Token count before: {token_count_before}")
                print(f"Token count after: {token_count_after}")
                print(f"Reduction: {token_count_before - token_count_after} tokens ({(token_count_before - token_count_after) / token_count_before * 100:.1f}%)")

            # Print stats
            print(f"\nToken count: {token_count_after} (change: {token_count_after - token_count_before:+})")
            print(f"Response time: {prompt_elapsed_time:.2f}s")

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

    # Calculate final metrics
    total_time = time.time() - start_time
    avg_response_time = sum(response_times) / len(response_times)
    final_token_count = assistant.conversation_manager.get_token_count()
    max_token_count = max(token_counts + [final_token_count])

    # Print summary
    print("\n================================================================================")
    print("Test Summary")
    print("================================================================================")
    print(f"Total prompts: {len(large_messages)}")
    print(f"Summarizations occurred: {summarizations}")
    print(f"Average response time: {avg_response_time:.2f}s")
    print(f"Final token count: {final_token_count}")
    print(f"Maximum token count: {max_token_count}")
    print()

    # Return results
    return {
        "total_time": total_time,
        "avg_response_time": avg_response_time,
        "final_token_count": final_token_count,
        "max_token_count": max_token_count,
        "summarizations": summarizations
    }


async def main():
    """Run the test"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fei Auto Summarizer Test")
    parser.add_argument("--provider", default="anthropic", help="Provider to use")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--token-threshold", type=int, default=30000, help="Token threshold for auto summarization")
    parser.add_argument("--message-size", type=int, default=5000, help="Size of each message in tokens")
    parser.add_argument("--num-messages", type=int, default=8, help="Number of messages to send")
    parser.add_argument("--comparison", action="store_true", help="Run comparison test with and without auto summarization")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging
    if args.debug:
        setup_logging(level="DEBUG")
    else:
        setup_logging()

    # Run test
    await run_test(
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        token_threshold=args.token_threshold,
        message_size_tokens=args.message_size,
        num_messages=args.num_messages,
        comparison=args.comparison,
        debug=args.debug
    )


if __name__ == "__main__":
    asyncio.run(main())
