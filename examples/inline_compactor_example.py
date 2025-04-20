#!/usr/bin/env python3
"""
Example demonstrating the Inline Prompt Compactor in Fei

This example shows how to use the Prompt Compactor with the inline response
generation approach, where the LLM both summarizes previous conversation
and responds to the current message in a single call.
"""

import os
import sys
import asyncio
import argparse
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.core.assistant import Assistant
from fei.utils.config import Config
from fei.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


async def main():
    """Run the example"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fei Inline Prompt Compactor Example")
    parser.add_argument("--provider", default="anthropic", help="Provider to use")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--enable-compaction", action="store_true", help="Enable prompt compaction")
    parser.add_argument("--tail-k", type=int, default=3, help="Number of recent messages to keep verbatim")
    parser.add_argument("--exchanges-before-compact", type=int, default=3, help="Number of exchanges before compaction")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Set up logging
    if args.debug:
        setup_logging(level="DEBUG")
    else:
        setup_logging()

    # Create config
    config = Config()

    # Set compaction settings
    config.set("compactor.enabled", args.enable_compaction)
    config.set("compactor.tail_k", args.tail_k)
    config.set("compactor.exchanges_before_compact", args.exchanges_before_compact)

    # Ensure we use the same model for summarization as the main conversation
    config.set("compactor.summarize_provider", args.provider)
    if args.model:
        config.set("compactor.summarize_model", args.model)

    # Create assistant
    assistant = Assistant(
        config=config,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider
    )

    print(f"Fei Inline Prompt Compactor Example")
    print(f"Provider: {args.provider}")
    print(f"Model: {assistant.provider_client.model}")
    print(f"Compaction: {'Enabled' if args.enable_compaction else 'Disabled'}")
    if args.enable_compaction:
        print(f"  Tail K: {args.tail_k}")
        print(f"  Exchanges Before Compaction: {args.exchanges_before_compact}")
    print()

    # Sample conversation prompts to demonstrate compaction
    sample_prompts = [
        "Hello! I'd like to write a Python program that reads a CSV file and calculates some statistics.",
        "I need to read a CSV file with columns for student name, age, and test scores. How do I start?",
        "Great, now I need to calculate the average test score for all students.",
        "How can I find the student with the highest score?",
        "I'd like to group students by age and calculate the average score for each age group.",
        "Can you show me how to create a bar chart of average scores by age using matplotlib?",
        "I want to save these statistics to a new CSV file. How do I do that?",
        "How can I handle missing values in the CSV file?",
    ]

    # Run chat loop with sample prompts
    try:
        for i, prompt in enumerate(sample_prompts):
            print(f"\nPrompt {i+1}/{len(sample_prompts)}: {prompt}")

            # Get token count before
            token_count_before = assistant.conversation_manager.get_token_count()

            # Process prompt
            start_time = asyncio.get_event_loop().time()
            response = await assistant.chat(prompt)
            elapsed_time = asyncio.get_event_loop().time() - start_time

            # Print response
            print(f"\nResponse: {response[:200]}..." if len(response) > 200 else f"\nResponse: {response}")

            # Get token count after
            token_count_after = assistant.conversation_manager.get_token_count()

            # Print stats
            print(f"\nToken count: {token_count_after} (change: {token_count_after - token_count_before:+})")
            print(f"Response time: {elapsed_time:.2f}s")

            # Pause between prompts
            if i < len(sample_prompts) - 1:
                print(f"\nPress Enter to continue to next prompt...")
                input()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
