#!/usr/bin/env python3
"""
Test script for the Prompt Compactor with Gemini models

This script tests the Prompt Compactor with Gemini models in a long conversation
to verify that it correctly compacts older messages while preserving context.
"""

import os
import sys
import asyncio
import time
import argparse
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.core.assistant import Assistant
from fei.utils.config import Config
from fei.utils.logging import get_logger, setup_logging

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Test conversation prompts - designed to create a coherent conversation that builds context
TEST_PROMPTS = [
    "Hello! I'd like to write a Python program that reads a CSV file and calculates some statistics.",
    "I need to read a CSV file with columns for student name, age, and test scores. How do I start?",
    "Great, now I need to calculate the average test score for all students.",
    "How can I find the student with the highest score?",
    "I'd like to group students by age and calculate the average score for each age group.",
    "Can you show me how to create a bar chart of average scores by age using matplotlib?",
    "I want to save these statistics to a new CSV file. How do I do that?",
    "How can I handle missing values in the CSV file?",
    "I'd like to add a new column that grades students based on their scores: A for 90+, B for 80-89, etc.",
    "Can you show me how to sort the data by student name?",
    "How can I filter the data to only include students with scores above 70?",
    "I want to create a scatter plot comparing age vs. score. How do I do that?",
    "Can you help me add a trend line to the scatter plot?",
    "How can I save the plots as PNG files?",
    "I'd like to create a function that does all these operations in one go. Can you help me structure that?",
    "How can I add command-line arguments to my script to specify the input and output files?",
    "Can you show me how to add error handling for file operations?",
    "I want to add logging to my script. What's the best way to do that?",
    "How can I optimize the script for large CSV files?",
    "Can you help me add unit tests for my functions?",
]

async def run_test(
    provider: str = "google",
    model: str = "gemini/gemini-1.5-flash-latest",
    enable_compaction: bool = True,
    tail_k: int = 6,
    target_tokens: int = 6000,
    debug: bool = False,
    api_key: Optional[str] = None,
    pause_seconds: int = 2,
    max_prompts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a test of the Prompt Compactor with a long conversation

    Args:
        provider: Provider to use
        model: Model to use
        enable_compaction: Whether to enable compaction
        tail_k: Number of recent messages to keep verbatim
        target_tokens: Target token count for context
        debug: Whether to enable debug logging
        api_key: API key to use
        pause_seconds: Seconds to pause between prompts
        max_prompts: Maximum number of prompts to use (None for all)

    Returns:
        Test results
    """
    # Set up logging
    if debug:
        setup_logging(level="DEBUG")
    else:
        setup_logging()

    # Create config
    config = Config()

    # Set compaction settings
    config.set("compactor.enabled", enable_compaction)
    config.set("compactor.tail_k", tail_k)
    config.set("compactor.target_tokens", target_tokens)
    config.set("compactor.summarize_provider", provider)
    config.set("compactor.summarize_model", model)

    # Create assistant
    start_time = time.time()
    assistant = Assistant(
        config=config,
        api_key=api_key,
        model=model,
        provider=provider
    )

    print(f"\n{'='*80}")
    print(f"Testing Prompt Compactor with {provider}/{model}")
    print(f"Compaction: {'Enabled' if enable_compaction else 'Disabled'}")
    if enable_compaction:
        print(f"  Tail K: {tail_k}")
        print(f"  Target Tokens: {target_tokens}")
    print(f"{'='*80}\n")

    # Limit prompts if specified
    prompts = TEST_PROMPTS
    if max_prompts is not None:
        prompts = prompts[:max_prompts]

    # Track results
    results = {
        "config": {
            "provider": provider,
            "model": model,
            "enable_compaction": enable_compaction,
            "tail_k": tail_k,
            "target_tokens": target_tokens
        },
        "prompts": [],
        "token_counts": [],
        "response_times": [],
        "compactions": [],
        "total_time": 0
    }

    # Run conversation
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}/{len(prompts)}: {prompt}")

        # Track token count before
        token_count_before = assistant.conversation_manager.get_token_count()
        results["token_counts"].append({"before": token_count_before})

        # Track compaction state before
        if enable_compaction and hasattr(assistant.conversation_manager, "prompt_compactor"):
            compactor = assistant.conversation_manager.prompt_compactor
            compaction_state_before = {
                "raw_log_length": len(compactor.raw_log),
                "summary_length": len(str(compactor.summary))
            }
        else:
            compaction_state_before = None

        # Process prompt
        prompt_start_time = time.time()
        response = await assistant.chat(prompt)
        prompt_time = time.time() - prompt_start_time

        # Print response
        print(f"\nResponse: {response[:200]}..." if len(response) > 200 else f"\nResponse: {response}")

        # Track token count after
        token_count_after = assistant.conversation_manager.get_token_count()
        results["token_counts"][-1]["after"] = token_count_after

        # Track compaction state after
        if enable_compaction and hasattr(assistant.conversation_manager, "prompt_compactor"):
            compactor = assistant.conversation_manager.prompt_compactor
            compaction_state_after = {
                "raw_log_length": len(compactor.raw_log),
                "summary_length": len(str(compactor.summary))
            }

            # Check if compaction occurred
            compaction_occurred = (
                compaction_state_before["raw_log_length"] > compaction_state_after["raw_log_length"] or
                compaction_state_before["summary_length"] < compaction_state_after["summary_length"]
            )

            results["compactions"].append({
                "before": compaction_state_before,
                "after": compaction_state_after,
                "occurred": compaction_occurred
            })
        else:
            results["compactions"].append(None)

        # Track results
        results["prompts"].append({
            "prompt": prompt,
            "response": response,
            "time": prompt_time
        })
        results["response_times"].append(prompt_time)

        # Print stats
        print(f"\nToken count: {token_count_after} (change: {token_count_after - token_count_before:+})")
        print(f"Response time: {prompt_time:.2f}s")

        if enable_compaction and hasattr(assistant.conversation_manager, "prompt_compactor"):
            compactor = assistant.conversation_manager.prompt_compactor
            if results["compactions"][-1]["occurred"]:
                print(f"Compaction occurred: {compaction_state_before['raw_log_length']} -> {compaction_state_after['raw_log_length']} messages")
                print(f"Summary length: {len(str(compactor.summary))} chars")
            else:
                print("No compaction occurred")

        # Pause between prompts
        if i < len(prompts) - 1:
            print(f"\nPausing for {pause_seconds} seconds...")
            await asyncio.sleep(pause_seconds)

    # Calculate total time
    results["total_time"] = time.time() - start_time

    # Print summary
    print(f"\n{'='*80}")
    print(f"Test Summary")
    print(f"{'='*80}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Average response time: {sum(results['response_times']) / len(results['response_times']):.2f}s")
    print(f"Final token count: {results['token_counts'][-1]['after']}")

    if enable_compaction:
        compactions_occurred = sum(1 for c in results["compactions"] if c and c["occurred"])
        print(f"Compactions occurred: {compactions_occurred}/{len(results['compactions'])}")

    return results

async def run_comparison_test(
    provider: str = "google",
    model: str = "gemini/gemini-1.5-flash-latest",
    tail_k: int = 6,
    target_tokens: int = 6000,
    debug: bool = False,
    api_key: Optional[str] = None,
    pause_seconds: int = 2,
    max_prompts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run a comparison test with and without compaction

    Args:
        provider: Provider to use
        model: Model to use
        tail_k: Number of recent messages to keep verbatim
        target_tokens: Target token count for context
        debug: Whether to enable debug logging
        api_key: API key to use
        pause_seconds: Seconds to pause between prompts
        max_prompts: Maximum number of prompts to use (None for all)

    Returns:
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Running comparison test with and without compaction")
    print(f"Provider: {provider}, Model: {model}")
    print(f"{'='*80}\n")

    # Run test with compaction
    print("\nRunning test WITH compaction...")
    with_compaction_results = await run_test(
        provider=provider,
        model=model,
        enable_compaction=True,
        tail_k=tail_k,
        target_tokens=target_tokens,
        debug=debug,
        api_key=api_key,
        pause_seconds=pause_seconds,
        max_prompts=max_prompts
    )

    # Pause between tests
    print(f"\nPausing for 5 seconds between tests...")
    await asyncio.sleep(5)

    # Run test without compaction
    print("\nRunning test WITHOUT compaction...")
    without_compaction_results = await run_test(
        provider=provider,
        model=model,
        enable_compaction=False,
        debug=debug,
        api_key=api_key,
        pause_seconds=pause_seconds,
        max_prompts=max_prompts
    )

    # Compare results
    print(f"\n{'='*80}")
    print(f"Comparison Results")
    print(f"{'='*80}")
    print(f"WITH compaction:")
    print(f"  Total time: {with_compaction_results['total_time']:.2f}s")
    print(f"  Average response time: {sum(with_compaction_results['response_times']) / len(with_compaction_results['response_times']):.2f}s")
    print(f"  Final token count: {with_compaction_results['token_counts'][-1]['after']}")

    print(f"\nWITHOUT compaction:")
    print(f"  Total time: {without_compaction_results['total_time']:.2f}s")
    print(f"  Average response time: {sum(without_compaction_results['response_times']) / len(without_compaction_results['response_times']):.2f}s")
    print(f"  Final token count: {without_compaction_results['token_counts'][-1]['after']}")

    # Calculate differences
    time_diff = without_compaction_results['total_time'] - with_compaction_results['total_time']
    time_diff_pct = (time_diff / without_compaction_results['total_time']) * 100

    avg_time_with = sum(with_compaction_results['response_times']) / len(with_compaction_results['response_times'])
    avg_time_without = sum(without_compaction_results['response_times']) / len(without_compaction_results['response_times'])
    avg_time_diff = avg_time_without - avg_time_with
    avg_time_diff_pct = (avg_time_diff / avg_time_without) * 100

    token_diff = without_compaction_results['token_counts'][-1]['after'] - with_compaction_results['token_counts'][-1]['after']
    token_diff_pct = (token_diff / without_compaction_results['token_counts'][-1]['after']) * 100

    print(f"\nDifferences (without - with):")
    print(f"  Total time: {time_diff:.2f}s ({time_diff_pct:.1f}%)")
    print(f"  Average response time: {avg_time_diff:.2f}s ({avg_time_diff_pct:.1f}%)")
    print(f"  Final token count: {token_diff} ({token_diff_pct:.1f}%)")

    return {
        "with_compaction": with_compaction_results,
        "without_compaction": without_compaction_results,
        "differences": {
            "total_time": time_diff,
            "total_time_pct": time_diff_pct,
            "avg_response_time": avg_time_diff,
            "avg_response_time_pct": avg_time_diff_pct,
            "final_token_count": token_diff,
            "final_token_count_pct": token_diff_pct
        }
    }

async def main():
    """Run the test script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test Prompt Compactor with Gemini")
    parser.add_argument("--provider", default="google", help="Provider to use")
    parser.add_argument("--model", default="gemini/gemini-1.5-flash-latest", help="Model to use")
    parser.add_argument("--api-key", help="API key (defaults to environment variable)")
    parser.add_argument("--no-compaction", action="store_true", help="Disable compaction for testing")
    parser.add_argument("--tail-k", type=int, default=6, help="Number of recent messages to keep verbatim")
    parser.add_argument("--target-tokens", type=int, default=6000, help="Target token count for context")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--pause", type=int, default=2, help="Seconds to pause between prompts")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to use")
    parser.add_argument("--comparison", action="store_true", help="Run comparison test with and without compaction")
    args = parser.parse_args()

    try:
        if args.comparison:
            # Run comparison test
            await run_comparison_test(
                provider=args.provider,
                model=args.model,
                tail_k=args.tail_k,
                target_tokens=args.target_tokens,
                debug=args.debug,
                api_key=args.api_key,
                pause_seconds=args.pause,
                max_prompts=args.max_prompts
            )
        else:
            # Run single test
            await run_test(
                provider=args.provider,
                model=args.model,
                enable_compaction=not args.no_compaction,
                tail_k=args.tail_k,
                target_tokens=args.target_tokens,
                debug=args.debug,
                api_key=args.api_key,
                pause_seconds=args.pause,
                max_prompts=args.max_prompts
            )
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
