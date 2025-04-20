#!/usr/bin/env python3
"""
Thorough test for the Inline Prompt Compactor

This script tests the Prompt Compactor with a long conversation to verify
that it correctly compacts older messages while preserving context and
generating responses inline during compaction.
"""

import os
import sys
import asyncio
import time
import argparse
import json
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

class CompactionTracker:
    """Track compaction events and statistics"""

    def __init__(self):
        self.compactions = []
        self.responses = []
        self.token_counts = []
        self.response_times = []

    def add_compaction(self, event_data: Dict[str, Any]):
        """Add a compaction event"""
        self.compactions.append(event_data)

    def add_response(self, prompt: str, response: str, tokens_before: int, tokens_after: int, response_time: float):
        """Add a response event"""
        self.responses.append({
            "prompt": prompt,
            "response": response,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "token_change": tokens_after - tokens_before,
            "response_time": response_time
        })
        self.token_counts.append(tokens_after)
        self.response_times.append(response_time)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            "compaction_count": len(self.compactions),
            "response_count": len(self.responses),
            "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            "final_token_count": self.token_counts[-1] if self.token_counts else 0,
            "max_token_count": max(self.token_counts) if self.token_counts else 0,
            "compactions": self.compactions
        }

    def print_summary(self):
        """Print summary statistics"""
        summary = self.get_summary()

        print(f"\n{'='*80}")
        print(f"Test Summary")
        print(f"{'='*80}")
        print(f"Total prompts: {len(self.responses)}")
        print(f"Compactions occurred: {summary['compaction_count']}")
        print(f"Average response time: {summary['avg_response_time']:.2f}s")
        print(f"Final token count: {summary['final_token_count']}")
        print(f"Maximum token count: {summary['max_token_count']}")

        if summary['compaction_count'] > 0:
            print(f"\nCompaction events:")
            for i, event in enumerate(self.compactions):
                print(f"  {i+1}. After prompt {event['prompt_index']+1}: {event['messages_before']} → {event['messages_after']} messages")
                print(f"     Summary length: {event['summary_length']} chars")
                print(f"     Response generated during compaction: {'Yes' if event['response_generated'] else 'No'}")

class ConversationMonitor:
    """Monitor conversation and compaction events"""

    def __init__(self, assistant: Assistant):
        self.assistant = assistant
        self.tracker = CompactionTracker()
        self.conversation_manager = assistant.conversation_manager
        self.prompt_compactor = getattr(self.conversation_manager, "prompt_compactor", None)

        # Patch the _compact method to track compaction events
        if self.prompt_compactor:
            self._original_compact = self.prompt_compactor._compact
            self.prompt_compactor._compact = self._patched_compact

    def _patched_compact(self, current_message: Optional[str] = None) -> Optional[str]:
        """Patched _compact method to track compaction events"""
        # Get state before compaction
        messages_before = len(self.prompt_compactor.raw_log)
        summary_length_before = len(str(self.prompt_compactor.summary))

        # Call original method
        response = self._original_compact(current_message)

        # Get state after compaction
        messages_after = len(self.prompt_compactor.raw_log)
        summary_length_after = len(str(self.prompt_compactor.summary))

        # Track compaction event
        self.tracker.add_compaction({
            "prompt_index": len(self.tracker.responses),
            "messages_before": messages_before,
            "messages_after": messages_after,
            "summary_length": summary_length_after,
            "summary_length_change": summary_length_after - summary_length_before,
            "response_generated": response is not None
        })

        return response

    def reset(self):
        """Reset the tracker"""
        self.tracker = CompactionTracker()

    def restore(self):
        """Restore original methods"""
        if self.prompt_compactor and hasattr(self, "_original_compact"):
            self.prompt_compactor._compact = self._original_compact

async def run_test(
    provider: str = "anthropic",
    model: Optional[str] = None,
    enable_compaction: bool = True,
    tail_k: int = 3,
    target_tokens: int = 3000,
    debug: bool = False,
    api_key: Optional[str] = None,
    pause_seconds: int = 0,
    max_prompts: Optional[int] = None,
    save_results: bool = False
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
        save_results: Whether to save results to a file

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
    config.set("compactor.exchanges_before_compact", 3)  # Compact after 3 full exchanges

    # Ensure we use the same model for summarization as the main conversation
    config.set("compactor.summarize_provider", provider)
    if model:
        config.set("compactor.summarize_model", model)

    # Create assistant
    start_time = time.time()
    assistant = Assistant(
        config=config,
        api_key=api_key,
        model=model,
        provider=provider
    )

    # Set up monitoring
    monitor = ConversationMonitor(assistant)

    print(f"\n{'='*80}")
    print(f"Testing Inline Prompt Compactor with {provider}/{assistant.model}")
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
            "model": assistant.model,
            "enable_compaction": enable_compaction,
            "tail_k": tail_k,
            "target_tokens": target_tokens
        },
        "prompts": [],
        "total_time": 0
    }

    # Run conversation
    try:
        for i, prompt in enumerate(prompts):
            print(f"\nPrompt {i+1}/{len(prompts)}: {prompt}")

            # Track token count before
            token_count_before = assistant.conversation_manager.get_token_count()

            # Process prompt
            prompt_start_time = time.time()
            response = await assistant.chat(prompt)
            prompt_time = time.time() - prompt_start_time

            # Track token count after
            token_count_after = assistant.conversation_manager.get_token_count()

            # Track response
            monitor.tracker.add_response(
                prompt=prompt,
                response=response,
                tokens_before=token_count_before,
                tokens_after=token_count_after,
                response_time=prompt_time
            )

            # Print response
            print(f"\nResponse: {response[:200]}..." if len(response) > 200 else f"\nResponse: {response}")

            # Print stats
            print(f"\nToken count: {token_count_after} (change: {token_count_after - token_count_before:+})")
            print(f"Response time: {prompt_time:.2f}s")

            # Pause between prompts
            if i < len(prompts) - 1 and pause_seconds > 0:
                print(f"\nPausing for {pause_seconds} seconds...")
                await asyncio.sleep(pause_seconds)

    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        raise
    finally:
        # Calculate total time
        results["total_time"] = time.time() - start_time

        # Get summary from tracker
        summary = monitor.tracker.get_summary()
        results.update(summary)

        # Print summary
        monitor.tracker.print_summary()

        # Restore original methods
        monitor.restore()

        # Save results if requested
        if save_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"compactor_test_{provider}_{timestamp}.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")

    return results

async def run_comparison_test(
    provider: str = "anthropic",
    model: Optional[str] = None,
    tail_k: int = 3,
    target_tokens: int = 3000,
    debug: bool = False,
    api_key: Optional[str] = None,
    pause_seconds: int = 0,
    max_prompts: Optional[int] = None,
    save_results: bool = False
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
        save_results: Whether to save results to a file

    Returns:
        Comparison results
    """
    print(f"\n{'='*80}")
    print(f"Running comparison test with and without compaction")
    print(f"Provider: {provider}")
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
        max_prompts=max_prompts,
        save_results=save_results
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
        max_prompts=max_prompts,
        save_results=save_results
    )

    # Compare results
    print(f"\n{'='*80}")
    print(f"Comparison Results")
    print(f"{'='*80}")
    print(f"WITH compaction:")
    print(f"  Total time: {with_compaction_results['total_time']:.2f}s")
    print(f"  Average response time: {with_compaction_results['avg_response_time']:.2f}s")
    print(f"  Final token count: {with_compaction_results['final_token_count']}")
    print(f"  Compactions occurred: {with_compaction_results['compaction_count']}")

    print(f"\nWITHOUT compaction:")
    print(f"  Total time: {without_compaction_results['total_time']:.2f}s")
    print(f"  Average response time: {without_compaction_results['avg_response_time']:.2f}s")
    print(f"  Final token count: {without_compaction_results['final_token_count']}")

    # Calculate differences
    time_diff = without_compaction_results['total_time'] - with_compaction_results['total_time']
    time_diff_pct = (time_diff / without_compaction_results['total_time']) * 100

    avg_time_with = with_compaction_results['avg_response_time']
    avg_time_without = without_compaction_results['avg_response_time']
    avg_time_diff = avg_time_without - avg_time_with
    avg_time_diff_pct = (avg_time_diff / avg_time_without) * 100

    token_diff = without_compaction_results['final_token_count'] - with_compaction_results['final_token_count']
    token_diff_pct = (token_diff / without_compaction_results['final_token_count']) * 100

    print(f"\nDifferences (without - with):")
    print(f"  Total time: {time_diff:.2f}s ({time_diff_pct:.1f}%)")
    print(f"  Average response time: {avg_time_diff:.2f}s ({avg_time_diff_pct:.1f}%)")
    print(f"  Final token count: {token_diff} ({token_diff_pct:.1f}%)")

    # Save comparison results if requested
    if save_results:
        comparison_results = {
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

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"compactor_comparison_{provider}_{timestamp}.json"
        with open(filename, "w") as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison results saved to {filename}")

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
    parser = argparse.ArgumentParser(description="Test Inline Prompt Compactor")
    parser.add_argument("--provider", default="anthropic", help="Provider to use")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--api-key", help="API key (defaults to environment variable)")
    parser.add_argument("--no-compaction", action="store_true", help="Disable compaction for testing")
    parser.add_argument("--tail-k", type=int, default=3, help="Number of recent messages to keep verbatim")
    parser.add_argument("--target-tokens", type=int, default=3000, help="Target token count for context")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--pause", type=int, default=0, help="Seconds to pause between prompts")
    parser.add_argument("--max-prompts", type=int, help="Maximum number of prompts to use")
    parser.add_argument("--comparison", action="store_true", help="Run comparison test with and without compaction")
    parser.add_argument("--save-results", action="store_true", help="Save test results to a file")
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
                max_prompts=args.max_prompts,
                save_results=args.save_results
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
                max_prompts=args.max_prompts,
                save_results=args.save_results
            )
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
