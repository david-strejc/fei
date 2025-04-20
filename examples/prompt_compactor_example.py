#!/usr/bin/env python3
"""
Example demonstrating the Prompt Compactor in Fei

This example shows how to use the Prompt Compactor to manage conversation
context efficiently.
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
    parser = argparse.ArgumentParser(description="Fei Prompt Compactor Example")
    parser.add_argument("--provider", default="anthropic", help="Provider to use")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--api-key", help="API key")
    parser.add_argument("--enable-compaction", action="store_true", help="Enable prompt compaction")
    parser.add_argument("--tail-k", type=int, default=6, help="Number of recent messages to keep verbatim")
    parser.add_argument("--target-tokens", type=int, default=6000, help="Target token count for context")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(debug=args.debug)
    
    # Create config
    config = Config()
    
    # Set compaction settings
    config.set("compactor.enabled", args.enable_compaction)
    config.set("compactor.tail_k", args.tail_k)
    config.set("compactor.target_tokens", args.target_tokens)
    
    # Create assistant
    assistant = Assistant(
        config=config,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider
    )
    
    print(f"Fei Prompt Compactor Example")
    print(f"Provider: {args.provider}")
    print(f"Model: {assistant.provider_client.model}")
    print(f"Compaction: {'Enabled' if args.enable_compaction else 'Disabled'}")
    if args.enable_compaction:
        print(f"  Tail K: {args.tail_k}")
        print(f"  Target Tokens: {args.target_tokens}")
    print()
    
    # Run chat loop
    try:
        while True:
            # Get user input
            user_input = input("You: ")
            
            # Exit on empty input
            if not user_input.strip():
                break
            
            # Process user input
            response = await assistant.chat(user_input)
            
            # Print response
            print(f"Assistant: {response}")
            
            # Print token count
            token_count = assistant.conversation_manager.get_token_count()
            print(f"Current token count: {token_count}")
            print()
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
