#!/usr/bin/env python3
"""
Example of using multiple LLM providers with Fei
"""

import asyncio
import os
import argparse
from pathlib import Path

from fei.core.assistant import Assistant

async def test_provider(provider, model=None):
    """Test a specific provider"""
    print(f"\nTesting provider: {provider} with model: {model or 'default'}")
    
    try:
        # Create assistant without tools
        assistant = Assistant(
            provider=provider,
            model=model
        )
        
        # Test simple query
        query = "What is the capital of France?"
        print(f"Sending query: {query}")
        response = await assistant.chat(query)
        print(f"Response: {response}\n")
        
        return True
    except Exception as e:
        print(f"Error testing {provider}: {e}")
        return False

async def main():
    """Run examples with different providers"""
    parser = argparse.ArgumentParser(description="Test multiple LLM providers")
    parser.add_argument("--provider", help="Provider to test (anthropic, openai, groq)", default=None)
    parser.add_argument("--model", help="Model to use", default=None)
    parser.add_argument("--all", help="Test all providers", action="store_true")
    args = parser.parse_args()
    
    # Load API keys from config/keys
    keys_file = Path("config/keys")
    if keys_file.exists():
        with open(keys_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    value = value.strip('"\'')
                    os.environ[key] = value
                    print(f"Loaded key: {key}")
    
    # If testing all providers
    if args.all:
        providers = ["anthropic", "openai", "groq"]
        results = {}
        
        for provider in providers:
            results[provider] = await test_provider(provider)
        
        # Print summary
        print("\n--- Test Summary ---")
        for provider, success in results.items():
            print(f"{provider}: {'Success' if success else 'Failed'}")
    
    # Otherwise test specific provider
    elif args.provider:
        await test_provider(args.provider, args.model)
    
    # Default to anthropic
    else:
        await test_provider("anthropic")

if __name__ == "__main__":
    asyncio.run(main())