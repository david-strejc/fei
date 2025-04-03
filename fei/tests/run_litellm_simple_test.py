#!/usr/bin/env python3
"""
Simple test script for LiteLLM integration in Fei

This script tests basic LiteLLM functionality without relying on code tools.
"""

import os
import asyncio
import argparse
from pathlib import Path

from fei.core.assistant import Assistant
from fei.utils.config import Config

def setup_keys():
    """Setup API keys from keys file"""
    keys_file = Path("config/keys")
    if keys_file.exists():
        with open(keys_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value
                    print(f"Loaded key: {key}")

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
        print(f"Sending test query...")
        response = await assistant.chat("What is the capital of France?")
        print(f"Response: {response}\n")
        
        return True
    except Exception as e:
        print(f"Error testing {provider}: {e}")
        return False

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test LiteLLM integration")
    parser.add_argument("--provider", help="Provider to test (anthropic, openai, groq)", default=None)
    parser.add_argument("--model", help="Model to use", default=None)
    parser.add_argument("--all", help="Test all providers", action="store_true")
    args = parser.parse_args()
    
    # Setup keys
    setup_keys()
    
    # Create config directory if it doesn't exist
    config_path = os.path.expanduser("~/.fei.ini")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
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