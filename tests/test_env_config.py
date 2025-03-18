#!/usr/bin/env python3
"""
Test script for validating .env file reading functionality
"""

import os
from fei.utils.config import get_config

def main():
    print("Testing .env file reading functionality")
    print("-" * 50)
    
    # Get config instance
    config = get_config()
    
    # Test API keys
    providers = ["anthropic", "openai", "groq", "brave"]
    for provider in providers:
        key = f"{provider}.api_key"
        value = config.get(key)
        print(f"{key}: {value}")
    
    # Test custom key
    custom_key = "CUSTOM_API_KEY"
    custom_value = os.environ.get(f"FEI_{custom_key}")
    print(f"FEI_{custom_key}: {custom_value}")
    
    # Test environment variable precedence
    # This should pick up the value from .env file
    brave_key = config.get("brave.api_key")
    print(f"Brave API key (from config): {brave_key}")
    
    # This should directly read the environment variable
    brave_env = os.environ.get("BRAVE_API_KEY")
    print(f"Brave API key (from env): {brave_env}")
    
    print("-" * 50)
    print("Environment variable test complete")

if __name__ == "__main__":
    main()