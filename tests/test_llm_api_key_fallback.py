#!/usr/bin/env python3
"""
Test LLM_API_KEY fallback functionality
"""

import os
import tempfile
from fei.utils.config import Config

def main():
    print("Testing LLM_API_KEY fallback functionality")
    print("-" * 50)
    
    # Create a temporary .env file with only LLM_API_KEY and unset any existing keys
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "BRAVE_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
        tmp.write("LLM_API_KEY=shared_api_key\n")
        tmp_path = tmp.name
    
    try:
        # Initialize config with the custom .env file
        custom_config = Config(env_file=tmp_path)
        
        # Test API keys - should all use the LLM_API_KEY
        providers = ["anthropic", "openai", "groq"]
        for provider in providers:
            key = f"{provider}.api_key"
            value = custom_config.get(key)
            print(f"{key}: {value}")
        
        # Brave isn't an LLM provider, so shouldn't use the fallback
        brave_key = custom_config.get("brave.api_key")
        print(f"brave.api_key: {brave_key}")
        
    finally:
        # Clean up
        os.unlink(tmp_path)
    
    print("-" * 50)
    print("LLM API key fallback test complete")

if __name__ == "__main__":
    main()