#!/usr/bin/env python3
"""
Test environment variable preservation during .env loading
"""

import os
from fei.utils.config import Config

def main():
    print("Testing environment variable preservation")
    print("-" * 50)
    
    # Set direct environment variables
    os.environ["ANTHROPIC_API_KEY"] = "direct_anthropic_key"
    os.environ["OPENAI_API_KEY"] = "direct_openai_key"
    
    # Create a temporary .env file with different values
    with open("/tmp/.env", "w") as f:
        f.write("ANTHROPIC_API_KEY=env_file_anthropic_key\n")
        f.write("GROQ_API_KEY=env_file_groq_key\n")
    
    # Initialize config with the .env file
    # Direct environment variables should take precedence
    config = Config(env_file="/tmp/.env")
    
    # Check values
    print(f"ANTHROPIC_API_KEY (direct): {config.get('anthropic.api_key')}")
    print(f"OPENAI_API_KEY (direct): {config.get('openai.api_key')}")
    print(f"GROQ_API_KEY (from .env): {config.get('groq.api_key')}")
    
    # Set GROQ_API_KEY to null to ensure we get the value from .env
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    
    # Reinitialize to pick up the .env value for GROQ
    config = Config(env_file="/tmp/.env")
    print(f"GROQ_API_KEY (after clearing): {config.get('groq.api_key')}")
    
    # Clean up
    os.remove("/tmp/.env")
    del os.environ["ANTHROPIC_API_KEY"]
    del os.environ["OPENAI_API_KEY"]
    
    print("-" * 50)
    print("Environment variable preservation test completed")

if __name__ == "__main__":
    main()