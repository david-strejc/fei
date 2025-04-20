#!/usr/bin/env python3
"""
Test API key precedence in fei config
"""

import os
import tempfile
from fei.utils.config import Config

def main():
    print("Testing API key precedence")
    print("-" * 50)
    
    # Clear existing environment variables
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    # 1. Create a config file with keys
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as config_tmp:
        config_tmp.write("[anthropic]\n")
        config_tmp.write("api_key=config_anthropic_key\n")
        config_tmp.write("[openai]\n")
        config_tmp.write("api_key=config_openai_key\n")
        config_path = config_tmp.name
    
    # 2. Create a .env file with keys (should override config)
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as env_tmp:
        env_tmp.write("ANTHROPIC_API_KEY=env_anthropic_key\n")
        env_tmp.write("LLM_API_KEY=env_llm_key\n")
        env_path = env_tmp.name
    
    try:
        # Initialize config with both files
        custom_config = Config(config_path=config_path, env_file=env_path)
        
        # Test precedence
        print("1. Precedence Test:")
        print(f"  anthropic.api_key (should be from .env): {custom_config.get('anthropic.api_key')}")
        print(f"  openai.api_key (should be from config): {custom_config.get('openai.api_key')}")
        print(f"  groq.api_key (should be from LLM_API_KEY): {custom_config.get('groq.api_key')}")
        
        # 3. Now set direct environment variables (should override .env)
        os.environ["ANTHROPIC_API_KEY"] = "direct_env_anthropic_key"
        os.environ["OPENAI_API_KEY"] = "direct_env_openai_key"
        
        # Force reload of env file to ensure direct env vars take precedence
        custom_config._load_env_file()
        
        print("\n2. After Setting Direct Environment Variables:")
        print(f"  anthropic.api_key (should be from direct env): {custom_config.get('anthropic.api_key')}")
        print(f"  openai.api_key (should be from direct env): {custom_config.get('openai.api_key')}")
        print(f"  groq.api_key (should be from LLM_API_KEY): {custom_config.get('groq.api_key')}")
        
    finally:
        # Clean up
        os.unlink(config_path)
        os.unlink(env_path)
        
        # Clean up environment
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]:
            if key in os.environ:
                del os.environ[key]
    
    print("-" * 50)
    print("Precedence test complete")

if __name__ == "__main__":
    main()