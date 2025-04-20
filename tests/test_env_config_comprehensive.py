#!/usr/bin/env python3
"""
Comprehensive test script for .env file functionality
"""

import os
import tempfile
from pathlib import Path
from fei.utils.config import Config, get_config

def test_default_env():
    """Test with the default .env file"""
    print("\n1. Testing with default .env file:")
    print("-" * 50)
    
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

def test_custom_env_file():
    """Test with a custom env file"""
    print("\n2. Testing with custom .env file:")
    print("-" * 50)
    
    # Create a temporary .env file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
        tmp.write("ANTHROPIC_API_KEY=custom_anthropic_key\n")
        tmp.write("OPENAI_API_KEY=custom_openai_key\n")
        tmp.write("FEI_TEST_OPTION=custom_test_value\n")
        tmp_path = tmp.name
    
    try:
        # Initialize config with the custom .env file
        custom_config = Config(env_file=tmp_path)
        
        # Test API keys
        print(f"anthropic.api_key: {custom_config.get('anthropic.api_key')}")
        print(f"openai.api_key: {custom_config.get('openai.api_key')}")
        print(f"groq.api_key: {custom_config.get('groq.api_key')}")  # Should be None
        
        # Test custom option
        env_value = os.environ.get("FEI_TEST_OPTION")
        print(f"FEI_TEST_OPTION: {env_value}")
    finally:
        # Clean up
        os.unlink(tmp_path)

def test_fallback_to_config():
    """Test fallback to config file when key not in .env"""
    print("\n3. Testing fallback to config file:")
    print("-" * 50)
    
    # Create a temporary config file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp:
        tmp.write("[test]\n")
        tmp.write("fallback_option=fallback_value\n")
        tmp_path = tmp.name
    
    try:
        # Create a temporary .env file without the test key
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as env_tmp:
            env_tmp.write("ANTHROPIC_API_KEY=env_anthropic_key\n")
            env_path = env_tmp.name
        
        # Initialize config with both files
        custom_config = Config(config_path=tmp_path, env_file=env_path)
        
        # Should get value from .env
        print(f"anthropic.api_key: {custom_config.get('anthropic.api_key')}")
        
        # Should get value from config file
        print(f"test.fallback_option: {custom_config.get('test.fallback_option')}")
        
        # Should get default value
        print(f"Missing key with default: {custom_config.get('missing.key', 'default_value')}")
    finally:
        # Clean up
        os.unlink(tmp_path)
        os.unlink(env_path)

def main():
    print("COMPREHENSIVE ENV FILE TESTING")
    
    # Run all tests
    test_default_env()
    test_custom_env_file()
    test_fallback_to_config()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    main()