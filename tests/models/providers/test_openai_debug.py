#!/usr/bin/env python3
"""
Debug script for OpenAI models using FEI's LiteLLM provider logic.

This script tests the accessibility of OpenAI models and debugs API key handling.
"""

import os
import sys
import litellm

# Ensure the 'fei' package directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from fei.providers.litellm_provider import LiteLLMProvider
except ImportError:
    print("Error: Could not import LiteLLMProvider from fei.providers.litellm_provider.")
    print("Please ensure the script is run from the project root directory.")
    print("and the fei package is correctly structured.")
    sys.exit(1)

# Get the API key from environment
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or environment.")
    sys.exit(1)

def test_openai_model(model_name: str):
    """Tests a single OpenAI model using FEI's LiteLLM provider logic with debugging."""
    print(f"\n--- Testing model: {model_name} ---")
    provider = None
    try:
        # Instantiate FEI's LiteLLM provider
        provider = LiteLLMProvider()

        # Debug: Print provider state before setup
        print("Provider state before setup:")
        print(f"  provider: {provider.provider}")
        print(f"  api_key: {'Set (first 10 chars): ' + provider.api_key[:10] + '...' if provider.api_key else 'Not set'}")
        print(f"  model: {provider.model}")

        # Set provider to 'openai'
        provider.provider = 'openai'

        # Debug: Print environment variables
        print("Environment variables:")
        print(f"  OPENAI_API_KEY: {'Set (first 10 chars): ' + os.environ.get('OPENAI_API_KEY', '')[:10] + '...' if 'OPENAI_API_KEY' in os.environ else 'Not set'}")

        # Setup the provider with explicit API key
        provider.setup(model=model_name, api_key=OPENAI_API_KEY)

        # Debug: Print provider state after setup
        print("Provider state after setup:")
        print(f"  provider: {provider.provider}")
        print(f"  api_key: {'Set (first 10 chars): ' + provider.api_key[:10] + '...' if provider.api_key else 'Not set'}")
        print(f"  model: {provider.model}")

        # Prepare a simple message
        messages = [{"role": "user", "content": "Briefly explain what model you are."}]

        # Call LiteLLM directly with the model and API key
        print("Sending request to OpenAI API...")
        model_with_prefix = f"openai/{provider.model}"

        response = litellm.completion(
            model=model_with_prefix,
            messages=messages,
            temperature=0.1,
            api_key=provider.api_key  # Use the provider's API key
        )

        # Extract content from the response
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = ""

        if content:
            print(f"Success! Received response:")
            print("-" * 20)
            print(content[:100] + "..." if len(content) > 100 else content)
            print("-" * 20)
            return True
        else:
            print("Failed: Received empty content in response.")
            print("Full response object:", response)
            return False

    except Exception as e:
        print(f"Failed: An error occurred.")
        print(f"Error details: {type(e).__name__}: {e}")
        if provider:
            print(f"Provider state: provider='{provider.provider}', model='{provider.model}', api_key_set={'Yes' if provider.api_key else 'No'}")
        return False

def main():
    """Main entry point"""
    print("=====================================================================")
    print("Starting OpenAI model debug test...")
    print("=====================================================================")

    # Test models
    models = ["gpt-4o", "gpt-4-turbo"]
    results = {}

    for model in models:
        results[model] = test_openai_model(model)

    print("\n==================== Test Summary ====================")
    all_successful = True
    for model, success in results.items():
        status = "✅ Accessible" if success else "❌ NOT Accessible"
        print(f"- {model}: {status}")
        if not success:
            all_successful = False
    print("======================================================")

    if all_successful:
        print("\n🎉 All specified OpenAI models were successfully accessed.")
    else:
        print("\n⚠️ Some OpenAI models were not accessible. Please review the logs above.")
    print("======================================================")

if __name__ == "__main__":
    main()
