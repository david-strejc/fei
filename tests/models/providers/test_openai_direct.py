#!/usr/bin/env python3
"""
Direct test script for OpenAI models using LiteLLM.

This script directly tests OpenAI models with the provided API key.
"""

import os
import sys
import litellm

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

# Define OpenAI models to test
MODELS = ["gpt-4o", "gpt-4-turbo"]

def test_openai_model(model_name):
    """Test a single OpenAI model using LiteLLM directly."""
    print(f"\n--- Testing model: {model_name} ---")

    try:
        # Prepare a simple message
        messages = [{"role": "user", "content": "Briefly explain what model you are."}]

        # Call LiteLLM directly
        print("Sending request to OpenAI API...")
        model_with_prefix = f"openai/{model_name}"

        response = litellm.completion(
            model=model_with_prefix,
            messages=messages,
            temperature=0.1,
            api_key=OPENAI_API_KEY
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
        return False

def main():
    """Main entry point"""
    print("=====================================================================")
    print("Starting direct OpenAI model test...")
    print("=====================================================================")

    results = {}

    for model in MODELS:
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
