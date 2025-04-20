#!/usr/bin/env python3
"""
Test script for Gemini models using FEI's LiteLLM provider logic.

This script tests the accessibility of specified Gemini models by sending a simple
request to each model and checking if a valid response is received.

It requires the GEMINI_API_KEY environment variable to be set. If only GOOGLE_API_KEY
is set, the script will copy its value to GEMINI_API_KEY.

Usage:
    python test_gemini_models.py
"""

import os
import sys
import litellm

# Ensure the 'fei' package directory is in the Python path
# This assumes the script is run from the project root (/home/david/Work/Programming/fei_2.0)
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from fei.providers.litellm_provider import LiteLLMProvider
except ImportError:
    print("Error: Could not import LiteLLMProvider from fei.providers.litellm_provider.")
    print("Please ensure the script is run from the project root directory '/home/david/Work/Programming/fei_2.0'")
    print("and the fei package is correctly structured.")
    sys.exit(1)


def test_gemini_model(model_name: str):
    """Tests a single Gemini model using FEI's LiteLLM provider logic."""
    print(f"\n--- Testing model: {model_name} ---")
    provider = None # Initialize provider to None
    try:
        # Instantiate FEI's LiteLLM provider
        provider = LiteLLMProvider()

        # First set the provider to 'gemini' before calling setup
        # LiteLLM expects 'gemini' as the provider for Google models
        provider.provider = 'gemini'

        # Add 'gemini' to the provider_key_map if it's not there
        if 'gemini' not in provider.provider_key_map:
            provider.provider_key_map['gemini'] = 'GEMINI_API_KEY'

        # Set the GEMINI_API_KEY environment variable if GOOGLE_API_KEY is set
        if 'GOOGLE_API_KEY' in os.environ and 'GEMINI_API_KEY' not in os.environ:
            os.environ['GEMINI_API_KEY'] = os.environ['GOOGLE_API_KEY']
            print(f"Set GEMINI_API_KEY from GOOGLE_API_KEY: {os.environ['GEMINI_API_KEY'][:10]}...")

        # Setup the provider - it will try to find GEMINI_API_KEY in env
        # Pass the specific model name to test
        # The setup method internally calls _setup_api_key and _setup_model
        # We need to override the provider again after setup since it gets reset
        provider.setup(model=model_name, api_key=os.environ.get('GEMINI_API_KEY'))
        provider.provider = 'gemini'

        print(f"Provider set to: {provider.provider}")
        print(f"Model set to: {provider.model}")

        # Check if API key was loaded by the setup method
        # The setup method raises ValueError if key is not found,
        # but we add an explicit check here for clarity before the call.
        if not provider.api_key:
             # This part should ideally not be reached if setup worked correctly
             # and GEMINI_API_KEY is set, as setup would raise ValueError.
             # If setup logic changes, this might become relevant.
             print("Error: GEMINI_API_KEY was not loaded by the provider setup.")
             print("Ensure the GEMINI_API_KEY environment variable is set correctly.")
             return False

        print("API Key found and loaded by provider.")

        # Prepare a simple message
        messages = [{"role": "user", "content": "Briefly explain what model you are."}]

        # Call the complete method using LiteLLM directly
        print("Sending request to Gemini API...")

        # Override the complete method to use the correct model format
        # LiteLLM expects the model to be in the format 'gemini/model_name'
        model_with_prefix = f"gemini/{provider.model}"

        # Call LiteLLM directly instead of using provider.complete
        response = litellm.completion(
            model=model_with_prefix,
            messages=messages,
            temperature=0.1,
            api_key=provider.api_key
        )

        # Extract content from the response
        # When using litellm directly, the response format is different
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = ""

        if content:
            print(f"Success! Received response:")
            print("-" * 20)
            # Print only the first 100 chars for brevity
            print(content[:100] + "..." if len(content) > 100 else content)
            print("-" * 20)
            return True
        else:
            print("Failed: Received empty content in response.")
            print("Full response object:", response)
            return False

    except ValueError as e:
        # This likely catches the API key not found error from provider.setup()
        print(f"Failed: Configuration error during setup.")
        print(f"Error details: {e}")
        print("Ensure the GOOGLE_API_KEY environment variable is set correctly.")
        return False
    except litellm.exceptions.APIConnectionError as e:
        print(f"Failed: API Connection Error.")
        print(f"Error details: {e}")
        print("Check network connectivity and API endpoint.")
        return False
    except litellm.exceptions.AuthenticationError as e:
        print(f"Failed: Authentication Error.")
        print(f"Error details: {e}")
        print("Check if the GOOGLE_API_KEY is correct and valid.")
        return False
    except litellm.exceptions.BadRequestError as e:
        print(f"Failed: Model not found or API error.")
        print(f"Error details: {e}")
        print(f"Ensure the model name '{model_name}' is correct and available.")
        return False
    except Exception as e:
        print(f"Failed: An unexpected error occurred.")
        print(f"Error details: {type(e).__name__}: {e}")
        # Print provider state if it exists, for debugging
        if provider:
            print(f"Provider state: provider='{provider.provider}', model='{provider.model}', api_key_set={'Yes' if provider.api_key else 'No'}")
        return False

if __name__ == "__main__":
    # List of specific Gemini models to test via FEI's LiteLLM provider
    models_to_test = [
        "gemini-2.5-flash-preview-04-17", # User specified model 1
        "gemini-2.5-pro-preview-03-25",  # User specified model 2
    ]

    print("=====================================================================")
    print("Starting Gemini model accessibility test using FEI's LiteLLM logic...")
    print("This script relies on the GOOGLE_API_KEY environment variable.")
    print("=====================================================================")

    results = {}
    for model in models_to_test:
        results[model] = test_gemini_model(model)

    print("\n==================== Test Summary ====================")
    all_successful = True
    for model, success in results.items():
        status = "✅ Accessible" if success else "❌ NOT Accessible"
        print(f"- {model}: {status}")
        if not success:
            all_successful = False
    print("======================================================")

    if all_successful:
        print("\n🎉 All specified Gemini models were successfully accessed.")
    else:
        print("\n⚠️ Some Gemini models were not accessible. Please review the logs above,")
        print("   check your GOOGLE_API_KEY, network connection, and model availability.")
    print("======================================================")
