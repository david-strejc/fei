#!/usr/bin/env python3
"""
Test script for Groq models using FEI's LiteLLM provider logic.

This script tests the accessibility of specified Groq models by sending a simple
request to each model and checking if a valid response is received.

It requires the GROQ_API_KEY environment variable to be set.

Usage:
    python test_groq_models.py [--model MODEL_NAME] [--test-tools]
"""

import os
import sys
import argparse
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

# Define Groq models to test
DEFAULT_MODELS = [
    "llama3-70b-8192",
    "llama3-8b-8192",
]

# Define test tools
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

def test_groq_model(model_name: str, test_tools: bool = False):
    """Tests a single Groq model using FEI's LiteLLM provider logic."""
    print(f"\n--- Testing model: {model_name} ---")
    provider = None
    try:
        # Instantiate FEI's LiteLLM provider
        provider = LiteLLMProvider()
        
        # Set provider to 'groq'
        provider.provider = 'groq'
        
        # Setup the provider - it will try to find GROQ_API_KEY in env
        provider.setup(model=model_name)
        provider.provider = 'groq'  # Reset provider as setup might change it
        
        print(f"Provider set to: {provider.provider}")
        print(f"Model set to: {provider.model}")
        
        # Check if API key was loaded by the setup method
        if not provider.api_key:
            print("Error: GROQ_API_KEY was not loaded by the provider setup.")
            print("Ensure the GROQ_API_KEY environment variable is set correctly.")
            return False
        
        print("API Key found and loaded by provider.")
        
        # Prepare a simple message
        messages = [{"role": "user", "content": "Briefly explain what model you are."}]
        
        # Call the complete method
        print("Sending request to Groq API...")
        
        # LiteLLM expects the model to be in the format 'groq/model_name'
        model_with_prefix = f"groq/{provider.model}"
        
        # Call LiteLLM directly
        response = litellm.completion(
            model=model_with_prefix,
            messages=messages,
            temperature=0.1,
            api_key=provider.api_key
        )
        
        # Extract content from the response
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
            
            # Test tool calling if requested
            if test_tools:
                print("\nTesting tool calling capabilities...")
                try:
                    tool_messages = [{"role": "user", "content": "What's the weather like in San Francisco today?"}]
                    tool_response = litellm.completion(
                        model=model_with_prefix,
                        messages=tool_messages,
                        temperature=0.1,
                        api_key=provider.api_key,
                        tools=TEST_TOOLS
                    )
                    
                    # Check if tool was called
                    if tool_response and "choices" in tool_response:
                        message = tool_response["choices"][0]["message"]
                        tool_calls = message.get("tool_calls", [])
                        
                        if tool_calls:
                            print("Success! Model used tool calling:")
                            print("-" * 20)
                            for tool_call in tool_calls:
                                print(f"Tool: {tool_call.get('function', {}).get('name')}")
                                print(f"Arguments: {tool_call.get('function', {}).get('arguments')}")
                            print("-" * 20)
                        else:
                            print("Model responded without using tools:")
                            print("-" * 20)
                            print(message.get("content", "")[:100] + "..." if len(message.get("content", "")) > 100 else message.get("content", ""))
                            print("-" * 20)
                    else:
                        print("Failed: Received invalid response in tool test.")
                        print("Full response object:", tool_response)
                except Exception as e:
                    print(f"Tool test failed with error: {type(e).__name__}: {e}")
            
            return True
        else:
            print("Failed: Received empty content in response.")
            print("Full response object:", response)
            return False
        
    except litellm.exceptions.APIConnectionError as e:
        print(f"Failed: API Connection Error.")
        print(f"Error details: {e}")
        print("Check network connectivity and API endpoint.")
        return False
    except litellm.exceptions.AuthenticationError as e:
        print(f"Failed: Authentication Error.")
        print(f"Error details: {e}")
        print("Check if the GROQ_API_KEY is correct and valid.")
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

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test Groq models using FEI's LiteLLM integration")
    
    parser.add_argument("--model", help="Specific model to test (defaults to testing all models)")
    parser.add_argument("--test-tools", action="store_true", help="Test tool calling capabilities")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print("=====================================================================")
    print("Starting Groq model accessibility test using FEI's LiteLLM logic...")
    print("This script relies on the GROQ_API_KEY environment variable.")
    print("=====================================================================")
    
    # Check if GROQ_API_KEY is set
    if "GROQ_API_KEY" not in os.environ:
        print("Error: GROQ_API_KEY environment variable is not set.")
        print("Please set it before running this script.")
        sys.exit(1)
    
    results = {}
    
    if args.model:
        # Test specific model
        results[args.model] = test_groq_model(args.model, args.test_tools)
    else:
        # Test all models
        for model in DEFAULT_MODELS:
            results[model] = test_groq_model(model, args.test_tools)
    
    print("\n==================== Test Summary ====================")
    all_successful = True
    for model, success in results.items():
        status = "✅ Accessible" if success else "❌ NOT Accessible"
        print(f"- {model}: {status}")
        if not success:
            all_successful = False
    print("======================================================")
    
    if all_successful:
        print("\n🎉 All specified Groq models were successfully accessed.")
    else:
        print("\n⚠️ Some Groq models were not accessible. Please review the logs above,")
        print("   check your GROQ_API_KEY, network connection, and model availability.")
    print("======================================================")
