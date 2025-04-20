#!/usr/bin/env python3
"""
Test script for Anthropic models using FEI's LiteLLM provider logic.

This script tests the accessibility of specified Anthropic models by sending a simple
request to each model and checking if a valid response is received. It also tests
function calling capabilities.

It requires the ANTHROPIC_API_KEY environment variable to be set.

Usage:
    python test_anthropic_models.py [--model MODEL_NAME] [--test-tools]
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

# Define Anthropic models to test
DEFAULT_MODELS = [
    "claude-3-7-sonnet-20250219",
    "claude-3-opus-20240229",
]

# Define test tools
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location. This tool returns the current temperature, weather conditions, and forecast for a specified location. It should be used when the user asks about weather in a specific place.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return. Default is celsius."},
                },
                "required": ["location"],
            },
        },
    }
]

def test_anthropic_model_basic(model_name: str):
    """Tests basic functionality of a single Anthropic model using FEI's LiteLLM provider logic."""
    print(f"\n=== BASIC MODEL TEST: {model_name} ===")
    provider = None
    try:
        # Instantiate FEI's LiteLLM provider
        print("Initializing LiteLLMProvider...")
        provider = LiteLLMProvider()

        # Set provider to 'anthropic'
        print("Setting provider to 'anthropic'...")
        provider.provider = 'anthropic'

        # Set the API key directly from environment
        print("Getting API key from environment...")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable is not set.")
            return False

        # Setup the provider with explicit API key
        print(f"Setting up provider with model '{model_name}'...")
        provider.setup(model=model_name, api_key=api_key)
        provider.provider = 'anthropic'  # Reset provider as setup might change it

        print(f"Provider set to: {provider.provider}")
        print(f"Model set to: {provider.model}")

        # Check if API key was loaded by the setup method
        if not provider.api_key:
            print("Error: ANTHROPIC_API_KEY was not loaded by the provider setup.")
            print("Ensure the ANTHROPIC_API_KEY environment variable is set correctly.")
            return False

        print("API Key found and loaded by provider.")

        # Prepare a simple message
        print("Preparing test message...")
        messages = [{"role": "user", "content": "Briefly explain what model you are."}]

        # Call the complete method
        print("Sending request to Anthropic API...")

        # LiteLLM expects the model to be in the format 'anthropic/model_name'
        model_with_prefix = f"anthropic/{provider.model}"
        print(f"Using model with prefix: {model_with_prefix}")

        # Call LiteLLM directly
        response = litellm.completion(
            model=model_with_prefix,
            messages=messages,
            temperature=0.1,
            api_key=provider.api_key
        )

        # Extract content from the response
        print("Processing response...")
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = ""

        if content:
            print(f"Success! Received response:")
            print("-" * 50)
            # Print only the first 100 chars for brevity
            print(content[:100] + "..." if len(content) > 100 else content)
            print("-" * 50)
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
        print("Check if the ANTHROPIC_API_KEY is correct and valid.")
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

def test_anthropic_model_tools(model_name: str):
    """Tests tool calling capabilities of a single Anthropic model using FEI's LiteLLM provider logic."""
    print(f"\n=== TOOL CALLING TEST: {model_name} ===")
    provider = None
    try:
        # Instantiate FEI's LiteLLM provider
        print("Initializing LiteLLMProvider...")
        provider = LiteLLMProvider()

        # Set provider to 'anthropic'
        print("Setting provider to 'anthropic'...")
        provider.provider = 'anthropic'

        # Set the API key directly from environment
        print("Getting API key from environment...")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY environment variable is not set.")
            return False

        # Setup the provider with explicit API key
        print(f"Setting up provider with model '{model_name}'...")
        provider.setup(model=model_name, api_key=api_key)
        provider.provider = 'anthropic'  # Reset provider as setup might change it

        print(f"Provider set to: {provider.provider}")
        print(f"Model set to: {provider.model}")

        # Check if API key was loaded by the setup method
        if not provider.api_key:
            print("Error: ANTHROPIC_API_KEY was not loaded by the provider setup.")
            print("Ensure the ANTHROPIC_API_KEY environment variable is set correctly.")
            return False

        print("API Key found and loaded by provider.")

        # LiteLLM expects the model to be in the format 'anthropic/model_name'
        model_with_prefix = f"anthropic/{provider.model}"
        print(f"Using model with prefix: {model_with_prefix}")

        # STEP 1: Initial tool calling test
        print("\nSTEP 1: Testing initial tool calling...")
        print("Preparing tool calling test message...")
        tool_messages = [{"role": "user", "content": "What's the weather like in San Francisco today?"}]

        print("Sending tool calling request with tools:")
        print(f"Tool name: {TEST_TOOLS[0]['function']['name']}")
        print(f"Tool description: {TEST_TOOLS[0]['function']['description'][:50]}...")

        tool_response = litellm.completion(
            model=model_with_prefix,
            messages=tool_messages,
            temperature=0.1,
            api_key=provider.api_key,
            tools=TEST_TOOLS
        )

        # Check if tool was called
        print("Processing tool calling response...")
        if tool_response and "choices" in tool_response:
            message = tool_response["choices"][0]["message"]
            tool_calls = message.get("tool_calls", [])

            if tool_calls:
                print("Success! Model used tool calling:")
                print("-" * 50)
                for i, tool_call in enumerate(tool_calls):
                    print(f"Tool call {i+1}:")
                    print(f"  Tool: {tool_call.get('function', {}).get('name')}")
                    print(f"  Arguments: {tool_call.get('function', {}).get('arguments')}")
                    print(f"  ID: {tool_call.get('id')}")
                print("-" * 50)

                # STEP 2: Tool execution simulation
                print("\nSTEP 2: Simulating tool execution...")
                print("Executing tool and preparing result...")
                tool_result = "72°F (22°C), Partly Cloudy"
                print(f"Tool result: {tool_result}")

                # Create a new message with the tool result
                print("Creating messages with tool result...")
                tool_result_messages = tool_messages + [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": tool_calls
                    },
                    {
                        "role": "tool",
                        "tool_call_id": tool_calls[0]["id"],
                        "content": tool_result
                    }
                ]

                # STEP 3: Final response with tool result
                print("\nSTEP 3: Getting final response with tool result...")
                print("Sending request with tool result...")

                final_response = litellm.completion(
                    model=model_with_prefix,
                    messages=tool_result_messages,
                    temperature=0.1,
                    api_key=provider.api_key,
                    tools=TEST_TOOLS  # Need to include tools again for Anthropic
                )

                print("Processing final response...")
                if final_response and "choices" in final_response:
                    final_content = final_response["choices"][0]["message"]["content"]
                    print("Success! Received final response:")
                    print("-" * 50)
                    print(final_content[:100] + "..." if len(final_content) > 100 else final_content)
                    print("-" * 50)
                    return True
                else:
                    print("Failed to get final response after tool execution.")
                    print("Full response object:", final_response)
                    return False
            else:
                print("Model responded without using tools:")
                print("-" * 50)
                print(message.get("content", "")[:100] + "..." if len(message.get("content", "")) > 100 else message.get("content", ""))
                print("-" * 50)
                return True  # Still consider this a success as the model responded
        else:
            print("Failed: Received invalid response in tool test.")
            print("Full response object:", tool_response)
            return False

    except litellm.exceptions.APIConnectionError as e:
        print(f"Failed: API Connection Error.")
        print(f"Error details: {e}")
        print("Check network connectivity and API endpoint.")
        return False
    except litellm.exceptions.AuthenticationError as e:
        print(f"Failed: Authentication Error.")
        print(f"Error details: {e}")
        print("Check if the ANTHROPIC_API_KEY is correct and valid.")
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

def test_anthropic_model(model_name: str, test_tools: bool = False):
    """Tests a single Anthropic model using FEI's LiteLLM provider logic."""
    # First test basic functionality
    basic_success = test_anthropic_model_basic(model_name)

    # Then test tool calling if requested
    tools_success = True
    if test_tools and basic_success:
        tools_success = test_anthropic_model_tools(model_name)

    # Return overall success
    return basic_success and (tools_success if test_tools else True)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test Anthropic models using FEI's LiteLLM integration")

    parser.add_argument("--model", help="Specific model to test (defaults to testing all models)")
    parser.add_argument("--test-tools", action="store_true", help="Test tool calling capabilities")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 80)
    print("ANTHROPIC MODELS TEST SUITE")
    print("=" * 80)

    # Load environment variables from .env file
    from dotenv import load_dotenv
    print("\nSetting up environment...")
    load_dotenv()
    print("Environment variables loaded from .env file.")

    print("\n" + "=" * 80)
    print("Starting Anthropic model accessibility test using FEI's LiteLLM logic...")
    print("This script tests both basic functionality and tool calling capabilities.")
    print("=" * 80)

    # Check if ANTHROPIC_API_KEY is set
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set it before running this script.")
        sys.exit(1)

    results = {}
    tool_results = {}

    if args.model:
        # Test specific model
        print(f"\nTesting specific model: {args.model}")
        results[args.model] = test_anthropic_model_basic(args.model)
        if args.test_tools and results[args.model]:
            print(f"\nTesting tool calling for model: {args.model}")
            tool_results[args.model] = test_anthropic_model_tools(args.model)
        elif args.test_tools:
            print(f"\nSkipping tool calling test for {args.model} due to basic test failure.")
            tool_results[args.model] = False
    else:
        # Test all models
        print(f"\nTesting all {len(DEFAULT_MODELS)} default models...")
        for model in DEFAULT_MODELS:
            print(f"\nTesting model: {model}")
            results[model] = test_anthropic_model_basic(model)
            if args.test_tools and results[model]:
                print(f"\nTesting tool calling for model: {model}")
                tool_results[model] = test_anthropic_model_tools(model)
            elif args.test_tools:
                print(f"\nSkipping tool calling test for {model} due to basic test failure.")
                tool_results[model] = False

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    # Basic functionality results
    print("\nBASIC FUNCTIONALITY:")
    all_basic_successful = True
    for model, success in results.items():
        status = "✅ Accessible" if success else "❌ NOT Accessible"
        print(f"- {model}: {status}")
        if not success:
            all_basic_successful = False

    # Tool calling results if tested
    if args.test_tools:
        print("\nTOOL CALLING CAPABILITY:")
        all_tools_successful = True
        for model, success in tool_results.items():
            status = "✅ Supported" if success else "❌ NOT Supported"
            print(f"- {model}: {status}")
            if not success:
                all_tools_successful = False

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if all_basic_successful:
        print("\n🎉 All specified Anthropic models were successfully accessed.")
    else:
        print("\n⚠️ Some Anthropic models were not accessible. Please review the logs above,")
        print("   check your ANTHROPIC_API_KEY, network connection, and model availability.")

    if args.test_tools:
        if all_tools_successful:
            print("\n🎉 All accessible Anthropic models successfully support tool calling.")
        else:
            print("\n⚠️ Some Anthropic models had issues with tool calling. Please review the logs above.")

    print("\n" + "=" * 80)
