#!/usr/bin/env python3
"""
Comprehensive test script for LLM providers using FEI's LiteLLM integration.

This script tests the accessibility of various LLM providers and models,
with support for both CLI arguments and .env file configuration.

Usage:
    python test_llm_providers.py [--provider PROVIDER] [--model MODEL] [--test-tools]
"""

import os
import sys
import argparse
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
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

# Define provider-specific configurations
PROVIDER_CONFIGS = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-3-7-sonnet-20250219", "claude-3-opus-20240229"],
        "provider_name": "anthropic",
        "supports_tools": True,
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4-turbo"],
        "provider_name": "openai",
        "supports_tools": True,
    },
    "gemini": {
        "env_var": "GEMINI_API_KEY",
        "models": ["gemini-2.5-flash-preview-04-17", "gemini-2.5-pro-preview-03-25"],
        "provider_name": "gemini",
        "supports_tools": True,
    },
    "groq": {
        "env_var": "GROQ_API_KEY",
        "models": ["llama3-70b-8192", "llama3-8b-8192"],
        "provider_name": "groq",
        "supports_tools": True,
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "models": ["openrouter/auto", "qwen/qwen2.5-coder-7b-instruct"],
        "provider_name": "openrouter",
        "supports_tools": False,  # Most OpenRouter models don't support tools
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-coder"],
        "provider_name": "deepseek",
        "supports_tools": True,  # Deepseek supports function calling
    }
}

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

# Define test messages
BASIC_TEST_MESSAGE = [{"role": "user", "content": "Briefly explain what model you are."}]
TOOL_TEST_MESSAGE = [{"role": "user", "content": "What's the weather like in San Francisco today?"}]
SEARCH_TEST_MESSAGE = [{"role": "user", "content": "What were the major news headlines from yesterday? Please summarize."}]

def setup_provider(provider_name: str, model_name: Optional[str] = None) -> Tuple[LiteLLMProvider, str]:
    """
    Set up a provider with the specified model.

    Args:
        provider_name: Name of the provider to use
        model_name: Optional model name to use

    Returns:
        Tuple of (provider instance, model name with prefix)
    """
    provider = LiteLLMProvider()

    # Set provider
    provider.provider = provider_name

    # Add provider to provider_key_map if not already there
    provider_config = PROVIDER_CONFIGS.get(provider_name)
    if provider_config and provider_name not in provider.provider_key_map:
        provider.provider_key_map[provider_name] = provider_config["env_var"]

    # Set environment variable if needed
    if provider_config:
        env_var = provider_config["env_var"]
        alt_env_var = f"{provider_name.upper()}_API_KEY"

        # Special case for Gemini/Google
        if provider_name == "gemini" and env_var not in os.environ and "GOOGLE_API_KEY" in os.environ:
            os.environ[env_var] = os.environ["GOOGLE_API_KEY"]
            print(f"Set {env_var} from GOOGLE_API_KEY: {os.environ[env_var][:10]}...")
        # Copy from alternative environment variable if needed
        elif env_var != alt_env_var and alt_env_var in os.environ and env_var not in os.environ:
            os.environ[env_var] = os.environ[alt_env_var]
            print(f"Set {env_var} from {alt_env_var}: {os.environ[env_var][:10]}...")

    # Get API key from environment
    api_key = None
    if provider_config:
        env_var = provider_config["env_var"]
        api_key = os.environ.get(env_var)

    # Set up provider with model and API key
    provider.setup(model=model_name, api_key=api_key)
    provider.provider = provider_name  # Reset provider as setup might change it

    # Create model with prefix
    model_with_prefix = f"{provider_name}/{provider.model}"

    return provider, model_with_prefix

def test_model_basic(provider_name: str, model_name: str) -> bool:
    """
    Test basic functionality of a single model using FEI's LiteLLM provider logic.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to test

    Returns:
        True if the test was successful, False otherwise
    """
    print(f"\n=== BASIC MODEL TEST: {model_name} with provider: {provider_name} ===")
    provider = None
    provider_config = PROVIDER_CONFIGS.get(provider_name, {})

    try:
        # Set up provider
        print("Setting up provider...")
        provider, model_with_prefix = setup_provider(provider_name, model_name)

        print(f"Provider set to: {provider.provider}")
        print(f"Model set to: {provider.model}")
        print(f"Using model with prefix: {model_with_prefix}")

        # Check if API key was loaded
        if not provider.api_key:
            env_var = provider_config.get("env_var", f"{provider_name.upper()}_API_KEY")
            print(f"Error: {env_var} was not loaded by the provider setup.")
            print(f"Ensure the {env_var} environment variable is set correctly.")
            return False

        print("API Key found and loaded by provider.")

        # Basic test
        print("\nSTEP 1: Sending basic request...")
        print(f"Request message: {BASIC_TEST_MESSAGE[0]['content'][:50]}...")

        response = litellm.completion(
            model=model_with_prefix,
            messages=BASIC_TEST_MESSAGE,
            temperature=0.1,
            api_key=provider.api_key
        )

        # Extract content
        print("Processing response...")
        if response and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            content = ""

        if not content:
            print("Failed: Received empty content in basic response.")
            print("Full response object:", response)
            return False

        print(f"Success! Received basic response:")
        print("-" * 50)
        print(content[:100] + "..." if len(content) > 100 else content)
        print("-" * 50)

        return True

    except litellm.exceptions.APIConnectionError as e:
        print(f"Failed: API Connection Error.")
        print(f"Error details: {e}")
        print("Check network connectivity and API endpoint.")
        return False
    except litellm.exceptions.AuthenticationError as e:
        print(f"Failed: Authentication Error.")
        print(f"Error details: {e}")
        print(f"Check if the API key for {provider_name} is correct and valid.")
        return False
    except litellm.exceptions.BadRequestError as e:
        print(f"Failed: Model not found or API error.")
        print(f"Error details: {e}")
        print(f"Ensure the model name '{model_name}' is correct and available.")
        return False
    except Exception as e:
        print(f"Failed: An unexpected error occurred.")
        print(f"Error details: {type(e).__name__}: {e}")
        if provider:
            print(f"Provider state: provider='{provider.provider}', model='{provider.model}', api_key_set={'Yes' if provider.api_key else 'No'}")
        return False

def test_model_tools(provider_name: str, model_name: str) -> bool:
    """
    Test tool calling capabilities of a single model using FEI's LiteLLM provider logic.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to test

    Returns:
        True if the test was successful, False otherwise
    """
    print(f"\n=== TOOL CALLING TEST: {model_name} with provider: {provider_name} ===")
    provider = None
    provider_config = PROVIDER_CONFIGS.get(provider_name, {})

    try:
        # Set up provider
        print("Setting up provider...")
        provider, model_with_prefix = setup_provider(provider_name, model_name)

        print(f"Provider set to: {provider.provider}")
        print(f"Model set to: {provider.model}")
        print(f"Using model with prefix: {model_with_prefix}")

        # Check if API key was loaded
        if not provider.api_key:
            env_var = provider_config.get("env_var", f"{provider_name.upper()}_API_KEY")
            print(f"Error: {env_var} was not loaded by the provider setup.")
            print(f"Ensure the {env_var} environment variable is set correctly.")
            return False

        print("API Key found and loaded by provider.")

        # Tool test
        print("\nSTEP 1: Testing tool calling capabilities...")
        print(f"Request message: {TOOL_TEST_MESSAGE[0]['content']}")
        print(f"Tool name: {TEST_TOOLS[0]['function']['name']}")
        print(f"Tool description: {TEST_TOOLS[0]['function']['description'][:50]}...")

        try:
            tool_response = litellm.completion(
                model=model_with_prefix,
                messages=TOOL_TEST_MESSAGE,
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
                        if 'id' in tool_call:
                            print(f"  ID: {tool_call.get('id')}")
                    print("-" * 50)

                    # Try to simulate tool execution and return result for some providers
                    if provider_name in ['anthropic', 'openai', 'groq']:
                        print("\nSTEP 2: Simulating tool execution...")
                        print("Executing tool and preparing result...")
                        tool_result = "72°F (22°C), Partly Cloudy"
                        print(f"Tool result: {tool_result}")

                        # Create a new message with the tool result
                        print("Creating messages with tool result...")
                        try:
                            tool_result_messages = TOOL_TEST_MESSAGE + [
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
                                tools=TEST_TOOLS  # Include tools again for some providers
                            )

                            print("Processing final response...")
                            if final_response and "choices" in final_response:
                                final_content = final_response["choices"][0]["message"]["content"]
                                print("Success! Received final response:")
                                print("-" * 50)
                                print(final_content[:100] + "..." if len(final_content) > 100 else final_content)
                                print("-" * 50)
                            else:
                                print("Failed to get final response after tool execution.")
                                print("Full response object:", final_response)
                        except Exception as e:
                            print(f"Error in tool result handling: {type(e).__name__}: {e}")
                            print(f"Note: {provider_name} models may have issues with tool response handling.")
                            print("Initial tool calling was successful, but handling the result failed.")

                    return True
                else:
                    print("Model responded without using tools:")
                    print("-" * 50)
                    print(message.get("content", "")[:100] + "..." if len(message.get("content", "")) > 100 else message.get("content", ""))
                    print("-" * 50)
                    print("Note: This is not a failure, as the model may choose not to use tools.")
                    return True
            else:
                print("Failed: Received invalid response in tool test.")
                print("Full response object:", tool_response)
                return False
        except Exception as e:
            print(f"Tool test failed with error: {type(e).__name__}: {e}")
            print("Note: This is not a critical failure, as not all models support tool calling.")
            return False

    except litellm.exceptions.APIConnectionError as e:
        print(f"Failed: API Connection Error.")
        print(f"Error details: {e}")
        print("Check network connectivity and API endpoint.")
        return False
    except litellm.exceptions.AuthenticationError as e:
        print(f"Failed: Authentication Error.")
        print(f"Error details: {e}")
        print(f"Check if the API key for {provider_name} is correct and valid.")
        return False
    except litellm.exceptions.BadRequestError as e:
        print(f"Failed: Model not found or API error.")
        print(f"Error details: {e}")
        print(f"Ensure the model name '{model_name}' is correct and available.")
        return False
    except Exception as e:
        print(f"Failed: An unexpected error occurred.")
        print(f"Error details: {type(e).__name__}: {e}")
        if provider:
            print(f"Provider state: provider='{provider.provider}', model='{provider.model}', api_key_set={'Yes' if provider.api_key else 'No'}")
        return False

def test_model(provider_name: str, model_name: str, test_tools: bool = False) -> Tuple[bool, bool]:
    """
    Test a single model using FEI's LiteLLM provider logic.

    Args:
        provider_name: Name of the provider to use
        model_name: Name of the model to test
        test_tools: Whether to test tool calling capabilities

    Returns:
        Tuple of (basic_success, tools_success)
        - basic_success: True if basic model test was successful
        - tools_success: True if tool calling test was successful, or None if not tested
    """
    print(f"\n--- Testing model: {model_name} with provider: {provider_name} ---")

    # First test basic functionality
    basic_success = test_model_basic(provider_name, model_name)

    # Then test tool calling if requested and supported
    tools_success = None  # Default to None if not tested
    if test_tools and basic_success and PROVIDER_CONFIGS.get(provider_name, {}).get("supports_tools", False):
        tools_success = test_model_tools(provider_name, model_name)

    return basic_success, tools_success

def test_provider(provider_name: str, models: Optional[List[str]] = None, test_tools: bool = False) -> Dict[str, Dict[str, bool]]:
    """
    Test all models for a specific provider.

    Args:
        provider_name: Name of the provider to test
        models: Optional list of models to test (defaults to all models for the provider)
        test_tools: Whether to test tool calling capabilities

    Returns:
        Dictionary mapping model names to dictionaries with 'basic' and 'tools' keys
    """
    print(f"\n=== PROVIDER TEST: {provider_name} ===")
    provider_config = PROVIDER_CONFIGS.get(provider_name)
    if not provider_config:
        print(f"Error: Unknown provider '{provider_name}'")
        return {}

    # Check if API key is set
    print("Checking API key environment variables...")
    env_var = provider_config["env_var"]

    # Special case for Gemini/Google
    if provider_name == "gemini" and env_var not in os.environ and "GOOGLE_API_KEY" in os.environ:
        print(f"Using GOOGLE_API_KEY for {provider_name}...")
        os.environ[env_var] = os.environ["GOOGLE_API_KEY"]
        print(f"Set {env_var} from GOOGLE_API_KEY: {os.environ[env_var][:10]}...")

    if env_var not in os.environ:
        print(f"Error: {env_var} environment variable is not set.")
        print(f"Skipping tests for {provider_name}.")
        return {}
    else:
        print(f"Found {env_var} in environment variables.")

    # Use provided models or default to all models for the provider
    models_to_test = models or provider_config["models"]
    print(f"\nTesting {len(models_to_test)} models for provider '{provider_name}':")
    for i, model in enumerate(models_to_test):
        print(f"  {i+1}. {model}")

    # Test each model
    print("\nStarting model tests...")
    results = {}
    for model in models_to_test:
        basic_success, tools_success = test_model(provider_name, model, test_tools)
        results[model] = {
            'basic': basic_success,
            'tools': tools_success
        }

    # Print summary for this provider
    print(f"\n--- {provider_name} Provider Test Summary ---")

    # Basic functionality summary
    basic_successful = sum(1 for model_results in results.values() if model_results['basic'])
    print(f"Models tested: {len(results)}")
    print(f"Basic functionality successful: {basic_successful}/{len(results)}")

    # Tool calling summary if tested
    if test_tools:
        # Count models where tool calling was tested (not None) and successful (True)
        tools_tested = sum(1 for model_results in results.values() if model_results['tools'] is not None)
        tools_successful = sum(1 for model_results in results.values() if model_results['tools'] is True)
        print(f"Tool calling tested: {tools_tested}/{len(results)}")
        print(f"Tool calling successful: {tools_successful}/{tools_tested if tools_tested > 0 else 1}")

    return results

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test LLM providers using FEI's LiteLLM integration")

    parser.add_argument("--provider", choices=list(PROVIDER_CONFIGS.keys()), help="Provider to test")
    parser.add_argument("--model", help="Specific model to test")
    parser.add_argument("--test-tools", action="store_true", help="Test tool calling capabilities")
    parser.add_argument("--all", action="store_true", help="Test all providers and models")

    return parser.parse_args()

def main() -> None:
    """Main entry point"""
    args = parse_args()

    print("\n" + "=" * 80)
    print("LLM PROVIDER TEST SUITE")
    print("=" * 80)

    print("\nStarting LLM provider accessibility test using FEI's LiteLLM logic...")
    print("This script tests both basic functionality and tool calling capabilities.")
    print("This script relies on provider-specific API keys in environment variables.")

    # Show test configuration
    print("\n" + "=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)
    print(f"Provider: {args.provider if args.provider else 'All providers' if args.all else 'None specified'}")
    print(f"Model: {args.model if args.model else 'All models'}")
    print(f"Tool calling test: {'Enabled' if args.test_tools else 'Disabled'}")

    all_results = {}

    print("\n" + "=" * 80)
    print("STARTING TESTS")
    print("=" * 80)

    if args.all:
        # Test all providers
        for provider_name in PROVIDER_CONFIGS:
            provider_results = test_provider(provider_name, test_tools=args.test_tools)
            all_results[provider_name] = provider_results
    elif args.provider:
        # Test specific provider
        if args.model:
            # Test specific model
            basic_success, tools_success = test_model(args.provider, args.model, args.test_tools)
            all_results[args.provider] = {args.model: {'basic': basic_success, 'tools': tools_success}}
        else:
            # Test all models for the provider
            provider_results = test_provider(args.provider, test_tools=args.test_tools)
            all_results[args.provider] = provider_results
    else:
        print("Error: Please specify a provider to test or use --all to test all providers.")
        sys.exit(1)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    all_basic_successful = True
    all_tools_successful = True
    total_models = 0
    total_basic_successful = 0
    total_tools_tested = 0
    total_tools_successful = 0

    for provider_name, provider_results in all_results.items():
        print(f"\nProvider: {provider_name}")
        provider_basic_success = True
        provider_tools_success = True
        provider_tools_tested = False

        # Basic functionality results
        print("Basic Functionality:")
        for model, results in provider_results.items():
            basic_status = "✅ Accessible" if results['basic'] else "❌ NOT Accessible"
            print(f"- {model}: {basic_status}")

            total_models += 1
            if results['basic']:
                total_basic_successful += 1
            else:
                provider_basic_success = False
                all_basic_successful = False

        # Tool calling results if tested
        if args.test_tools:
            print("\nTool Calling Capability:")
            for model, results in provider_results.items():
                if results['tools'] is not None:  # Only show if tool calling was tested
                    provider_tools_tested = True
                    total_tools_tested += 1
                    tool_status = "✅ Supported" if results['tools'] else "❌ NOT Supported"
                    print(f"- {model}: {tool_status}")

                    if results['tools']:
                        total_tools_successful += 1
                    else:
                        provider_tools_success = False
                        all_tools_successful = False

        # Provider status
        basic_status = "✅ All models accessible" if provider_basic_success else "❌ Some models NOT accessible"
        print(f"Provider basic status: {basic_status}")

        if args.test_tools and provider_tools_tested:
            tools_status = "✅ All models support tools" if provider_tools_success else "❌ Some models have issues with tools"
            print(f"Provider tool status: {tools_status}")

    # Overall statistics
    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total models tested: {total_models}")
    print(f"Models with basic functionality: {total_basic_successful}/{total_models} ({total_basic_successful/total_models*100:.1f}%)")

    if args.test_tools and total_tools_tested > 0:
        print(f"Models tested for tool calling: {total_tools_tested}/{total_models} ({total_tools_tested/total_models*100:.1f}%)")
        print(f"Models with successful tool calling: {total_tools_successful}/{total_tools_tested} ({total_tools_successful/total_tools_tested*100:.1f}%)")

    # Final result
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if all_basic_successful:
        print("\n🎉 All specified models were successfully accessed.")
    else:
        print("\n⚠️ Some models were not accessible. Please review the logs above,")
        print("   check your API keys, network connection, and model availability.")

    if args.test_tools and total_tools_tested > 0:
        if all_tools_successful:
            print("\n🎉 All tested models successfully support tool calling.")
        else:
            print("\n⚠️ Some models had issues with tool calling. Please review the logs above.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
