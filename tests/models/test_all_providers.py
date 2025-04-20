#!/usr/bin/env python3
"""
Comprehensive test script for all LLM providers using FEI's LiteLLM integration.

This script tests all configured providers and models, with support for both
CLI arguments and .env file configuration.

Usage:
    python test_all_providers.py [--test-tools] [--report REPORT_FILE]
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any

# Import the test_llm_providers module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from test_llm_providers import PROVIDER_CONFIGS, test_provider

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Test all LLM providers using FEI's LiteLLM integration")

    parser.add_argument("--test-tools", action="store_true", help="Test tool calling capabilities")
    parser.add_argument("--report", help="File to save the test report to (JSON format)")

    return parser.parse_args()

def generate_report(results: Dict[str, Dict[str, Dict[str, bool]]], test_tools: bool) -> Dict[str, Any]:
    """
    Generate a report from the test results.

    Args:
        results: Dictionary mapping provider names to dictionaries mapping model names to dictionaries with 'basic' and 'tools' keys
        test_tools: Whether tool calling was tested

    Returns:
        Report dictionary
    """
    # Count statistics
    total_models = 0
    basic_successful = 0
    tools_tested = 0
    tools_successful = 0
    provider_basic_status = {}
    provider_tools_status = {}

    for provider, models in results.items():
        provider_basic_success = True
        provider_tools_success = True
        provider_tools_tested = False

        for model, model_results in models.items():
            total_models += 1

            # Basic functionality
            if model_results['basic']:
                basic_successful += 1
            else:
                provider_basic_success = False

            # Tool calling
            if model_results['tools'] is not None:
                tools_tested += 1
                provider_tools_tested = True
                if model_results['tools']:
                    tools_successful += 1
                else:
                    provider_tools_success = False

        provider_basic_status[provider] = provider_basic_success
        if provider_tools_tested:
            provider_tools_status[provider] = provider_tools_success

    report = {
        "timestamp": datetime.now().isoformat(),
        "test_tools": test_tools,
        "results": results,
        "summary": {
            "providers_tested": len(results),
            "models_tested": total_models,
            "basic": {
                "successful": basic_successful,
                "failed": total_models - basic_successful,
                "provider_status": provider_basic_status
            },
            "tools": {
                "tested": tools_tested,
                "successful": tools_successful,
                "failed": tools_tested - tools_successful,
                "provider_status": provider_tools_status
            }
        }
    }

    return report

def main() -> None:
    """Main entry point"""
    args = parse_args()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE LLM PROVIDER TEST SUITE")
    print("=" * 80)

    print("\nStarting comprehensive LLM provider test using FEI's LiteLLM logic...")
    print("This script tests all configured providers and models.")

    # Show test configuration
    print("\n" + "=" * 80)
    print("TEST CONFIGURATION")
    print("=" * 80)
    print(f"Tool calling test: {'Enabled' if args.test_tools else 'Disabled'}")
    print(f"Report file: {args.report if args.report else 'None'}")
    print(f"Providers to test: {len(PROVIDER_CONFIGS)}")

    # Count total models to test
    total_models = sum(len(config['models']) for config in PROVIDER_CONFIGS.values())
    print(f"Total models to test: {total_models}")

    # List all providers and models
    print("\nProviders and models to test:")
    for provider_name, config in PROVIDER_CONFIGS.items():
        print(f"  - {provider_name}: {', '.join(config['models'])}")

    print("\n" + "=" * 80)
    print("STARTING TESTS")
    print("=" * 80)

    all_results = {}

    # Test each provider
    for i, provider_name in enumerate(PROVIDER_CONFIGS):
        print(f"\n\n=== Testing provider {i+1}/{len(PROVIDER_CONFIGS)}: {provider_name} ===")
        provider_results = test_provider(provider_name, test_tools=args.test_tools)
        all_results[provider_name] = provider_results

    # Generate report
    print("\n" + "=" * 80)
    print("GENERATING REPORT")
    print("=" * 80)

    report = generate_report(all_results, args.test_tools)

    # Save report if requested
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.report}")

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)

    print(f"\nOVERALL STATISTICS:")
    print(f"Providers tested: {report['summary']['providers_tested']}")
    print(f"Models tested: {report['summary']['models_tested']}")
    print(f"Models with basic functionality: {report['summary']['basic']['successful']}/{report['summary']['models_tested']} ({report['summary']['basic']['successful']/report['summary']['models_tested']*100:.1f}%)")

    if args.test_tools and report['summary']['tools']['tested'] > 0:
        print(f"Models tested for tool calling: {report['summary']['tools']['tested']}/{report['summary']['models_tested']} ({report['summary']['tools']['tested']/report['summary']['models_tested']*100:.1f}%)")
        print(f"Models with successful tool calling: {report['summary']['tools']['successful']}/{report['summary']['tools']['tested']} ({report['summary']['tools']['successful']/report['summary']['tools']['tested']*100:.1f}%)")

    print("\nRESULTS BY PROVIDER:")
    for provider_name, provider_results in all_results.items():
        if provider_results:  # Only show providers that were actually tested
            print(f"\nProvider: {provider_name}")

            # Basic functionality results
            print("Basic Functionality:")
            for model, results in provider_results.items():
                basic_status = "✅ Accessible" if results['basic'] else "❌ NOT Accessible"
                print(f"- {model}: {basic_status}")

            # Tool calling results if tested
            if args.test_tools:
                # Check if any model was tested for tool calling
                tools_tested = any(results['tools'] is not None for results in provider_results.values())
                if tools_tested:
                    print("\nTool Calling Capability:")
                    for model, results in provider_results.items():
                        if results['tools'] is not None:  # Only show if tool calling was tested
                            tool_status = "✅ Supported" if results['tools'] else "❌ NOT Supported"
                            print(f"- {model}: {tool_status}")

            # Provider status
            provider_basic_status = report['summary']['basic']['provider_status'].get(provider_name, False)
            basic_status = "✅ All models accessible" if provider_basic_status else "❌ Some models NOT accessible"
            print(f"Provider basic status: {basic_status}")

            if args.test_tools and provider_name in report['summary']['tools']['provider_status']:
                provider_tools_status = report['summary']['tools']['provider_status'].get(provider_name, False)
                tools_status = "✅ All models support tools" if provider_tools_status else "❌ Some models have issues with tools"
                print(f"Provider tool status: {tools_status}")

    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if report['summary']['basic']['failed'] == 0:
        print("\n🎉 All specified models were successfully accessed.")
    else:
        print(f"\n⚠️ {report['summary']['basic']['failed']} models were not accessible. Please review the logs above,")
        print("   check your API keys, network connection, and model availability.")

    if args.test_tools and report['summary']['tools']['tested'] > 0:
        if report['summary']['tools']['failed'] == 0:
            print("\n🎉 All tested models successfully support tool calling.")
        else:
            print(f"\n⚠️ {report['summary']['tools']['failed']} models had issues with tool calling. Please review the logs above.")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
