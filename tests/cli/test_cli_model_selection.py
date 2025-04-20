#!/usr/bin/env python3
"""
Test script for CLI model selection in FEI.

This script tests the ability to select models via CLI arguments,
verifying that the specified model is correctly used.

Usage:
    python test_cli_model_selection.py
"""

import os
import sys
import subprocess
import argparse
from typing import Dict, List, Optional, Any, Tuple

# Ensure the 'fei' package directory is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Define test configurations
TEST_CONFIGS = [
    {
        "name": "Default model (from .env)",
        "args": [],
        "expected_provider": "anthropic",  # From .env
        "expected_model": "claude-3-7-sonnet-20250219"  # From .env
    },
    {
        "name": "CLI provider override",
        "args": ["--provider", "gemini"],
        "expected_provider": "gemini",
        "expected_model": "gemini-1.5-pro-latest"  # Default for gemini
    },
    {
        "name": "CLI model override",
        "args": ["--model", "gemini-2.5-flash-preview-04-17"],
        "expected_provider": "anthropic",  # From .env
        "expected_model": "gemini-2.5-flash-preview-04-17"
    },
    {
        "name": "CLI provider and model override",
        "args": ["--provider", "gemini", "--model", "gemini-2.5-pro-preview-03-25"],
        "expected_provider": "gemini",
        "expected_model": "gemini-2.5-pro-preview-03-25"
    }
]

def create_test_script(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """
    Create a temporary test script that prints the provider and model.
    
    Args:
        provider: Optional provider to use
        model: Optional model to use
        
    Returns:
        Path to the created script
    """
    script_content = """#!/usr/bin/env python3
import os
import sys
from fei.core.assistant import Assistant

# Create assistant with specified provider and model
assistant = Assistant(
    provider={provider},
    model={model}
)

# Print provider and model
print(f"PROVIDER: {{assistant.provider}}")
print(f"MODEL: {{assistant.model}}")
""".format(
        provider=f'"{provider}"' if provider else "None",
        model=f'"{model}"' if model else "None"
    )
    
    script_path = "temp_test_script.py"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path

def run_test(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Run a single test configuration.
    
    Args:
        config: Test configuration
        
    Returns:
        Tuple of (success, output)
    """
    print(f"\n--- Testing: {config['name']} ---")
    
    # Extract provider and model from args if present
    provider = None
    model = None
    
    for i, arg in enumerate(config["args"]):
        if arg == "--provider" and i + 1 < len(config["args"]):
            provider = config["args"][i + 1]
        elif arg == "--model" and i + 1 < len(config["args"]):
            model = config["args"][i + 1]
    
    # Create test script
    script_path = create_test_script(provider, model)
    
    try:
        # Run the script
        cmd = [sys.executable, script_path] + config["args"]
        print(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        
        print(f"Output:\n{output}")
        
        # Check if expected provider and model are in the output
        provider_line = f"PROVIDER: {config['expected_provider']}"
        model_line = f"MODEL: {config['expected_model']}"
        
        success = provider_line in output and model_line in output
        
        if success:
            print(f"✅ Test passed: Found expected provider and model")
        else:
            print(f"❌ Test failed: Did not find expected provider and model")
            print(f"Expected provider: {config['expected_provider']}")
            print(f"Expected model: {config['expected_model']}")
        
        return success, output
    
    finally:
        # Clean up
        if os.path.exists(script_path):
            os.remove(script_path)

def main() -> None:
    """Main entry point"""
    print("=====================================================================")
    print("Starting CLI model selection test for FEI...")
    print("This script tests the ability to select models via CLI arguments.")
    print("=====================================================================")
    
    results = {}
    
    for config in TEST_CONFIGS:
        success, output = run_test(config)
        results[config["name"]] = success
    
    # Print summary
    print("\n==================== Test Summary ====================")
    all_successful = True
    
    for name, success in results.items():
        status = "✅ Passed" if success else "❌ Failed"
        print(f"- {name}: {status}")
        if not success:
            all_successful = False
    
    print("======================================================")
    
    if all_successful:
        print("\n🎉 All CLI model selection tests passed.")
    else:
        print("\n⚠️ Some CLI model selection tests failed. Please review the logs above.")
    print("======================================================")

if __name__ == "__main__":
    main()
