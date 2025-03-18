#!/usr/bin/env python3
"""
Example demonstrating the repository mapping features in Fei
"""

import os
import sys
import argparse
from typing import Dict, Any

# Add the parent directory to the path so we can import fei
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.tools.registry import ToolRegistry
from fei.tools.code import create_code_tools
from fei.utils.logging import get_logger, setup_logging

def print_results(tool_name: str, results: Dict[str, Any]) -> None:
    """Print results from a tool in a formatted way"""
    print(f"\n===== {tool_name} Results =====")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
        
    # RepoMap results
    if tool_name == "RepoMap":
        print(f"Repository: {results['repository']}")
        print(f"Token count: {int(results['token_count'])}")
        print("\nRepository Map:")
        print("-" * 80)
        print(results["map"])
        print("-" * 80)
    
    # RepoSummary results
    elif tool_name == "RepoSummary":
        print(f"Repository: {results['repository']}")
        print(f"Module count: {results['module_count']}")
        print(f"Token count: {int(results['token_count'])}")
        print("\nRepository Summary:")
        print("-" * 80)
        print(results["summary"])
        print("-" * 80)
    
    # RepoDependencies results
    elif tool_name == "RepoDependencies":
        print(f"Repository: {results['repository']}")
        print(f"Total files analyzed: {results['file_count']}")
        print("\nDependency Visualization:")
        print("-" * 80)
        print(results["visual"])
        print("-" * 80)
        
        print("\nModule-level dependencies:")
        for module, deps in results["module_dependencies"].items():
            if deps:
                print(f"  {module} → {', '.join(deps)}")
    
    # Default format for other tools
    else:
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 10:
                print(f"{key}: {value[:10]} ... (and {len(value) - 10} more)")
            elif isinstance(value, dict) and len(value) > 10:
                keys = list(value.keys())
                print(f"{key}: {keys[:10]} ... (and {len(keys) - 10} more keys)")
            else:
                print(f"{key}: {value}")

def main():
    """Main function to demonstrate the repository mapping tools"""
    parser = argparse.ArgumentParser(description="Demonstrate Fei's repository mapping tools")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--path", help="Path to repository (default: current directory)")
    parser.add_argument("--tokens", type=int, default=1000, help="Token budget for repo map (default: 1000)")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger = get_logger(__name__)
    
    repo_path = args.path or os.getcwd()
    logger.info(f"Analyzing repository: {repo_path}")
    
    # Create tool registry and register tools
    registry = ToolRegistry()
    create_code_tools(registry)
    
    # Example 1: Generate a repository map
    print("\n1. Generating Repository Map...")
    repo_map_results = registry.invoke_tool("RepoMap", {
        "path": repo_path,
        "token_budget": args.tokens
    })
    print_results("RepoMap", repo_map_results)
    
    # Example 2: Generate a repository summary
    print("\n2. Generating Repository Summary...")
    repo_summary_results = registry.invoke_tool("RepoSummary", {
        "path": repo_path,
        "max_tokens": args.tokens // 2  # Use half the tokens for summary
    })
    print_results("RepoSummary", repo_summary_results)
    
    # Example 3: Extract repository dependencies
    print("\n3. Extracting Repository Dependencies...")
    repo_deps_results = registry.invoke_tool("RepoDependencies", {
        "path": repo_path,
        "depth": 1
    })
    print_results("RepoDependencies", repo_deps_results)
    
    print("\nExamples completed. The repository mapping tools provide a concise understanding of the codebase structure.")

if __name__ == "__main__":
    main()