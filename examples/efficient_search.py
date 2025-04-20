#!/usr/bin/env python3
"""
Example demonstrating the token-efficient search tools in Fei
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
        
    # BatchGlob results
    if tool_name == "BatchGlob":
        print(f"Found {results['total_file_count']} files across {results['pattern_count']} patterns:\n")
        
        for pattern, files in results["results"].items():
            if isinstance(files, list):
                print(f"Pattern '{pattern}': {len(files)} files")
                for i, file in enumerate(files[:5]):  # Show first 5 files
                    print(f"  {i+1}. {file}")
                if len(files) > 5:
                    print(f"  ... and {len(files) - 5} more files")
            else:
                print(f"Pattern '{pattern}': {files}")  # Probably an error message
        
    # FindInFiles results
    elif tool_name == "FindInFiles":
        print(f"Found {results['match_count']} matches in {results['files_with_matches']} files (out of {results['file_count']} searched):\n")
        
        for file_path, matches in results["results"].items():
            if isinstance(matches, list) and matches:
                print(f"File: {file_path} ({len(matches)} matches)")
                for i, (line_num, content) in enumerate(matches[:3]):  # Show first 3 matches
                    print(f"  Line {line_num}: {content.strip()}")
                if len(matches) > 3:
                    print(f"  ... and {len(matches) - 3} more matches")
            elif isinstance(matches, dict) and "error" in matches:
                print(f"File: {file_path} - Error: {matches['error']}")
                
    # SmartSearch results
    elif tool_name == "SmartSearch":
        print(f"Searched {results['files_searched']} files with {results['patterns_searched']} patterns in {results['language']} language\n")
        
        if results["summary"]:
            print("Summary of findings:")
            for item in results["summary"]:
                print(f"\nPattern: {item['pattern']}")
                print(f"  Found in {item['files']} files with {item['matches']} total matches")
                print("  Examples:")
                for sample in item["samples"]:
                    print(f"    {sample}")
        else:
            print("No matches found.")
            
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
    """Main function to demonstrate the token-efficient search tools"""
    parser = argparse.ArgumentParser(description="Demonstrate Fei's token-efficient search tools")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--path", help="Path to search in (default: current directory)")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger = get_logger(__name__)
    
    search_path = args.path or os.getcwd()
    logger.info(f"Searching in: {search_path}")
    
    # Create tool registry and register tools
    registry = ToolRegistry()
    create_code_tools(registry)
    
    # Example 1: Using BatchGlob to search for multiple file patterns
    print("\n1. Demonstrating BatchGlob - searching for multiple file patterns at once...")
    batch_glob_results = registry.invoke_tool("BatchGlob", {
        "patterns": ["**/*.py", "**/*.md", "**/*.json"],
        "path": search_path,
        "limit_per_pattern": 10
    })
    print_results("BatchGlob", batch_glob_results)
    
    # Get some Python files to search in
    py_files = registry.invoke_tool("GlobTool", {
        "pattern": "**/*.py",
        "path": search_path
    })["files"][:10]  # Limit to 10 files
    
    if py_files:
        # Example 2: Using FindInFiles to search for patterns in specific files
        print("\n2. Demonstrating FindInFiles - searching in specific files...")
        find_results = registry.invoke_tool("FindInFiles", {
            "files": py_files,
            "pattern": "def\\s+\\w+",  # Find function definitions
            "case_sensitive": False
        })
        print_results("FindInFiles", find_results)
    
        # Example 3: Using SmartSearch for context-aware code search
        print("\n3. Demonstrating SmartSearch - context-aware code search...")
        smart_results = registry.invoke_tool("SmartSearch", {
            "query": "function search",
            "language": "python",
            "context": "tools"
        })
        print_results("SmartSearch", smart_results)
    else:
        print("\nNo Python files found to demonstrate FindInFiles and SmartSearch")
    
    print("\nExamples completed. These tools provide more efficient alternatives to multiple GlobTool and GrepTool calls.")

if __name__ == "__main__":
    main()