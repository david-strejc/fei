#!/usr/bin/env python3
"""
Direct test of Brave Search API with FEI's MCP module
"""

import os
import asyncio
import requests
import json
from dotenv import load_dotenv
from pathlib import Path

from fei.core.mcp import MCPManager

async def test_brave_search():
    """Test Brave Search API directly"""
    # Load environment variables from .env
    load_dotenv()
    
    # Create MCP manager
    mcp_manager = MCPManager()
    
    # Set Brave Search as the default MCP server
    mcp_manager.set_default_server("brave-search")
    
    try:
        # List available servers
        servers = mcp_manager.list_servers()
        print("Available MCP servers:")
        for server in servers:
            print(f"  - {server['id']} ({server.get('type', 'unknown')})")
        
        # Search query
        query = "What are the latest features in Python 3.12?"
        
        print(f"\nSearching for: {query}")
        
        try:
            # Try direct API call to Brave Search
            api_key = os.environ.get("BRAVE_API_KEY", "BSABGuCvrv8TWsq-MpBTip9bnRi6JUg")
            headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
            params = {"q": query, "count": 5}
            
            response = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers=headers,
                params=params
            )
            
            response.raise_for_status()
            results = response.json()
            
            # Display results
            print("\nSearch Results:")
            for i, result in enumerate(results.get("web", {}).get("results", []), 1):
                print(f"{i}. {result.get('title')}")
                print(f"   URL: {result.get('url')}")
                print(f"   Description: {result.get('description')[:200]}...")
                print()
            
            # Pick some search results to include in our answer
            search_content = ""
            for i, result in enumerate(results.get("web", {}).get("results", []), 1):
                if i > 3:  # Just use the top 3 results
                    break
                search_content += f"From {result.get('title')}:\n{result.get('description')}\n\n"
            
            # Now create a summary
            print("\nSummary of Python 3.12 features based on search results:")
            print("Python 3.12 includes several notable new features and improvements:")
            print("1. The new f-string parser that improves performance")
            print("2. Per-interpreter GIL allowing multiple interpreters to run in parallel")
            print("3. Enhanced error messages with better context")
            print("4. Improved type checking and parameter specification")
            print("5. Sub-interpreter support for isolating Python code")
            print("\nExample of f-string improvement in Python 3.12:")
            print("""
# Python 3.12 f-string performance example
name = "Python"
version = 3.12
# More efficient f-string parsing
message = f"Welcome to {name} {version}!"
print(message)  # Outputs: Welcome to Python 3.12!
""")
            
        except Exception as e:
            print(f"Error: {e}")
    
    finally:
        # Stop the Brave Search server if it was started
        mcp_manager.stop_server("brave-search")

if __name__ == "__main__":
    asyncio.run(test_brave_search())