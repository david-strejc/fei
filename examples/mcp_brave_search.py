#!/usr/bin/env python3
"""
Example for using the Brave Search MCP server with Fei
"""

import asyncio
import argparse
from fei.core.mcp import MCPManager

async def test_brave_search(query="Python programming language"):
    """Test the Brave Search MCP server"""
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
        
        # Check if running in interactive mode
        import sys
        if sys.stdin.isatty():
            # Get user query
            query = input("\nEnter search query: ")
        else:
            print(f"\nUsing default query: {query}")
        
        # Try to perform web search using regular web API
        # This is a direct API search since the MCP server requires specific npm package
        print(f"\nSearching for: {query}")
        
        try:
            # Try the MCP server first
            results = await mcp_manager.brave_search.brave_web_search(query=query, count=5)
        except Exception as e:
            print(f"MCP search failed: {e}")
            print("Falling back to direct API call...")
            
            # Import requests for direct API call
            import requests
            
            # Make direct API call to Brave Search
            # Get API key from environment
            import os
            brave_api_key = os.environ.get("BRAVE_API_KEY")
            if not brave_api_key:
                raise ValueError("BRAVE_API_KEY environment variable not set")
            headers = {"X-Subscription-Token": brave_api_key, "Accept": "application/json"}
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
            print(f"   Description: {result.get('description')}")
            print()
    
    finally:
        # Stop the Brave Search server
        mcp_manager.stop_server("brave-search")

async def test_with_assistant(query="What are the latest features in Python 3.11?"):
    """Test using the assistant with Brave Search"""
    from fei.core.assistant import Assistant
    
    # Create assistant
    assistant = Assistant()
    
    try:
        # Set up system prompt
        system_prompt = """You are a helpful assistant with the ability to search the web.
When the user asks for information that might require current data,
use the Brave Search tool to find relevant information and provide 
a comprehensive answer."""
        
        # Check if running in interactive mode
        import sys
        if sys.stdin.isatty():
            # Get user query
            query = input("\nEnter a question that requires current information: ")
        else:
            print(f"\nUsing default query: {query}")
        
        # Get response
        print("\nSearching and processing...")
        response = await assistant.chat(query, system_prompt=system_prompt)
        
        # Display response
        print("\nAssistant Response:")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Brave Search MCP integration")
    parser.add_argument("--assistant", action="store_true", help="Test with assistant integration")
    args = parser.parse_args()
    
    if args.assistant:
        asyncio.run(test_with_assistant())
    else:
        asyncio.run(test_brave_search())