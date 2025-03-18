#!/usr/bin/env python3
"""
Brave Search Direct API Test Tool

This script bypasses the MCP server and directly calls the Brave Search API.
Use this to test the Brave search functionality and isolate MCP server issues.
"""

import os
import sys
import json
import argparse
import asyncio
import requests
from pathlib import Path

# Add the project root to the Python path if not already there
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fei.utils.config import Config
from fei.utils.logging import get_logger, setup_logging
from fei.core.mcp import MCPManager

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}

def colorize(text, color):
    """Apply color to text"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

async def direct_api_search(query, count=5, offset=0):
    """
    Search directly with the Brave Search API
    
    Args:
        query: Search query
        count: Number of results (default: 5)
        offset: Pagination offset (default: 0)
        
    Returns:
        Search results
    """
    # Get API key from environment or use default
    config = Config()
    brave_api_key = os.environ.get("BRAVE_API_KEY", config.get("brave.api_key"))
    
    if not brave_api_key:
        # Use the default developer key as a last resort
        brave_api_key = "BSABGuCvrv8TWsq-MpBTip9bnRi6JUg"
        logger.warning("Using default Brave API key. For production, set BRAVE_API_KEY in your .env file.")
    
    # Configure the request
    headers = {"X-Subscription-Token": brave_api_key, "Accept": "application/json"}
    params = {"q": query, "count": count, "offset": offset}
    
    # Make the request
    logger.info(f"Making direct API call to Brave Search with API key: {brave_api_key[:4]}...{brave_api_key[-4:]}")
    
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Parse the response
        results = response.json()
        
        return results
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Status code: {e.response.status_code}")
            logger.error(f"Response body: {e.response.text}")
        raise

async def mcp_server_search(query, count=5, offset=0):
    """
    Search using the MCP server
    
    Args:
        query: Search query
        count: Number of results (default: 5)
        offset: Pagination offset (default: 0)
        
    Returns:
        Search results
    """
    # Create MCP manager
    config = Config()
    mcp_manager = MCPManager(config)
    
    # Set Brave Search as the default MCP server
    mcp_manager.set_default_server("brave-search")
    
    try:
        # Try with brave_web_search method
        logger.info("Trying brave_web_search method...")
        try:
            results = await mcp_manager.brave_search.brave_web_search(
                query=query,
                count=count,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"brave_web_search method failed: {e}")
        
        # Try with web_search method
        logger.info("Trying web_search method...")
        try:
            results = await mcp_manager.brave_search.search(
                query=query,
                count=count,
                offset=offset
            )
            return results
        except Exception as e:
            logger.error(f"web_search method failed: {e}")
        
        # Both methods failed, raise exception
        raise Exception("All MCP methods failed")
    
    finally:
        # Stop the Brave Search server
        mcp_manager.stop_server("brave-search")

def display_results(results):
    """Display search results"""
    if not results:
        print(colorize("No results found.", "yellow"))
        return
    
    # Print web results
    web_results = results.get("web", {}).get("results", [])
    if web_results:
        print(colorize("\nWeb Results:", "bold"))
        for i, result in enumerate(web_results, 1):
            print(colorize(f"{i}. {result.get('title')}", "green"))
            print(colorize(f"   URL: {result.get('url')}", "blue"))
            print(f"   {result.get('description')}")
            print()
    
    # Print news results if available
    news_results = results.get("news", {}).get("results", [])
    if news_results:
        print(colorize("\nNews Results:", "bold"))
        for i, result in enumerate(news_results, 1):
            print(colorize(f"{i}. {result.get('title')}", "green"))
            print(colorize(f"   URL: {result.get('url')}", "blue"))
            print(f"   {result.get('description')}")
            print()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test Brave Search API directly or through MCP")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--offset", type=int, default=0, help="Pagination offset (default: 0)")
    parser.add_argument("--mcp", action="store_true", help="Use MCP server instead of direct API call")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--dump", action="store_true", help="Dump raw JSON response")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.debug:
        setup_logging(level="DEBUG")
    
    print(colorize(f"Search query: {args.query}", "bold"))
    
    try:
        if args.mcp:
            print(colorize("Using MCP server...", "cyan"))
            results = await mcp_server_search(args.query, args.count, args.offset)
        else:
            print(colorize("Using direct API call...", "cyan"))
            results = await direct_api_search(args.query, args.count, args.offset)
        
        if args.dump:
            # Dump raw JSON response
            print(colorize("\nRaw JSON response:", "bold"))
            print(json.dumps(results, indent=2))
        else:
            # Display formatted results
            display_results(results)
            
    except Exception as e:
        print(colorize(f"Error: {e}", "red"))
        if args.debug:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())