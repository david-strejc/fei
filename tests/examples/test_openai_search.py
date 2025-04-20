#!/usr/bin/env python3
"""
Test script for Fei with OpenAI and Brave Search
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
from fei.tools.code import create_code_tools
from fei.core.mcp import MCPManager

async def test_openai_search():
    """Test Fei with OpenAI and Brave Search"""
    # Load environment variables from .env
    load_dotenv()
    
    # Create tool registry and add code tools
    tool_registry = ToolRegistry()
    create_code_tools(tool_registry)
    
    # Register brave search as a separate tool
    tool_registry.register_tool(
        name="brave_web_search",
        description="Performs a web search using the Brave Search API, ideal for general queries, news, articles, and online content.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "Search query (max 400 chars, 50 words)"
                },
                "count": {
                    "type": "number",
                    "description": "Number of results (1-20, default 10)"
                },
                "offset": {
                    "type": "number",
                    "description": "Pagination offset (max 9, default 0)"
                }
            },
            "required": ["query"]
        },
        handler_func=lambda args: {"message": "Search executed. This is a placeholder handler."}
    )
    
    # Create assistant with OpenAI
    assistant = Assistant(
        provider="openai",
        model="gpt-4o",
        tool_registry=tool_registry
    )
    
    print(f"Using provider: {assistant.provider}")
    print(f"Using model: {assistant.model}")
    
    try:
        # Create a system prompt that encourages internet search
        system_prompt = """You are a helpful assistant with internet search capabilities.
When asked about current information, use the Brave Search tool to find up-to-date information.
Always search for information before giving your answer if the question is about current events,
technologies, or facts that might have changed recently."""
        
        # Ask a question that requires internet search
        question = "What are the latest features in Python 3.12? Create a short example of one of the new features."
        
        print(f"\nQuestion: {question}")
        print("Searching and processing...")
        
        # Get the response
        response = await assistant.chat(question, system_prompt=system_prompt)
        
        print("\nResponse:")
        print(response)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_openai_search())