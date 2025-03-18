#!/usr/bin/env python3
"""
Test script for Fei with MCP Brave Search
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
from fei.tools.code import create_code_tools
from fei.core.mcp import MCPManager

async def test_mcp_search():
    """Test Fei with MCP Brave Search"""
    # Load environment variables from .env
    load_dotenv()
    
    # Create MCP manager
    mcp_manager = MCPManager()
    
    # Set Brave Search as the default MCP server
    mcp_manager.set_default_server("brave-search")
    
    # Create tool registry and add code tools
    tool_registry = ToolRegistry()
    create_code_tools(tool_registry)
    
    # Create assistant with MCP manager
    assistant = Assistant(
        provider="anthropic",
        tool_registry=tool_registry,
        mcp_manager=mcp_manager
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
        
    finally:
        # Stop the Brave Search server
        mcp_manager.stop_server("brave-search")

if __name__ == "__main__":
    asyncio.run(test_mcp_search())