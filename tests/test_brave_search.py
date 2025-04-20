#!/usr/bin/env python3
"""
Test script for Brave Search MCP integration
"""

import asyncio
from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
from fei.tools.code import create_code_tools
from fei.core.mcp import MCPManager

async def main():
    """Test Brave Search with direct MCP method"""
    
    print("Testing Brave Search integration...")
    
    # Create MCP manager
    mcp_manager = MCPManager()
    
    try:
        # Create tool registry
        tool_registry = ToolRegistry()
        create_code_tools(tool_registry)
        
        # Create assistant
        assistant = Assistant(
            provider="anthropic",
            tool_registry=tool_registry,
            mcp_manager=mcp_manager
        )
        
        # Set up system prompt that encourages using search
        system_prompt = """You are a helpful assistant with internet search capabilities.
When asked about current information, use the brave_web_search tool to find up-to-date information.
Always search for information before answering if the question is about current events,
technologies, or facts that might have changed recently."""
        
        # Ask a question that requires search
        question = "What is the current weather in Prague?"
        
        print(f"\nQuestion: {question}")
        print("Searching and processing...")
        
        # Get assistant response
        response = await assistant.chat(question, system_prompt=system_prompt)
        
        print("\nResponse:")
        print(response)
        
    finally:
        # Stop the Brave Search server
        mcp_manager.stop_server("brave-search")

if __name__ == "__main__":
    asyncio.run(main())