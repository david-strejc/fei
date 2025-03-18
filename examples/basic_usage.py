#!/usr/bin/env python3
"""
Basic usage example for Fei code assistant
"""

import asyncio
import os
from pathlib import Path

from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
from fei.tools.code import create_code_tools

async def main():
    """Run a simple example of Fei code assistant"""
    
    # Load API keys from config/keys
    keys_file = Path("config/keys")
    if keys_file.exists():
        with open(keys_file, "r") as f:
            for line in f:
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    value = value.strip('"\'')
                    os.environ[key] = value
    
    # Create tool registry and add code tools
    tool_registry = ToolRegistry()
    create_code_tools(tool_registry)
    
    # Create assistant
    assistant = Assistant(
        provider="anthropic",  # Can be "anthropic", "openai", "groq", etc.
        tool_registry=tool_registry
    )
    
    print(f"Using provider: {assistant.provider}")
    print(f"Using model: {assistant.model}")
    print("Ask a question about your code or request a code-related task...")
    
    # Get user input
    user_message = input("> ")
    
    # Get assistant response
    print("Fei is thinking...")
    response = await assistant.chat(user_message)
    
    # Display response
    print("\nFei's response:")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())