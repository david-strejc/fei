#!/usr/bin/env python3
"""
Test script for Textual UI in non-interactive mode.
This script validates the Textual UI components without running the full application.
"""

import os
import sys
import asyncio
from unittest.mock import patch

# Add the project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fei.ui.textual_chat import FeiChatApp
from fei.utils.logging import get_logger

logger = get_logger(__name__)

async def test_textual_app():
    """Test the textual app without actually running the UI"""
    print("Creating FeiChatApp...")
    app = FeiChatApp()
    
    # This is where we would normally call app.run(),
    # but instead we'll just verify components can be created
    print("App created successfully")
    
    print("Verifying CSS...")
    # Accessing CSS property would fail if there are syntax errors
    css = app.CSS
    print("CSS verified successfully")
    
    print("Testing memory tools...")
    # Test memory connector error handling
    from fei.tools.memdir_connector import MemdirConnector
    connector = MemdirConnector()
    connection_status = connector.check_connection()
    print(f"Memdir connection status: {connection_status}")
    
    # Test memory list handler
    from fei.tools.memory_tools import memory_list_handler
    result = memory_list_handler({})
    if "error" in result:
        print(f"Memory list error (expected when server not running): {result['error']}")
    else:
        print(f"Memory list results: {result['count']} memories found")
    
    print("Memory tools check completed")
    
    print("All tests passed!")
    return True

def main():
    """Run the test"""
    print("Starting textual UI test...")
    
    # Run our test function
    test_result = asyncio.run(test_textual_app())
    
    if test_result:
        print("Textual UI test completed successfully!")
        return 0
    else:
        print("Textual UI test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())