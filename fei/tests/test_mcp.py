#!/usr/bin/env python3
"""
Tests for MCP integration in Fei
"""

import os
import asyncio
import unittest
from unittest.mock import patch, MagicMock

from fei.core.mcp import MCPClient, MCPManager, MCPBraveSearchService

class TestMCPIntegration(unittest.TestCase):
    """Test MCP integration"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a test client
        self.client = MCPClient()
        
        # Ensure brave-search server exists
        self.assertTrue("brave-search" in self.client.servers, 
                      "Brave Search server configuration not found")
    
    def test_server_config(self):
        """Test server configuration"""
        # Check server configuration
        server = self.client.get_server("brave-search")
        self.assertIsNotNone(server, "Failed to get server configuration")
        self.assertEqual(server.get("type"), "stdio", "Server type is not stdio")
        self.assertEqual(server.get("command"), "npx", "Command is not npx")
        self.assertTrue(any("@modelcontextprotocol/server-brave-search" in arg 
                          for arg in server.get("args", [])), 
                       "server-brave-search not found in args")
    
    def test_list_servers(self):
        """Test listing servers"""
        servers = self.client.list_servers()
        self.assertTrue(any(s.get("id") == "brave-search" for s in servers),
                       "Brave Search server not found in list")
    
    @patch("subprocess.Popen")
    @patch("asyncio.sleep")
    def test_start_server(self, mock_sleep, mock_popen):
        """Test starting a server"""
        # Mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is running
        mock_popen.return_value = mock_process
        
        # Create client and start server
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            self.client._start_stdio_server("brave-search", self.client.servers["brave-search"])
        )
        
        # Check process was started
        mock_popen.assert_called_once()
        self.assertIn("brave-search", self.client.processes)
        
        # Clean up
        self.client.stop_server("brave-search")
    
    @patch("subprocess.Popen")
    @patch("asyncio.sleep")
    def test_call_service(self, mock_sleep, mock_popen):
        """Test calling a service"""
        # Mock process with response
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is running
        mock_process.stdout.readline.return_value = '{"jsonrpc":"2.0","id":1,"result":{"web":{"results":[{"title":"Test Result"}]}}}'
        mock_popen.return_value = mock_process
        
        # Create brave search service
        brave_search = MCPBraveSearchService(self.client, "brave-search")
        
        # Call service
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            brave_search.brave_web_search("test query")
        )
        
        # Check results
        self.assertIn("web", result)
        self.assertIn("results", result["web"])
        self.assertEqual(result["web"]["results"][0]["title"], "Test Result")
        
        # Verify stdin.write was called with correct payload
        mock_process.stdin.write.assert_called_once()
        payload = mock_process.stdin.write.call_args[0][0]
        # Check for method and query but in a more flexible way
        self.assertIn('brave-search.brave_web_search', payload)
        self.assertIn('test query', payload)
        
        # Clean up
        self.client.stop_server("brave-search")
    
    def test_stop_server(self):
        """Test stopping a server"""
        # Create a mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # process is running
        
        # Add to processes
        self.client.processes["test-server"] = mock_process
        
        # Stop server
        result = self.client.stop_server("test-server")
        
        # Check result
        self.assertTrue(result, "Failed to stop server")
        self.assertNotIn("test-server", self.client.processes)
        mock_process.terminate.assert_called_once()


class TestMCPManager(unittest.TestCase):
    """Test MCP Manager"""
    
    def setUp(self):
        """Set up test environment"""
        self.manager = MCPManager()
    
    def test_services_exist(self):
        """Test that services exist"""
        self.assertIsNotNone(self.manager.brave_search, "Brave Search service not created")
        self.assertIsNotNone(self.manager.memory, "Memory service not created")
        self.assertIsNotNone(self.manager.fetch, "Fetch service not created")
        self.assertIsNotNone(self.manager.github, "GitHub service not created")
    
    def test_brave_search_service(self):
        """Test Brave Search service"""
        # Mock the call_service method
        async def mock_call(*args, **kwargs):
            return {"web": {"results": [{"title": "Test Result"}]}}
            
        with patch.object(MCPClient, "call_service", side_effect=mock_call):
            # Call service
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.manager.brave_search.brave_web_search("test query")
            )
            
            # Check results
            self.assertIsInstance(result, dict)
            self.assertIn("web", result)
            self.assertIn("results", result["web"])
            self.assertEqual(result["web"]["results"][0]["title"], "Test Result")
        


if __name__ == "__main__":
    unittest.main()