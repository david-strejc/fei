"""
Tests for the MCP system
"""

import unittest
from unittest.mock import MagicMock, patch
import asyncio

from fei.mcp.base import MCPServiceInterface, MCPServerInterface, MCPServiceRegistry, MCPServerRegistry
from fei.mcp.manager import MCPManager, MCPConnectionError, MCPExecutionError
from fei.utils.config import Config


class TestMCPSystem(unittest.TestCase):
    """Test the MCP system"""
    
    def test_service_registry(self):
        """Test service registry"""
        # Clear existing services
        MCPServiceRegistry._services = {}
        
        # Register a mock service
        class MockService(MCPServiceInterface):
            def __init__(self, **kwargs):
                self.config = kwargs.get("config", {})
            
            def setup(self, config=None):
                self.config = config or {}
            
            async def call_method(self, method, params):
                return {"result": f"Called {method} with {params}"}
            
            def get_methods(self):
                return [
                    {
                        "name": "test_method",
                        "description": "Test method",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "param1": {"type": "string"}
                            }
                        }
                    }
                ]
        
        MCPServiceRegistry.register_service("mock", MockService)
        
        # Check if service is registered
        self.assertIn("mock", MCPServiceRegistry.list_services())
        
        # Create service instance
        service = MCPServiceRegistry.create_service("mock", config={"test": "value"})
        self.assertIsInstance(service, MockService)
        self.assertEqual(service.config, {"test": "value"})
    
    def test_server_registry(self):
        """Test server registry"""
        # Clear existing servers
        MCPServerRegistry._servers = {}
        
        # Register a mock server
        class MockServer(MCPServerInterface):
            def __init__(self, **kwargs):
                self.config = kwargs.get("config", {})
                self.running = False
                self.services = {}
            
            def setup(self, config=None):
                self.config = config or {}
            
            async def start(self):
                self.running = True
                return True
            
            async def stop(self):
                self.running = False
                return True
            
            def is_running(self):
                return self.running
            
            def get_services(self):
                return list(self.services.keys())
            
            def get_service(self, service_name):
                return self.services.get(service_name)
        
        MCPServerRegistry.register_server("mock", MockServer)
        
        # Check if server is registered
        self.assertIn("mock", MCPServerRegistry.list_servers())
        
        # Create server instance
        server = MCPServerRegistry.create_server("mock", config={"test": "value"})
        self.assertIsInstance(server, MockServer)
        self.assertEqual(server.config, {"test": "value"})
    
    def test_mcp_manager(self):
        """Test MCP manager"""
        # Create a mock server class
        class MockServer(MCPServerInterface):
            def __init__(self, **kwargs):
                self.config = kwargs.get("config", {})
                self.running = False
                self.services = {}
            
            def setup(self, config=None):
                self.config = config or {}
                
                # Add a mock service
                class MockService(MCPServiceInterface):
                    def setup(self, config=None):
                        pass
                    
                    async def call_method(self, method, params):
                        if method == "test_method":
                            return {"result": params.get("param1", "")}
                        raise ValueError(f"Unknown method: {method}")
                    
                    def get_methods(self):
                        return [
                            {
                                "name": "test_method",
                                "description": "Test method",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "param1": {"type": "string"}
                                    }
                                }
                            }
                        ]
                
                self.services["mock_service"] = MockService()
            
            async def start(self):
                self.running = True
                return True
            
            async def stop(self):
                self.running = False
                return True
            
            def is_running(self):
                return self.running
            
            def get_services(self):
                return list(self.services.keys())
            
            def get_service(self, service_name):
                return self.services.get(service_name)
        
        # Register the mock server
        MCPServerRegistry._servers = {}
        MCPServerRegistry.register_server("mock", MockServer)
        
        # Create a config with a mock server
        config = Config()
        config.set("mcp_servers", {
            "test_server": {
                "type": "mock",
                "default": True,
                "auto_start": True
            }
        })
        
        # Create MCP manager
        manager = MCPManager(config)
        
        # Check if server is registered
        servers = manager.list_servers()
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0]["id"], "test_server")
        self.assertEqual(servers[0]["is_default"], True)
        
        # Start the server
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(manager.start_server("test_server"))
        self.assertTrue(result)
        
        # Get server services
        services = manager.get_server_services()
        self.assertIn("mock_service", services)
        
        # Call a service method
        result = loop.run_until_complete(manager.call_service("mock_service", "test_method", {"param1": "test"}))
        self.assertEqual(result["result"], "test")
        
        # Stop the server
        result = loop.run_until_complete(manager.stop_server("test_server"))
        self.assertTrue(result)
        
        # Try to call a service method when server is stopped
        with self.assertRaises(MCPConnectionError):
            loop.run_until_complete(manager.call_service("mock_service", "test_method", {"param1": "test"}))
    
    def test_service_proxy(self):
        """Test service proxy"""
        # Create a mock server class
        class MockServer(MCPServerInterface):
            def __init__(self, **kwargs):
                self.config = kwargs.get("config", {})
                self.running = True
                self.services = {}
                
                # Add a mock service
                class MockService(MCPServiceInterface):
                    def setup(self, config=None):
                        pass
                    
                    async def call_method(self, method, params):
                        if method == "test_method":
                            return {"result": params.get("param1", "")}
                        raise ValueError(f"Unknown method: {method}")
                    
                    def get_methods(self):
                        return [
                            {
                                "name": "test_method",
                                "description": "Test method",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "param1": {"type": "string"}
                                    }
                                }
                            }
                        ]
                
                self.services["mock_service"] = MockService()
            
            def setup(self, config=None):
                self.config = config or {}
            
            async def start(self):
                self.running = True
                return True
            
            async def stop(self):
                self.running = False
                return True
            
            def is_running(self):
                return self.running
            
            def get_services(self):
                return list(self.services.keys())
            
            def get_service(self, service_name):
                return self.services.get(service_name)
        
        # Register the mock server and service
        MCPServerRegistry._servers = {}
        MCPServerRegistry.register_server("mock", MockServer)
        
        MCPServiceRegistry._services = {}
        MCPServiceRegistry.register_service("mock_service", MagicMock())
        
        # Create a config with a mock server
        config = Config()
        config.set("mcp_servers", {
            "test_server": {
                "type": "mock",
                "default": True,
                "auto_start": True
            }
        })
        
        # Create MCP manager
        manager = MCPManager(config)
        
        # Access service as attribute
        service_proxy = manager.mock_service
        
        # Call method using proxy
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(service_proxy.test_method(param1="test"))
        self.assertEqual(result["result"], "test")


if __name__ == "__main__":
    unittest.main()
