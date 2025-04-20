"""
Tests for the MCP manager.
"""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from fei.mcp.base import MCPServiceInterface, MCPServerInterface, MCPServiceRegistry, MCPServerRegistry
from fei.mcp.manager import MCPManager, MCPConnectionError, MCPExecutionError
from fei.utils.config import Config


class TestMCPManager:
    """Test the MCP manager."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing servers and services
        MCPServerRegistry._servers = {}
        MCPServiceRegistry._services = {}
        
        # Create a mock server class
        self.mock_server_class = MagicMock()
        self.mock_server = MagicMock(spec=MCPServerInterface)
        self.mock_server_class.return_value = self.mock_server
        
        # Set up mock server methods
        self.mock_server.is_running.return_value = True
        self.mock_server.get_services.return_value = ["mock_service"]
        
        async def start():
            return True
        
        async def stop():
            return True
        
        self.mock_server.start = MagicMock(side_effect=start)
        self.mock_server.stop = MagicMock(side_effect=stop)
        
        # Create a mock service
        self.mock_service = MagicMock(spec=MCPServiceInterface)
        self.mock_server.get_service.return_value = self.mock_service
        
        # Set up mock service methods
        self.mock_service.get_methods.return_value = [
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
        
        async def call_method(method, params):
            return {"result": f"Called {method} with {params}"}
        
        self.mock_service.call_method = MagicMock(side_effect=call_method)
        
        # Register the mock server
        MCPServerRegistry.register_server("mock", self.mock_server_class)
        
        # Register a mock service
        MCPServiceRegistry.register_service("mock_service", MagicMock())
        
        # Create a config with a mock server
        self.config = Config()
        self.config.set("mcp_servers", {
            "test_server": {
                "type": "mock",
                "default": True,
                "auto_start": True
            }
        })
        
        # Create MCP manager
        self.manager = MCPManager(self.config)
    
    def test_init(self):
        """Test initialization."""
        # Check if manager is initialized correctly
        assert "test_server" in self.manager.servers
        assert self.manager.default_server == "test_server"
        self.mock_server_class.assert_called_once_with(config={"type": "mock", "default": True, "auto_start": True})
        self.mock_server.setup.assert_called_once_with({"type": "mock", "default": True, "auto_start": True})
    
    def test_list_servers(self):
        """Test list_servers."""
        # List servers
        servers = self.manager.list_servers()
        
        # Check if servers are correct
        assert len(servers) == 1
        assert servers[0]["id"] == "test_server"
        assert servers[0]["is_default"] is True
        assert servers[0]["running"] is True
    
    def test_list_available_server_types(self):
        """Test list_available_server_types."""
        # List available server types
        server_types = self.manager.list_available_server_types()
        
        # Check if server types are correct
        assert "mock" in server_types
    
    def test_list_available_service_types(self):
        """Test list_available_service_types."""
        # List available service types
        service_types = self.manager.list_available_service_types()
        
        # Check if service types are correct
        assert "mock_service" in service_types
    
    @pytest.mark.asyncio
    async def test_add_server(self):
        """Test add_server."""
        # Add a server
        result = await self.manager.add_server("new_server", "mock", {"test": "value"})
        
        # Check if server was added
        assert result is True
        assert "new_server" in self.manager.servers
        
        # Check if server was created with correct arguments
        self.mock_server_class.assert_called_with(config={"test": "value"})
        
        # Check if server was set up
        self.mock_server.setup.assert_called_with({"test": "value"})
    
    @pytest.mark.asyncio
    async def test_add_existing_server(self):
        """Test add_server with existing server."""
        # Try to add an existing server
        result = await self.manager.add_server("test_server", "mock", {"test": "value"})
        
        # Check if server was not added
        assert result is False
    
    @pytest.mark.asyncio
    async def test_remove_server(self):
        """Test remove_server."""
        # Remove a server
        result = await self.manager.remove_server("test_server")
        
        # Check if server was removed
        assert result is True
        assert "test_server" not in self.manager.servers
        
        # Check if server was stopped
        self.mock_server.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_nonexistent_server(self):
        """Test remove_server with non-existent server."""
        # Try to remove a non-existent server
        result = await self.manager.remove_server("nonexistent")
        
        # Check if server was not removed
        assert result is False
    
    def test_set_default_server(self):
        """Test set_default_server."""
        # Add another server
        self.manager.servers["new_server"] = MagicMock()
        
        # Set default server
        result = self.manager.set_default_server("new_server")
        
        # Check if default server was set
        assert result is True
        assert self.manager.default_server == "new_server"
    
    def test_set_nonexistent_default_server(self):
        """Test set_default_server with non-existent server."""
        # Try to set a non-existent server as default
        result = self.manager.set_default_server("nonexistent")
        
        # Check if default server was not set
        assert result is False
        assert self.manager.default_server == "test_server"
    
    @pytest.mark.asyncio
    async def test_start_server(self):
        """Test start_server."""
        # Start a server
        result = await self.manager.start_server("test_server")
        
        # Check if server was started
        assert result is True
        self.mock_server.start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_nonexistent_server(self):
        """Test start_server with non-existent server."""
        # Try to start a non-existent server
        result = await self.manager.start_server("nonexistent")
        
        # Check if server was not started
        assert result is False
    
    @pytest.mark.asyncio
    async def test_stop_server(self):
        """Test stop_server."""
        # Stop a server
        result = await self.manager.stop_server("test_server")
        
        # Check if server was stopped
        assert result is True
        self.mock_server.stop.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_nonexistent_server(self):
        """Test stop_server with non-existent server."""
        # Try to stop a non-existent server
        result = await self.manager.stop_server("nonexistent")
        
        # Check if server was not stopped
        assert result is False
    
    @pytest.mark.asyncio
    async def test_stop_all_servers(self):
        """Test stop_all_servers."""
        # Add another server
        mock_server2 = MagicMock(spec=MCPServerInterface)
        
        async def stop2():
            return True
        
        mock_server2.stop = MagicMock(side_effect=stop2)
        self.manager.servers["new_server"] = mock_server2
        
        # Stop all servers
        results = await self.manager.stop_all_servers()
        
        # Check if all servers were stopped
        assert results == {"test_server": True, "new_server": True}
        self.mock_server.stop.assert_called_once()
        mock_server2.stop.assert_called_once()
    
    def test_get_server_services(self):
        """Test get_server_services."""
        # Get server services
        services = self.manager.get_server_services()
        
        # Check if services are correct
        assert "mock_service" in services
        assert services["mock_service"] == self.mock_service.get_methods.return_value
    
    def test_get_server_services_nonexistent_server(self):
        """Test get_server_services with non-existent server."""
        # Try to get services from a non-existent server
        with pytest.raises(MCPConnectionError):
            self.manager.get_server_services("nonexistent")
    
    def test_get_server_services_not_running(self):
        """Test get_server_services with server not running."""
        # Set server to not running
        self.mock_server.is_running.return_value = False
        
        # Try to get services
        with pytest.raises(MCPConnectionError):
            self.manager.get_server_services()
    
    @pytest.mark.asyncio
    async def test_call_service(self):
        """Test call_service."""
        # Call a service method
        result = await self.manager.call_service("mock_service", "test_method", {"param1": "test"})
        
        # Check if method was called with correct arguments
        self.mock_service.call_method.assert_called_once_with("test_method", {"param1": "test"})
        
        # Check if result is correct
        assert result == {"result": "Called test_method with {'param1': 'test'}"}
    
    @pytest.mark.asyncio
    async def test_call_service_nonexistent_server(self):
        """Test call_service with non-existent server."""
        # Try to call a service on a non-existent server
        with pytest.raises(MCPConnectionError):
            await self.manager.call_service("mock_service", "test_method", {}, server_id="nonexistent")
    
    @pytest.mark.asyncio
    async def test_call_service_not_running(self):
        """Test call_service with server not running."""
        # Set server to not running
        self.mock_server.is_running.return_value = False
        
        # Try to call a service
        with pytest.raises(MCPConnectionError):
            await self.manager.call_service("mock_service", "test_method", {})
    
    @pytest.mark.asyncio
    async def test_call_nonexistent_service(self):
        """Test call_service with non-existent service."""
        # Set server to return None for service
        self.mock_server.get_service.return_value = None
        
        # Try to call a non-existent service
        with pytest.raises(MCPExecutionError):
            await self.manager.call_service("nonexistent", "test_method", {})
    
    @pytest.mark.asyncio
    async def test_service_proxy(self):
        """Test service proxy."""
        # Get service proxy
        service_proxy = self.manager.mock_service
        
        # Call method using proxy
        result = await service_proxy.test_method(param1="test")
        
        # Check if method was called with correct arguments
        self.mock_service.call_method.assert_called_once_with("test_method", {"param1": "test"})
        
        # Check if result is correct
        assert result == {"result": "Called test_method with {'param1': 'test'}"}
    
    def test_nonexistent_service_proxy(self):
        """Test non-existent service proxy."""
        # Try to get a non-existent service proxy
        with pytest.raises(AttributeError):
            self.manager.nonexistent
