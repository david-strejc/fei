"""
Tests for the MCP base interfaces.
"""

import pytest
from unittest.mock import MagicMock

from fei.mcp.base import MCPServiceInterface, MCPServerInterface, MCPServiceRegistry, MCPServerRegistry


class TestMCPServiceRegistry:
    """Test the MCP service registry."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing services
        MCPServiceRegistry._services = {}
    
    def test_register_service(self):
        """Test registering a service."""
        # Create a mock service class
        mock_service_class = MagicMock()
        
        # Register the service
        MCPServiceRegistry.register_service("mock", mock_service_class)
        
        # Check if service is registered
        assert "mock" in MCPServiceRegistry.list_services()
    
    def test_create_service(self):
        """Test creating a service."""
        # Create a mock service class
        mock_service_class = MagicMock()
        
        # Register the service
        MCPServiceRegistry.register_service("mock", mock_service_class)
        
        # Create a service instance
        service = MCPServiceRegistry.create_service("mock", config={"test": "value"})
        
        # Check if service was created with correct arguments
        mock_service_class.assert_called_once_with(config={"test": "value"})
        assert service is mock_service_class.return_value
    
    def test_create_nonexistent_service(self):
        """Test creating a non-existent service."""
        # Try to create a non-existent service
        with pytest.raises(ValueError):
            MCPServiceRegistry.create_service("nonexistent")
    
    def test_list_services(self):
        """Test listing services."""
        # Register some services
        MCPServiceRegistry.register_service("mock1", MagicMock())
        MCPServiceRegistry.register_service("mock2", MagicMock())
        
        # List services
        services = MCPServiceRegistry.list_services()
        
        # Check if all services are listed
        assert "mock1" in services
        assert "mock2" in services
        assert len(services) == 2


class TestMCPServerRegistry:
    """Test the MCP server registry."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing servers
        MCPServerRegistry._servers = {}
    
    def test_register_server(self):
        """Test registering a server."""
        # Create a mock server class
        mock_server_class = MagicMock()
        
        # Register the server
        MCPServerRegistry.register_server("mock", mock_server_class)
        
        # Check if server is registered
        assert "mock" in MCPServerRegistry.list_servers()
    
    def test_create_server(self):
        """Test creating a server."""
        # Create a mock server class
        mock_server_class = MagicMock()
        
        # Register the server
        MCPServerRegistry.register_server("mock", mock_server_class)
        
        # Create a server instance
        server = MCPServerRegistry.create_server("mock", config={"test": "value"})
        
        # Check if server was created with correct arguments
        mock_server_class.assert_called_once_with(config={"test": "value"})
        assert server is mock_server_class.return_value
    
    def test_create_nonexistent_server(self):
        """Test creating a non-existent server."""
        # Try to create a non-existent server
        with pytest.raises(ValueError):
            MCPServerRegistry.create_server("nonexistent")
    
    def test_list_servers(self):
        """Test listing servers."""
        # Register some servers
        MCPServerRegistry.register_server("mock1", MagicMock())
        MCPServerRegistry.register_server("mock2", MagicMock())
        
        # List servers
        servers = MCPServerRegistry.list_servers()
        
        # Check if all servers are listed
        assert "mock1" in servers
        assert "mock2" in servers
        assert len(servers) == 2
