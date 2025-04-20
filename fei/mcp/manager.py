"""
MCP manager for Fei.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union

from fei.mcp.base import MCPServiceInterface, MCPServerInterface, MCPServiceRegistry, MCPServerRegistry
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Exception raised for MCP connection errors."""
    pass


class MCPExecutionError(Exception):
    """Exception raised for MCP execution errors."""
    pass


class MCPManager:
    """Manager for MCP services and servers."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the MCP manager.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.servers: Dict[str, MCPServerInterface] = {}
        self.default_server: Optional[str] = None
        
        # Initialize servers from config
        self._initialize_servers()
    
    def _initialize_servers(self) -> None:
        """Initialize servers from configuration."""
        servers_config = self.config.get("mcp_servers", {})
        
        for server_id, server_config in servers_config.items():
            try:
                server_type = server_config.get("type", "stdio")
                server = MCPServerRegistry.create_server(server_type, config=server_config)
                server.setup(server_config)
                self.servers[server_id] = server
                
                # Set as default if specified or if it's the first server
                if server_config.get("default", False) or not self.default_server:
                    self.default_server = server_id
            except Exception as e:
                logger.error(f"Failed to initialize MCP server '{server_id}': {e}")
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """
        List all registered servers.
        
        Returns:
            List of server dictionaries with id, type, and status
        """
        result = []
        for server_id, server in self.servers.items():
            result.append({
                "id": server_id,
                "type": server.__class__.__name__,
                "running": server.is_running(),
                "is_default": server_id == self.default_server
            })
        return result
    
    def list_available_server_types(self) -> List[str]:
        """
        List all available server types.
        
        Returns:
            List of server type names
        """
        return MCPServerRegistry.list_servers()
    
    def list_available_service_types(self) -> List[str]:
        """
        List all available service types.
        
        Returns:
            List of service type names
        """
        return MCPServiceRegistry.list_services()
    
    async def add_server(self, server_id: str, server_type: str, config: Dict[str, Any]) -> bool:
        """
        Add a new server.
        
        Args:
            server_id: Server ID
            server_type: Server type
            config: Server configuration
            
        Returns:
            True if added successfully, False otherwise
        """
        if server_id in self.servers:
            logger.error(f"Server '{server_id}' already exists")
            return False
        
        try:
            server = MCPServerRegistry.create_server(server_type, config=config)
            server.setup(config)
            self.servers[server_id] = server
            
            # Set as default if specified or if it's the first server
            if config.get("default", False) or not self.default_server:
                self.default_server = server_id
            
            # Start the server if auto_start is specified
            if config.get("auto_start", False):
                await server.start()
            
            return True
        except Exception as e:
            logger.error(f"Failed to add MCP server '{server_id}': {e}")
            return False
    
    async def remove_server(self, server_id: str) -> bool:
        """
        Remove a server.
        
        Args:
            server_id: Server ID
            
        Returns:
            True if removed successfully, False otherwise
        """
        if server_id not in self.servers:
            logger.error(f"Server '{server_id}' not found")
            return False
        
        try:
            server = self.servers[server_id]
            
            # Stop the server if it's running
            if server.is_running():
                await server.stop()
            
            # Remove from servers dictionary
            del self.servers[server_id]
            
            # Update default server if needed
            if server_id == self.default_server:
                self.default_server = next(iter(self.servers)) if self.servers else None
            
            return True
        except Exception as e:
            logger.error(f"Failed to remove MCP server '{server_id}': {e}")
            return False
    
    def set_default_server(self, server_id: str) -> bool:
        """
        Set the default server.
        
        Args:
            server_id: Server ID
            
        Returns:
            True if set successfully, False otherwise
        """
        if server_id not in self.servers:
            logger.error(f"Server '{server_id}' not found")
            return False
        
        self.default_server = server_id
        return True
    
    async def start_server(self, server_id: str) -> bool:
        """
        Start a server.
        
        Args:
            server_id: Server ID
            
        Returns:
            True if started successfully, False otherwise
        """
        if server_id not in self.servers:
            logger.error(f"Server '{server_id}' not found")
            return False
        
        try:
            server = self.servers[server_id]
            return await server.start()
        except Exception as e:
            logger.error(f"Failed to start MCP server '{server_id}': {e}")
            return False
    
    async def stop_server(self, server_id: str) -> bool:
        """
        Stop a server.
        
        Args:
            server_id: Server ID
            
        Returns:
            True if stopped successfully, False otherwise
        """
        if server_id not in self.servers:
            logger.error(f"Server '{server_id}' not found")
            return False
        
        try:
            server = self.servers[server_id]
            return await server.stop()
        except Exception as e:
            logger.error(f"Failed to stop MCP server '{server_id}': {e}")
            return False
    
    async def stop_all_servers(self) -> Dict[str, bool]:
        """
        Stop all servers.
        
        Returns:
            Dictionary of server IDs and stop results
        """
        results = {}
        for server_id in self.servers:
            results[server_id] = await self.stop_server(server_id)
        return results
    
    def get_server_services(self, server_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get services available on a server.
        
        Args:
            server_id: Optional server ID, defaults to default server
            
        Returns:
            Dictionary of service names and method lists
            
        Raises:
            MCPConnectionError: If server is not found or not running
        """
        server_id = server_id or self.default_server
        
        if not server_id:
            raise MCPConnectionError("No default server configured")
        
        if server_id not in self.servers:
            raise MCPConnectionError(f"Server '{server_id}' not found")
        
        server = self.servers[server_id]
        
        if not server.is_running():
            raise MCPConnectionError(f"Server '{server_id}' is not running")
        
        result = {}
        for service_name in server.get_services():
            service = server.get_service(service_name)
            if service:
                result[service_name] = service.get_methods()
        
        return result
    
    async def call_service(self, service_name: str, method_name: str, params: Dict[str, Any], server_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Call a service method.
        
        Args:
            service_name: Service name
            method_name: Method name
            params: Method parameters
            server_id: Optional server ID, defaults to default server
            
        Returns:
            Method result
            
        Raises:
            MCPConnectionError: If server is not found or not running
            MCPExecutionError: If method execution fails
        """
        server_id = server_id or self.default_server
        
        if not server_id:
            raise MCPConnectionError("No default server configured")
        
        if server_id not in self.servers:
            raise MCPConnectionError(f"Server '{server_id}' not found")
        
        server = self.servers[server_id]
        
        if not server.is_running():
            raise MCPConnectionError(f"Server '{server_id}' is not running")
        
        service = server.get_service(service_name)
        if not service:
            raise MCPExecutionError(f"Service '{service_name}' not found on server '{server_id}'")
        
        try:
            return await service.call_method(method_name, params)
        except Exception as e:
            logger.error(f"Error calling method '{method_name}' on service '{service_name}': {e}")
            raise MCPExecutionError(f"Error calling method '{method_name}' on service '{service_name}': {str(e)}")
    
    def __getattr__(self, name: str) -> Any:
        """
        Get a service by name.
        
        This allows accessing services as attributes, e.g., mcp_manager.memory.
        
        Args:
            name: Service name
            
        Returns:
            Service proxy
            
        Raises:
            AttributeError: If service is not found
        """
        # Check if it's a valid service name
        if name in self.list_available_service_types():
            # Return a service proxy
            return MCPServiceProxy(self, name)
        
        # Not a service, raise AttributeError
        raise AttributeError(f"'MCPManager' object has no attribute '{name}'")


class MCPServiceProxy:
    """Proxy for MCP services."""
    
    def __init__(self, manager: MCPManager, service_name: str):
        """
        Initialize the service proxy.
        
        Args:
            manager: MCP manager
            service_name: Service name
        """
        self.manager = manager
        self.service_name = service_name
    
    def __getattr__(self, name: str) -> Any:
        """
        Get a method by name.
        
        This allows calling methods as attributes, e.g., mcp_manager.memory.search().
        
        Args:
            name: Method name
            
        Returns:
            Method proxy
            
        Raises:
            AttributeError: If method is not found
        """
        # Return a method proxy
        return MCPMethodProxy(self.manager, self.service_name, name)


class MCPMethodProxy:
    """Proxy for MCP service methods."""
    
    def __init__(self, manager: MCPManager, service_name: str, method_name: str):
        """
        Initialize the method proxy.
        
        Args:
            manager: MCP manager
            service_name: Service name
            method_name: Method name
        """
        self.manager = manager
        self.service_name = service_name
        self.method_name = method_name
    
    async def __call__(self, **kwargs) -> Dict[str, Any]:
        """
        Call the method.
        
        Args:
            **kwargs: Method parameters
            
        Returns:
            Method result
        """
        return await self.manager.call_service(self.service_name, self.method_name, kwargs)
