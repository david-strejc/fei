"""
Base MCP interfaces for Fei.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union


class MCPServiceInterface(ABC):
    """Abstract interface for MCP services."""
    
    @abstractmethod
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the service.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    @abstractmethod
    async def call_method(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a method on the service.
        
        Args:
            method: Method name
            params: Method parameters
            
        Returns:
            Method result
        """
        pass
    
    @abstractmethod
    def get_methods(self) -> List[Dict[str, Any]]:
        """
        Get a list of available methods.
        
        Returns:
            List of method dictionaries with name, description, and parameters
        """
        pass


class MCPServerInterface(ABC):
    """Abstract interface for MCP servers."""
    
    @abstractmethod
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the server.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    @abstractmethod
    async def start(self) -> bool:
        """
        Start the server.
        
        Returns:
            True if started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop the server.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the server is running.
        
        Returns:
            True if running, False otherwise
        """
        pass
    
    @abstractmethod
    def get_services(self) -> List[str]:
        """
        Get a list of available services.
        
        Returns:
            List of service names
        """
        pass
    
    @abstractmethod
    def get_service(self, service_name: str) -> Optional[MCPServiceInterface]:
        """
        Get a service by name.
        
        Args:
            service_name: Service name
            
        Returns:
            Service instance or None if not found
        """
        pass


class MCPServiceRegistry:
    """Registry for MCP services."""
    
    _services: Dict[str, type] = {}
    
    @classmethod
    def register_service(cls, service_name: str, service_class: type) -> None:
        """
        Register a service class.
        
        Args:
            service_name: Name of the service
            service_class: Service class
        """
        cls._services[service_name] = service_class
    
    @classmethod
    def create_service(cls, service_name: str, **kwargs) -> MCPServiceInterface:
        """
        Create a service instance.
        
        Args:
            service_name: Name of the service
            **kwargs: Additional arguments for the service
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If service is not registered
        """
        if service_name not in cls._services:
            raise ValueError(f"Service '{service_name}' not registered")
        
        return cls._services[service_name](**kwargs)
    
    @classmethod
    def list_services(cls) -> List[str]:
        """
        List all registered services.
        
        Returns:
            List of service names
        """
        return list(cls._services.keys())


class MCPServerRegistry:
    """Registry for MCP servers."""
    
    _servers: Dict[str, type] = {}
    
    @classmethod
    def register_server(cls, server_name: str, server_class: type) -> None:
        """
        Register a server class.
        
        Args:
            server_name: Name of the server
            server_class: Server class
        """
        cls._servers[server_name] = server_class
    
    @classmethod
    def create_server(cls, server_name: str, **kwargs) -> MCPServerInterface:
        """
        Create a server instance.
        
        Args:
            server_name: Name of the server
            **kwargs: Additional arguments for the server
            
        Returns:
            Server instance
            
        Raises:
            ValueError: If server is not registered
        """
        if server_name not in cls._servers:
            raise ValueError(f"Server '{server_name}' not registered")
        
        return cls._servers[server_name](**kwargs)
    
    @classmethod
    def list_servers(cls) -> List[str]:
        """
        List all registered servers.
        
        Returns:
            List of server names
        """
        return list(cls._servers.keys())
