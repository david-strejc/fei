#!/usr/bin/env python3
"""
MCP Servers integration for Fei code assistant

This module provides integration with MCP (Model Control Protocol) servers
for enhanced capabilities.
"""

import os
import json
import time
import asyncio
import requests
import subprocess
import urllib.parse
import signal
import ssl
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from contextlib import asynccontextmanager

from fei.utils.logging import get_logger
from fei.utils.config import Config

logger = get_logger(__name__)

class MCPServerConfigError(Exception):
    """Exception raised for MCP server configuration errors"""
    pass

class MCPConnectionError(Exception):
    """Exception raised for MCP connection errors"""
    pass

class MCPExecutionError(Exception):
    """Exception raised for MCP execution errors"""
    pass

class ProcessManager:
    """Manager for child processes"""
    
    def __init__(self):
        """Initialize process manager"""
        self.processes = {}
        self._lock = asyncio.Lock()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_all)
    
    async def start_process(
        self, 
        process_id: str, 
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> subprocess.Popen:
        """
        Start a process safely
        
        Args:
            process_id: Unique identifier for the process
            command: Command list to execute
            env: Environment variables
            **kwargs: Additional arguments for subprocess.Popen
            
        Returns:
            Process instance
            
        Raises:
            ValueError: If the command is invalid
            OSError: If the process cannot be started
        """
        # Validate command
        if not command or not isinstance(command, list):
            raise ValueError("Command must be a non-empty list")
        
        # Use lock to prevent race conditions
        async with self._lock:
            # Check if process already exists and is running
            if process_id in self.processes and self.processes[process_id].poll() is None:
                return self.processes[process_id]
            
            # Set up process environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)
            
            # Add default arguments for safety and proper handling
            default_kwargs = {
                "stdin": subprocess.PIPE,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
                "start_new_session": True  # Create a new process group for proper cleanup
            }
            
            # Override defaults with provided kwargs
            process_kwargs = {**default_kwargs, **kwargs}
            
            try:
                # Start the process
                process = subprocess.Popen(command, env=process_env, **process_kwargs)
                
                # Store the process
                self.processes[process_id] = process
                
                logger.info(f"Started process {process_id} (PID: {process.pid})")
                return process
                
            except (OSError, subprocess.SubprocessError) as e:
                logger.error(f"Failed to start process {process_id}: {e}")
                raise OSError(f"Failed to start process {process_id}: {e}")
    
    async def stop_process(self, process_id: str, timeout: float = 3.0) -> bool:
        """
        Stop a process safely
        
        Args:
            process_id: Process identifier
            timeout: Timeout in seconds
            
        Returns:
            Whether the process was stopped successfully
        """
        async with self._lock:
            if process_id not in self.processes:
                return False
                
            process = self.processes[process_id]
            
            # Check if already terminated
            if process.poll() is not None:
                del self.processes[process_id]
                return True
            
            try:
                # Try to terminate gracefully
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait for the process to terminate
                for _ in range(int(timeout * 10)):  # 10 checks per second
                    if process.poll() is not None:
                        break
                    await asyncio.sleep(0.1)
                    
                # If still running, force kill
                if process.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                    
                    # Wait a bit more
                    for _ in range(5):
                        if process.poll() is not None:
                            break
                        await asyncio.sleep(0.1)
                
                # Clean up
                del self.processes[process_id]
                logger.info(f"Stopped process {process_id}")
                return True
                
            except (OSError, subprocess.SubprocessError) as e:
                logger.error(f"Error stopping process {process_id}: {e}")
                
                # Still remove from our tracking if we can't manage it
                if process_id in self.processes:
                    del self.processes[process_id]
                    
                return False
    
    def cleanup_all(self):
        """Clean up all running processes"""
        # Create a new event loop for the cleanup if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Get a copy of process IDs to avoid modification during iteration
        process_ids = list(self.processes.keys())
        
        for process_id in process_ids:
            # Use run_until_complete for the cleanup
            try:
                loop.run_until_complete(self.stop_process(process_id))
            except Exception as e:
                logger.error(f"Error during cleanup of process {process_id}: {e}")


class MCPClient:
    """Client for MCP servers"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize MCP client
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        
        # Initialize process manager
        self.process_manager = ProcessManager()
        
        # Get server configurations
        self.servers = self._load_servers()
        
        # Set default server
        self.default_server = self.config.get("mcp.default_server")
        
        # SSL context for secure requests
        self.ssl_context = self._create_ssl_context()
        
        # HTTP session for connection pooling
        self.session = requests.Session()
        # Set default timeout and headers
        self.session.timeout = 30
        
        logger.debug(f"Initialized MCP client with {len(self.servers)} servers")
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Create a secure SSL context for HTTPS requests
        
        Returns:
            SSL context
        """
        context = ssl.create_default_context()
        # Always verify SSL certificates
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        
        # Use system CA certificates
        context.load_default_certs()
        
        return context
    
    def _load_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Load server configurations
        
        Returns:
            Dictionary of server configurations
        """
        servers = {}
        
        # Load from config
        server_section = self.config.get_section("mcp.servers")
        if server_section:
            for server_id, server_config in server_section.items():
                # Check if it's a string (old format with just URL)
                if isinstance(server_config, str):
                    # Validate URL
                    if self._validate_url(server_config):
                        servers[server_id] = {"url": server_config, "type": "http"}
                else:
                    # It's a dictionary
                    if server_config.get("type") == "http":
                        # Validate URL
                        url = server_config.get("url")
                        if url and self._validate_url(url):
                            servers[server_id] = server_config
                    else:
                        # Other types (stdio, etc.)
                        servers[server_id] = server_config
        
        # Load from environment variables
        for env_var, value in os.environ.items():
            if env_var.startswith("FEI_MCP_SERVER_"):
                server_id = env_var[15:].lower()
                # Validate URL
                if self._validate_url(value):
                    servers[server_id] = {"url": value, "type": "http"}
        
        # Add brave-search server if not already defined
        if "brave-search" not in servers:
            # Get BRAVE_API_KEY from environment or config, never use default/hardcoded
            brave_api_key = (
                self.config.get("brave.api_key") or 
                os.environ.get("BRAVE_API_KEY")
            )
            
            # Only define server if we have an API key
            if brave_api_key:
                servers["brave-search"] = {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {
                        "BRAVE_API_KEY": brave_api_key
                    }
                }
            
        return servers
    
    def _validate_url(self, url: str) -> bool:
        """
        Validate a URL
        
        Args:
            url: URL to validate
            
        Returns:
            Whether the URL is valid
        """
        try:
            result = urllib.parse.urlparse(url)
            # Basic validation - must have scheme and netloc
            valid = all([result.scheme in ['http', 'https'], result.netloc])
            
            # Reject obviously dangerous URLs
            dangerous_patterns = ["file://", "ftp://", "data:"]
            if any(pattern in url for pattern in dangerous_patterns):
                logger.warning(f"Rejected dangerous URL: {url}")
                return False
                
            return valid
        except Exception:
            return False
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """
        List available MCP servers
        
        Returns:
            List of server information
        """
        result = []
        for server_id, config in self.servers.items():
            server_info = {"id": server_id, "type": config.get("type", "http")}
            
            if config.get("type") == "stdio":
                server_info["command"] = config.get("command")
                server_info["args"] = config.get("args", [])
                
                # Check if process is running using asyncio run
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        process = self.process_manager.processes.get(server_id)
                        server_info["status"] = "running" if process and process.poll() is None else "stopped"
                    else:
                        # Can't check properly in running loop, make best guess
                        server_info["status"] = "unknown"
                except Exception:
                    server_info["status"] = "unknown"
            else:
                # Sanitize URL by removing API keys if present
                url = config.get("url", "")
                parsed = urllib.parse.urlparse(url)
                if parsed.username or parsed.password:
                    # Redact auth info
                    url = urllib.parse.urlunparse((
                        parsed.scheme, 
                        f"***:***@{parsed.netloc.split('@')[-1]}" if '@' in parsed.netloc else parsed.netloc,
                        parsed.path, 
                        parsed.params, 
                        parsed.query, 
                        parsed.fragment
                    ))
                
                server_info["url"] = url
                
            result.append(server_info)
            
        return result
    
    def get_server(self, server_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get server configuration
        
        Args:
            server_id: Server ID (defaults to default server)
            
        Returns:
            Server configuration or None if not found
        """
        if not server_id:
            server_id = self.default_server
        
        if not server_id or server_id not in self.servers:
            return None
        
        return self.servers[server_id]
    
    def add_server(self, server_id: str, url: str) -> bool:
        """
        Add a new MCP server
        
        Args:
            server_id: Server ID
            url: Server URL
            
        Returns:
            Whether the server was added
            
        Raises:
            ValueError: If the server ID is invalid or the URL is invalid
        """
        # Validate server ID
        if not server_id or not server_id.isalnum():
            raise ValueError("Server ID must be alphanumeric")
            
        # Check for existing server
        if server_id in self.servers:
            return False
        
        # Validate URL
        if not self._validate_url(url):
            raise ValueError(f"Invalid URL: {url}")
        
        # Add server
        self.servers[server_id] = {"url": url, "type": "http"}
        
        # Save to config
        self.config.set(f"mcp.servers.{server_id}", url)
        
        return True
    
    def remove_server(self, server_id: str) -> bool:
        """
        Remove an MCP server
        
        Args:
            server_id: Server ID
            
        Returns:
            Whether the server was removed
        """
        if server_id not in self.servers:
            return False
        
        # Stop server if running
        if server_id in self.process_manager.processes:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                loop.run_until_complete(self.process_manager.stop_process(server_id))
            else:
                # Can't stop properly in running loop, log warning
                logger.warning(f"Cannot stop server {server_id} in running event loop")
        
        # Remove from servers
        del self.servers[server_id]
        
        # Remove from config
        self.config.delete(f"mcp.servers.{server_id}")
        
        return True
    
    def set_default_server(self, server_id: str) -> bool:
        """
        Set default MCP server
        
        Args:
            server_id: Server ID
            
        Returns:
            Whether the default server was set
        """
        if server_id not in self.servers:
            return False
        
        self.default_server = server_id
        self.config.set("mcp.default_server", server_id)
        
        return True
        
    async def stop_server(self, server_id: str) -> bool:
        """
        Stop a running MCP server
        
        Args:
            server_id: Server ID
            
        Returns:
            Whether the server was stopped
        """
        return await self.process_manager.stop_process(server_id)
    
    async def _start_stdio_server(self, server_id: str, config: Dict[str, Any]) -> None:
        """
        Start a stdio-based MCP server
        
        Args:
            server_id: Server ID
            config: Server configuration
            
        Raises:
            MCPServerConfigError: If server configuration is invalid
            OSError: If server fails to start
        """
        # Validate configuration
        command = config.get("command")
        args = config.get("args", [])
        env_vars = config.get("env", {})
        
        if not command:
            raise MCPServerConfigError(f"Command not specified for MCP server: {server_id}")
        
        # Start the process
        cmd = [command] + args
        logger.info(f"Starting MCP server: {server_id} with command: {' '.join(cmd)}")
        
        try:
            await self.process_manager.start_process(server_id, cmd, env=env_vars)
            
            # Wait for server to be ready (simple wait for now)
            await asyncio.sleep(2)
            
            logger.info(f"MCP server started: {server_id}")
        except OSError as e:
            logger.error(f"Error starting MCP server {server_id}: {e}")
            raise MCPServerConfigError(f"Error starting MCP server {server_id}: {e}")
    
    @asynccontextmanager
    async def _ensure_server_running(self, server_id: str) -> None:
        """
        Ensure a server is running before making a request
        
        Args:
            server_id: Server ID
            
        Yields:
            Server config
            
        Raises:
            MCPServerConfigError: If server configuration is invalid
            MCPConnectionError: If server fails to start
        """
        server = self.get_server(server_id)
        if not server:
            raise MCPServerConfigError(f"MCP server not found: {server_id}")
        
        server_type = server.get("type", "http")
        
        if server_type == "stdio":
            # For stdio servers, ensure process is running
            process = self.process_manager.processes.get(server_id)
            if not process or process.poll() is not None:
                # Start the server
                await self._start_stdio_server(server_id, server)
        
        try:
            yield server
        except Exception as e:
            logger.error(f"Error during server operation: {e}")
            raise
    
    async def _call_stdio_service(
        self, 
        server_id: str, 
        service: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP service method via stdio
        
        Args:
            server_id: Server ID
            service: Service name
            method: Method name
            params: Method parameters
            
        Returns:
            Service response
            
        Raises:
            MCPConnectionError: If communication with server fails
            MCPExecutionError: If server returns an error
        """
        async with self._ensure_server_running(server_id) as _:
            process = self.process_manager.processes.get(server_id)
            if not process:
                raise MCPConnectionError(f"Process not found for server: {server_id}")
            
            # Build request payload
            payload = {
                "jsonrpc": "2.0",
                "method": f"{service}.{method}",
                "params": params or {},
                "id": 1
            }
            
            payload_str = json.dumps(payload) + "\n"
            
            logger.debug(f"Calling stdio MCP service: {service}.{method}")
            
            try:
                # Send request
                process.stdin.write(payload_str)
                process.stdin.flush()
                
                # Read response with timeout
                response_str = ""
                for _ in range(300):  # 30-second timeout (100ms intervals)
                    if process.stdout.readable():
                        response_str = process.stdout.readline()
                        if response_str:
                            break
                    await asyncio.sleep(0.1)
                
                if not response_str:
                    raise MCPConnectionError(f"Timeout waiting for response from MCP server: {server_id}")
                
                # Parse response
                try:
                    result = json.loads(response_str)
                except json.JSONDecodeError as e:
                    raise MCPConnectionError(f"Invalid JSON response from MCP server: {e}")
                
                if "error" in result:
                    error = result["error"]
                    error_msg = error.get("message", "Unknown error")
                    raise MCPExecutionError(f"MCP service error: {error_msg}")
                
                return result.get("result", {})
            
            except (json.JSONDecodeError, MCPConnectionError, MCPExecutionError) as e:
                # Re-raise specific exceptions
                raise
            except Exception as e:
                logger.error(f"Unexpected error in MCP service call: {e}")
                raise MCPConnectionError(f"Unexpected error in MCP service call: {e}")
    
    async def call_service(
        self, 
        service: str, 
        method: str, 
        params: Optional[Dict[str, Any]] = None,
        server_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call an MCP service method
        
        Args:
            service: Service name
            method: Method name
            params: Method parameters
            server_id: Server ID (defaults to default server)
            
        Returns:
            Service response
            
        Raises:
            MCPServerConfigError: If server configuration is invalid
            MCPConnectionError: If communication with server fails
            MCPExecutionError: If server returns an error
        """
        server_id = server_id or self.default_server
        if not server_id:
            raise MCPServerConfigError("No default MCP server specified")
        
        async with self._ensure_server_running(server_id) as server:
            server_type = server.get("type", "http")
            
            if server_type == "stdio":
                return await self._call_stdio_service(server_id, service, method, params or {})
            else:
                # HTTP server
                url = server.get("url")
                if not url:
                    raise MCPServerConfigError(f"URL not specified for HTTP MCP server: {server_id}")
                
                # Build request payload
                payload = {
                    "jsonrpc": "2.0",
                    "method": f"{service}.{method}",
                    "params": params or {},
                    "id": 1
                }
                
                logger.debug(f"Calling HTTP MCP service: {service}.{method}")
                
                # Use executor for blocking HTTP requests
                loop = asyncio.get_running_loop()
                try:
                    # Make request in a separate thread
                    response = await loop.run_in_executor(
                        None,
                        lambda: self.session.post(
                            url, 
                            json=payload,
                            timeout=30,
                            verify=True  # Always verify SSL certificates
                        )
                    )
                    
                    # Raise for HTTP errors
                    response.raise_for_status()
                    
                    # Parse response
                    try:
                        result = response.json()
                    except json.JSONDecodeError as e:
                        raise MCPConnectionError(f"Invalid JSON response from MCP server: {e}")
                    
                    if "error" in result:
                        error = result["error"]
                        error_msg = error.get("message", "Unknown error")
                        raise MCPExecutionError(f"MCP service error: {error_msg}")
                    
                    return result.get("result", {})
                
                except requests.exceptions.RequestException as e:
                    raise MCPConnectionError(f"MCP service request error: {e}")
                except (json.JSONDecodeError, MCPConnectionError, MCPExecutionError) as e:
                    # Re-raise specific exceptions
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in MCP service call: {e}")
                    raise MCPConnectionError(f"Unexpected error in MCP service call: {e}")

# Base class for all MCP services
class MCPBaseService:
    """Base class for MCP services"""
    
    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
        """
        Initialize MCP service
        
        Args:
            client: MCP client
            server_id: Server ID (defaults to default server)
        """
        self.client = client
        self.server_id = server_id
        self.service_name = self.__class__.__name__.replace("MCP", "").replace("Service", "").lower()
    
    async def call_method(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a service method
        
        Args:
            method: Method name
            params: Method parameters
            
        Returns:
            Service response
        """
        return await self.client.call_service(
            self.service_name, 
            method,
            params,
            self.server_id
        )


class MCPMemoryService(MCPBaseService):
    """MCP Memory service"""
    
    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
        """Initialize MCP Memory service"""
        super().__init__(client, server_id)
        self.service_name = "memory"
    
    async def create_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create entities
        
        Args:
            entities: List of entities to create
            
        Returns:
            Service response
        """
        return await self.call_method("create_entities", {"entities": entities})
    
    async def create_relations(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create relations
        
        Args:
            relations: List of relations to create
            
        Returns:
            Service response
        """
        return await self.call_method("create_relations", {"relations": relations})
    
    async def add_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add observations
        
        Args:
            observations: List of observations to add
            
        Returns:
            Service response
        """
        return await self.call_method("add_observations", {"observations": observations})
    
    async def delete_entities(self, entity_names: List[str]) -> Dict[str, Any]:
        """
        Delete entities
        
        Args:
            entity_names: List of entity names to delete
            
        Returns:
            Service response
        """
        return await self.call_method("delete_entities", {"entityNames": entity_names})
    
    async def delete_observations(self, deletions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Delete observations
        
        Args:
            deletions: List of observations to delete
            
        Returns:
            Service response
        """
        return await self.call_method("delete_observations", {"deletions": deletions})
    
    async def delete_relations(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Delete relations
        
        Args:
            relations: List of relations to delete
            
        Returns:
            Service response
        """
        return await self.call_method("delete_relations", {"relations": relations})
    
    async def read_graph(self) -> Dict[str, Any]:
        """
        Read the entire knowledge graph
        
        Returns:
            Service response
        """
        return await self.call_method("read_graph", {})
    
    async def search_nodes(self, query: str) -> Dict[str, Any]:
        """
        Search for nodes
        
        Args:
            query: Search query
            
        Returns:
            Service response
        """
        return await self.call_method("search_nodes", {"query": query})
    
    async def open_nodes(self, names: List[str]) -> Dict[str, Any]:
        """
        Open nodes
        
        Args:
            names: List of node names
            
        Returns:
            Service response
        """
        return await self.call_method("open_nodes", {"names": names})


class MCPFetchService(MCPBaseService):
    """MCP Fetch service"""
    
    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
        """Initialize MCP Fetch service"""
        super().__init__(client, server_id)
        self.service_name = "fetch"
    
    async def fetch(
        self, 
        url: str, 
        max_length: int = 5000, 
        raw: bool = False, 
        start_index: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch a URL
        
        Args:
            url: URL to fetch
            max_length: Maximum length of content to return
            raw: Whether to return raw HTML
            start_index: Start index for content
            
        Returns:
            Service response
        """
        # Validate URL
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {url}")
        
        # Check for dangerous URLs
        dangerous_patterns = ["file://", "ftp://", "data:"]
        if any(pattern in url for pattern in dangerous_patterns):
            raise ValueError(f"Rejected dangerous URL: {url}")
        
        return await self.call_method("fetch", {
            "url": url,
            "max_length": max_length,
            "raw": raw,
            "start_index": start_index
        })


class MCPBraveSearchService(MCPBaseService):
    """MCP Brave Search service"""
    
    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
        """Initialize MCP Brave Search service"""
        super().__init__(client, server_id)
        self.service_name = "brave-search"
    
    async def brave_web_search(self, query: str, count: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Perform a web search
        
        Args:
            query: Search query
            count: Number of results
            offset: Offset for pagination
            
        Returns:
            Service response
        """
        # Validate parameters
        if not query:
            raise ValueError("Search query cannot be empty")
        
        if count < 1 or count > 20:
            count = min(max(1, count), 20)
            
        if offset < 0:
            offset = 0
            
        try:
            # Try via MCP service first
            return await self.call_method("brave_web_search", {
                "query": query,
                "count": count,
                "offset": offset
            })
        except Exception as e:
            logger.warning(f"MCP brave_web_search failed, falling back to direct API: {e}")
            
            # Fallback to direct API call
            return await self._direct_brave_api_call(query, count, offset)
    
    async def _direct_brave_api_call(self, query: str, count: int, offset: int) -> Dict[str, Any]:
        """
        Make a direct API call to Brave Search
        
        Args:
            query: Search query
            count: Number of results
            offset: Offset for pagination
            
        Returns:
            API response
            
        Raises:
            MCPConnectionError: If API call fails
            MCPExecutionError: If API returns an error
        """
        # Get API key
        brave_api_key = (
            self.client.config.get("brave.api_key") or 
            os.environ.get("BRAVE_API_KEY")
        )
        
        if not brave_api_key:
            raise MCPExecutionError("Brave API key not found")
        
        # Prepare request
        headers = {"X-Subscription-Token": brave_api_key, "Accept": "application/json"}
        params = {"q": query, "count": count, "offset": offset}
        url = "https://api.search.brave.com/res/v1/web/search"
        
        # Use executor for blocking HTTP request
        loop = asyncio.get_running_loop()
        try:
            # Make request in a separate thread
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=30,
                    verify=True  # Always verify SSL certificates
                )
            )
            
            # Raise for HTTP errors
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            raise MCPConnectionError(f"Brave Search API request error: {e}")
        except json.JSONDecodeError as e:
            raise MCPConnectionError(f"Invalid JSON response from Brave Search API: {e}")
        except Exception as e:
            raise MCPExecutionError(f"Unexpected error calling Brave Search API: {e}")
    
    # Alias for compatibility
    async def search(self, query: str, count: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Alias for brave_web_search"""
        return await self.brave_web_search(query, count, offset)
    
    async def local_search(self, query: str, count: int = 5) -> Dict[str, Any]:
        """
        Perform a local search
        
        Args:
            query: Search query
            count: Number of results
            
        Returns:
            Service response
        """
        # Add "near me" if not already in query
        if "near me" not in query.lower() and "near" not in query.lower():
            query = f"{query} near me"
            
        try:
            # Try via MCP service first
            return await self.call_method("local_search", {
                "query": query,
                "count": count
            })
        except Exception as e:
            logger.warning(f"MCP local_search failed, falling back to web search: {e}")
            
            # Fallback to regular web search
            return await self.brave_web_search(query, count)


class MCPGitHubService(MCPBaseService):
    """MCP GitHub service"""
    
    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
        """Initialize MCP GitHub service"""
        super().__init__(client, server_id)
        self.service_name = "GitHub"
    
    async def create_or_update_file(
        self, 
        owner: str, 
        repo: str, 
        path: str, 
        content: str,
        message: str, 
        branch: str, 
        sha: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create or update a file
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            content: File content
            message: Commit message
            branch: Branch name
            sha: File SHA (for updates)
            
        Returns:
            Service response
        """
        # Validate parameters
        if not all([owner, repo, path, message, branch]):
            raise ValueError("Missing required parameters")
            
        params = {
            "owner": owner,
            "repo": repo,
            "path": path,
            "content": content,
            "message": message,
            "branch": branch
        }
        
        if sha:
            params["sha"] = sha
        
        return await self.call_method("create_or_update_file", params)


class MCPManager:
    """Manager for MCP services"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize MCP manager
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.client = MCPClient(self.config)
        
        # Initialize services
        self.memory = MCPMemoryService(self.client)
        self.fetch = MCPFetchService(self.client)
        self.brave_search = MCPBraveSearchService(self.client)
        self.github = MCPGitHubService(self.client)
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """
        List available MCP servers
        
        Returns:
            List of server information
        """
        return self.client.list_servers()
    
    def add_server(self, server_id: str, url: str) -> bool:
        """
        Add a new MCP server
        
        Args:
            server_id: Server ID
            url: Server URL
            
        Returns:
            Whether the server was added
            
        Raises:
            ValueError: If server ID or URL is invalid
        """
        return self.client.add_server(server_id, url)
    
    def remove_server(self, server_id: str) -> bool:
        """
        Remove an MCP server
        
        Args:
            server_id: Server ID
            
        Returns:
            Whether the server was removed
        """
        return self.client.remove_server(server_id)
    
    def set_default_server(self, server_id: str) -> bool:
        """
        Set default MCP server
        
        Args:
            server_id: Server ID
            
        Returns:
            Whether the default server was set
        """
        return self.client.set_default_server(server_id)
    
    async def stop_server(self, server_id: str) -> bool:
        """
        Stop a running MCP server
        
        Args:
            server_id: Server ID
            
        Returns:
            Whether the server was stopped
        """
        return await self.client.stop_server(server_id)
        
    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers"""
        # Get server IDs first to avoid modification during iteration
        server_ids = [server["id"] for server in self.list_servers() 
                    if server.get("type") == "stdio" and server.get("status") == "running"]
        
        # Stop each server
        for server_id in server_ids:
            await self.stop_server(server_id)