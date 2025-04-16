"""
Memdir memory system implementation for Fei.
"""

import os
import sys
import json
import logging
import socket
import subprocess
import time
import signal
import atexit
from typing import Dict, List, Optional, Any, Union, Tuple
import requests
from datetime import datetime

from fei.memory.base import MemoryInterface, MemoryFactory
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class MemdirMemory(MemoryInterface):
    """Memdir memory system implementation."""
    
    # Class variable to track server process
    _server_process: Optional[subprocess.Popen] = None
    _managed_data_dir: Optional[str] = None
    _port = 5000
    
    def __init__(self):
        """Initialize the Memdir memory system."""
        self.config = Config()
        self.server_url = None
        self.api_key = None
        self.auto_start = False
    
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the memory system.
        
        Args:
            config: Optional configuration dictionary
        """
        config = config or {}
        
        # Set server URL and API key (priority: config > env > defaults)
        self.server_url = (
            config.get("server_url") or 
            os.environ.get("MEMDIR_SERVER_URL") or 
            "http://localhost:5000"
        )
        
        self.api_key = (
            config.get("api_key") or 
            os.environ.get("MEMDIR_API_KEY") or 
            "default_api_key"
        )
        
        # Parse port from server URL
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(self.server_url)
            self._port = parsed_url.port or 5000
            MemdirMemory._port = self._port
        except:
            self._port = 5000
            MemdirMemory._port = 5000
        
        # Auto-start server if configured
        self.auto_start = config.get("auto_start", False)
        if self.auto_start:
            self._start_server(config.get("data_dir"))
    
    def _setup_headers(self) -> Dict[str, str]:
        """Set up request headers with API key."""
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the Memdir server.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (without leading slash)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Response data as dictionary
            
        Raises:
            Exception: If the request fails
        """
        url = f"{self.server_url}/{endpoint}"
        
        # Add headers if not provided
        if "headers" not in kwargs:
            kwargs["headers"] = self._setup_headers()
        
        # Add timeout if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = 10.0
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Memdir API request failed: {e}")
            
            # If server might not be running, provide helpful message
            if isinstance(e, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
                raise Exception("Cannot connect to Memdir server. Use the memdir_server_start tool to start it.")
            
            if hasattr(e, "response") and e.response is not None:
                error_msg = f"Status: {e.response.status_code}"
                try:
                    error_data = e.response.json()
                    if "error" in error_data:
                        error_msg += f" - {error_data['error']}"
                except:
                    error_msg += f" - {e.response.text}"
                raise Exception(error_msg)
            raise Exception(f"Connection error: {str(e)}")
    
    def _is_port_in_use(self, port: int) -> bool:
        """
        Check if a port is already in use.
        
        Args:
            port: Port number to check
            
        Returns:
            True if the port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def _start_server(self, data_dir: Optional[str] = None) -> bool:
        """
        Start the Memdir server if not already running.
        
        Args:
            data_dir: Explicit data directory path to use
            
        Returns:
            True if server was started successfully, False otherwise
        """
        # Check if the class already has a running server process
        if MemdirMemory._server_process is not None and MemdirMemory._server_process.poll() is None:
            logger.info("Managed server process already running. Skipping start.")
            return False
        
        try:
            logger.info(f"Starting Memdir server on port {self._port}...")
            cmd = [
                sys.executable,
                "-m",
                "memdir_tools.run_server",
                "--port",
                str(self._port),
            ]
            
            # Use passed data_dir argument if provided
            if data_dir is None:
                data_dir = self.config.get("memdir", {}).get("data_dir")
            
            logger.info(f"Connector fetched data_dir from config: {data_dir}")
            if data_dir:
                cmd.extend(["--data-dir", data_dir])
                logger.info(f"Adding --data-dir {data_dir} to server command.")
            else:
                logger.warning("memdir.data_dir not found in config, server will use its default.")
            
            # Prepare environment for the subprocess
            server_env = os.environ.copy()
            server_env["MEMDIR_API_KEY"] = self.api_key
            server_env["MEMDIR_SERVER_URL"] = self.server_url
            if data_dir:
                server_env["MEMDIR_DATA_DIR"] = data_dir
                logger.info(f"Adding MEMDIR_DATA_DIR={data_dir} to server subprocess environment.")
            else:
                logger.warning("data_dir is None or empty, MEMDIR_DATA_DIR not added to server env.")
            
            logger.debug(f"Full server_env passed to Popen includes MEMDIR_DATA_DIR: {server_env.get('MEMDIR_DATA_DIR')}")
            
            log_dir = os.path.join(os.path.expanduser("~"), ".memdir_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"memdir_server_{self._port}.log")
            f_log = open(log_file, 'a')
            
            logger.info(f"Starting server subprocess with command: {' '.join(cmd)}")
            logger.info(f"Server subprocess environment includes: MEMDIR_API_KEY='{server_env['MEMDIR_API_KEY']}', MEMDIR_SERVER_URL='{server_env['MEMDIR_SERVER_URL']}'")
            
            if os.name == 'nt':
                MemdirMemory._server_process = subprocess.Popen(
                    cmd,
                    stdout=f_log,
                    stderr=f_log,
                    env=server_env,
                    creationflags=subprocess.DETACHED_PROCESS
                )
            else:
                MemdirMemory._server_process = subprocess.Popen(
                    cmd,
                    stdout=f_log,
                    stderr=f_log,
                    env=server_env,
                    preexec_fn=os.setpgrp
                )
            
            # Wait for server to start
            for i in range(10):
                time.sleep(0.5)
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=0.5)
                    if response.status_code == 200:
                        logger.info(f"Memdir server started successfully after {i/2:.1f}s using data_dir: {data_dir}")
                        MemdirMemory._managed_data_dir = data_dir
                        return True
                except Exception:
                    pass
                continue
            
            # If loop finishes, check process status
            if MemdirMemory._server_process.poll() is None:
                logger.info(f"Memdir server process started (using data_dir: {data_dir}) but not responding to health checks yet.")
                MemdirMemory._managed_data_dir = data_dir
                return True
            else:
                logger.error("Memdir server process exited unexpectedly during startup.")
                MemdirMemory._managed_data_dir = None
                return False
        
        except Exception as e:
            logger.error(f"Error starting Memdir server: {e}", exc_info=True)
            MemdirMemory._managed_data_dir = None
            return False
    
    @classmethod
    def _stop_server(cls) -> None:
        """Stop the Memdir server if it was started by this class."""
        if cls._server_process is not None:
            logger.info("Stopping Memdir server...")
            try:
                if os.name == 'nt':
                    # Windows
                    cls._server_process.terminate()
                else:
                    # Unix - send SIGTERM to process group
                    os.killpg(os.getpgid(cls._server_process.pid), signal.SIGTERM)
                
                cls._server_process.wait(timeout=5)
                logger.info("Memdir server stopped")
            except Exception as e:
                logger.error(f"Error stopping Memdir server: {e}")
            finally:
                cls._server_process = None
                cls._managed_data_dir = None
    
    def check_connection(self) -> bool:
        """
        Check if the connection to the Memdir server is working.
        
        Returns:
            True if connected, False otherwise
        """
        try:
            response = requests.get(f"{self.server_url}/health", timeout=1.0)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            if self.auto_start:
                logger.info("Starting Memdir server on demand...")
                return self._start_server()
            return False
        except Exception as e:
            logger.debug(f"Error checking Memdir connection: {e}")
            return False
    
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for memories matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memories
        """
        folder = kwargs.get("folder")
        status = kwargs.get("status")
        offset = kwargs.get("offset", 0)
        with_content = kwargs.get("with_content", False)
        debug = kwargs.get("debug", False)
        
        params = {"q": query}
        
        if folder:
            params["folder"] = folder
        if status:
            params["status"] = status
        if limit is not None:
            params["limit"] = str(limit)
        if offset:
            params["offset"] = str(offset)
        if with_content:
            params["with_content"] = "true"
        if debug:
            params["debug"] = "true"
        
        result = self._make_request("GET", "search", params=params)
        return result.get("results", [])
    
    def create(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new memory.
        
        Args:
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Created memory
        """
        metadata = metadata or {}
        
        # Extract folder and flags from metadata
        folder = metadata.pop("folder", "")
        flags = metadata.pop("flags", "")
        
        # Use remaining metadata as headers
        headers = metadata
        
        # Add default headers if not provided
        if "Subject" not in headers:
            headers["Subject"] = "Memory created at " + datetime.now().isoformat()
        if "Source" not in headers:
            headers["Source"] = "Fei Assistant"
        if "Date" not in headers:
            headers["Date"] = datetime.now().isoformat()
        
        data = {
            "content": content,
            "headers": headers,
            "folder": folder,
            "flags": flags
        }
        
        return self._make_request("POST", "memories", json=data)
    
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory or None if not found
        """
        try:
            return self._make_request("GET", f"memories/{memory_id}")
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            return None
    
    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID
            content: Optional new content
            metadata: Optional new metadata
            
        Returns:
            Updated memory or None if not found
        """
        # Get the current memory
        current_memory = self.get(memory_id)
        if not current_memory:
            return None
        
        # Extract current values
        current_folder = current_memory.get("metadata", {}).get("folder", "")
        current_status = current_memory.get("metadata", {}).get("status", "cur")
        current_flags = current_memory.get("metadata", {}).get("flags", "")
        
        # Update content if provided
        if content is not None:
            # We need to create a new memory with the updated content
            # and then delete the old one
            new_memory = self.create(content, {
                **current_memory.get("headers", {}),
                "folder": current_folder,
                "flags": current_flags
            })
            
            # Delete the old memory
            self.delete(memory_id)
            
            return new_memory
        
        # Update metadata if provided
        if metadata:
            # Extract folder and flags from metadata
            new_folder = metadata.pop("folder", current_folder)
            new_flags = metadata.pop("flags", current_flags)
            
            # Update headers
            if metadata:
                # We need to create a new memory with the updated headers
                # and then delete the old one
                new_memory = self.create(current_memory.get("content", ""), {
                    **current_memory.get("headers", {}),
                    **metadata,
                    "folder": new_folder,
                    "flags": new_flags
                })
                
                # Delete the old memory
                self.delete(memory_id)
                
                return new_memory
            
            # Update folder or flags
            if new_folder != current_folder or new_flags != current_flags:
                data = {
                    "source_folder": current_folder,
                    "target_folder": new_folder,
                    "source_status": current_status,
                    "target_status": current_status,
                    "flags": new_flags
                }
                
                return self._make_request("PUT", f"memories/{memory_id}", json=data)
        
        # No updates needed
        return current_memory
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            self._make_request("DELETE", f"memories/{memory_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def list(self, limit: int = 100, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
        """
        List memories.
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            **kwargs: Additional filter parameters
            
        Returns:
            List of memories
        """
        folder = kwargs.get("folder", "")
        status = kwargs.get("status", "cur")
        with_content = kwargs.get("with_content", False)
        
        params = {
            "folder": folder,
            "status": status,
            "with_content": "true" if with_content else "false"
        }
        
        result = self._make_request("GET", "memories", params=params)
        memories = result.get("memories", [])
        
        # Apply limit and offset
        return memories[offset:offset + limit]
    
    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for memories by tag.
        
        Args:
            tag: Tag to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        # Use the search method with a tag query
        return self.search(f"tag:{tag}", limit=limit)
    
    def start_server(self, data_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Start the Memdir server.
        
        Args:
            data_dir: Optional data directory path
            
        Returns:
            Status dictionary
        """
        # Check if server is already running
        if self._is_port_in_use(self._port):
            return {"status": "already_running", "message": "Memdir server is already running"}
        
        # If we already have a process reference, check if it's still alive
        if MemdirMemory._server_process is not None:
            if MemdirMemory._server_process.poll() is None:
                return {"status": "already_running", "message": "Memdir server is already running"}
            else:
                MemdirMemory._server_process = None
        
        # Start the server
        success = self._start_server(data_dir)
        
        # Wait a bit for the server to be ready
        time.sleep(1.0)
        
        # Try to connect to verify it started
        try:
            response = requests.get(f"{self.server_url}/health", timeout=1.0)
            if response.status_code == 200:
                return {"status": "started", "message": "Memdir server started successfully"}
        except:
            pass
        
        # If we got here, the server might still be starting but not ready yet
        if success:
            return {"status": "started", "message": "Memdir server is starting"}
        else:
            return {"status": "error", "message": "Failed to start Memdir server"}
    
    def stop_server(self) -> Dict[str, Any]:
        """
        Stop the Memdir server.
        
        Returns:
            Status dictionary
        """
        if MemdirMemory._server_process is None:
            return {"status": "not_running", "message": "Memdir server is not running"}
        
        self._stop_server()
        return {"status": "stopped", "message": "Memdir server stopped successfully"}
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get the status of the Memdir server.
        
        Returns:
            Status dictionary
        """
        is_running = False
        pid = None
        data_dir = None
        
        # Check the class variable holding the process
        process = MemdirMemory._server_process
        if process is not None:
            if process.poll() is None:
                is_running = True
                pid = process.pid
                data_dir = MemdirMemory._managed_data_dir
            else:
                logger.info(f"Server process (PID: {process.pid}) found but has terminated. Clearing reference.")
                MemdirMemory._server_process = None
        
        return {
            "running": is_running,
            "pid": pid,
            "port": self._port,
            "url": self.server_url,
            "data_dir": data_dir
        }


# Register the memory system
MemoryFactory.register_memory_system("memdir", MemdirMemory)
