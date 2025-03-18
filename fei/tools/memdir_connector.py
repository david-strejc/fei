#!/usr/bin/env python3
"""
Memdir connector for FEI assistant
This module provides a connector class for FEI to interact with a Memdir server
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
from typing import Dict, List, Any, Optional, Union, Tuple
import requests
from datetime import datetime

from fei.utils.config import get_config

# Set up logging
logger = logging.getLogger(__name__)

class MemdirConnector:
    """Connector for interacting with a Memdir server"""
    
    # Class variable to track server process
    _server_process = None
    _port = 5000
    
    def __init__(self, server_url: Optional[str] = None, api_key: Optional[str] = None, 
                 auto_start: bool = False):
        """
        Initialize the connector
        
        Args:
            server_url: The URL of the Memdir server (default: from config)
            api_key: The API key for authentication (default: from config)
            auto_start: Whether to automatically start the server if not running
        """
        # Get configuration
        config = get_config()
        memdir_config = config.get("memdir", {})
        
        # Set server URL and API key (priority: args > config > env > defaults)
        self.server_url = (
            server_url or 
            memdir_config.get("server_url") or 
            os.environ.get("MEMDIR_SERVER_URL") or 
            "http://localhost:5000"
        )
        
        self.api_key = (
            api_key or 
            memdir_config.get("api_key") or 
            os.environ.get("MEMDIR_API_KEY") or 
            "default_api_key"  # Use a default key instead of None
        )
        
        # Parse port from server URL
        try:
            from urllib.parse import urlparse
            parsed_url = urlparse(self.server_url)
            self._port = parsed_url.port or 5000
            MemdirConnector._port = self._port  # Set the class variable
        except:
            self._port = 5000
            MemdirConnector._port = 5000
            
        # Auto-start server if needed - only when explicitly requested
        self.auto_start = auto_start
    
    def _setup_headers(self) -> Dict[str, str]:
        """Set up request headers with API key"""
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """
        Make an HTTP request to the Memdir server
        
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
            kwargs["timeout"] = 10.0  # 10 second timeout
        
        # Don't auto-start server here - we'll let the user explicitly start it
        # with the memdir_server_start tool when needed
            
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
        Check if a port is already in use
        
        Args:
            port: Port number to check
            
        Returns:
            True if the port is in use, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
            
    def _start_server(self) -> bool:
        """
        Start the Memdir server if not already running
        
        Returns:
            True if server was started successfully, False otherwise
        """
        # Check if server is already running - either our process or something else
        if MemdirConnector._server_process is not None:
            # Process exists but might have terminated
            if MemdirConnector._server_process.poll() is None:
                # Process is still running
                logger.info("Memdir server process is already running.")
                return True
            # Process has terminated, clean up the reference
            logger.info("Previous Memdir server process has terminated. Starting a new one.")
            MemdirConnector._server_process = None
            
        # Check if some other process is using our port
        if self._is_port_in_use(self._port):
            logger.info(f"Port {self._port} is already in use. Assuming server is running.")
            
            # Verify that it's actually a Memdir server by checking the health endpoint
            try:
                response = requests.get(f"{self.server_url}/health", timeout=1.0)
                if response.status_code == 200:
                    logger.info("Verified Memdir server is running on the port.")
                    return True
                logger.warning("Something is using the port but it's not a Memdir server.")
            except Exception:
                logger.warning("Something is using the port but it's not responding to health checks.")
            
        try:
            # Start the server as a subprocess with API key
            logger.info(f"Starting Memdir server on port {self._port}...")
            cmd = [
                sys.executable, 
                "-m", 
                "memdir_tools.run_server", 
                "--port", 
                str(self._port),
                "--api-key", 
                self.api_key
            ]
            
            # Create a log file for the server output
            log_dir = os.path.join(os.path.expanduser("~"), ".memdir_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "memdir_server.log")
            
            # Open the log file
            f_log = open(log_file, 'a')
            
            # Use DETACHED_PROCESS flag on Windows to run in background
            if os.name == 'nt':
                MemdirConnector._server_process = subprocess.Popen(
                    cmd, 
                    stdout=f_log,  # Redirect output to log file
                    stderr=f_log,
                    creationflags=subprocess.DETACHED_PROCESS
                )
            else:
                # On Unix, just use normal subprocess
                MemdirConnector._server_process = subprocess.Popen(
                    cmd,
                    stdout=f_log,  # Redirect output to log file
                    stderr=f_log,
                    preexec_fn=os.setpgrp  # Run in a new process group
                )
                
            # Register cleanup function (only once)
            if not hasattr(MemdirConnector, "_cleanup_registered"):
                atexit.register(self._stop_server)
                MemdirConnector._cleanup_registered = True
            
            # Wait longer for server to start (up to 5 seconds)
            for i in range(10):  # 10 attempts, 0.5s each = 5s total
                time.sleep(0.5)
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=0.5)
                    if response.status_code == 200:
                        logger.info(f"Memdir server started successfully after {i/2:.1f}s")
                        return True
                except Exception:
                    continue
                    
            # If we get here, the server didn't respond in time, but might still be starting
            # Check if the process is still running
            if MemdirConnector._server_process.poll() is None:
                logger.info("Memdir server process started but not responding to health checks yet.")
                return True
            else:
                # Process exited
                logger.error("Memdir server process exited unexpectedly.")
                return False
            
        except Exception as e:
            logger.error(f"Error starting Memdir server: {e}")
            return False
            
    @classmethod
    def _stop_server(cls) -> None:
        """Stop the Memdir server if it was started by this class"""
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
    
    def check_connection(self, start_if_needed: bool = False) -> bool:
        """
        Check if the connection to the Memdir server is working
        
        Args:
            start_if_needed: Whether to start the server if not running
            
        Returns:
            True if connected, False otherwise
        """
        # Skip connection check during object initialization
        if not hasattr(self, "_initialized_check"):
            self._initialized_check = True
            return False
            
        try:
            # Use a shorter timeout to avoid hanging
            response = requests.get(f"{self.server_url}/health", timeout=1.0)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            # Don't log every connection failure - it's too noisy
            if start_if_needed and self.auto_start:
                logger.info("Starting Memdir server on demand...")
                return self._start_server()
            return False
        except Exception as e:
            logger.debug(f"Error checking Memdir connection: {e}")
            return False
    
    def list_memories(self, folder: str = "", status: str = "cur", with_content: bool = False) -> List[Dict[str, Any]]:
        """
        List memories in a folder
        
        Args:
            folder: Folder name (default: root folder)
            status: Status folder (cur, new, tmp)
            with_content: Whether to include content
            
        Returns:
            List of memory dictionaries
        """
        params = {
            "folder": folder,
            "status": status,
            "with_content": "true" if with_content else "false"
        }
        
        result = self._make_request("GET", "memories", params=params)
        return result["memories"]
    
    def create_memory(self, content: str, headers: Dict[str, str] = None, folder: str = "", flags: str = "") -> Dict[str, Any]:
        """
        Create a new memory
        
        Args:
            content: Memory content
            headers: Memory headers
            folder: Target folder
            flags: Memory flags
            
        Returns:
            Result dictionary
        """
        data = {
            "content": content,
            "headers": headers or {},
            "folder": folder,
            "flags": flags
        }
        
        return self._make_request("POST", "memories", json=data)
    
    def get_memory(self, memory_id: str, folder: str = "") -> Dict[str, Any]:
        """
        Get a specific memory
        
        Args:
            memory_id: Memory ID or filename
            folder: Folder to search in (default: all folders)
            
        Returns:
            Memory dictionary
        """
        params = {}
        if folder:
            params["folder"] = folder
        
        return self._make_request("GET", f"memories/{memory_id}", params=params)
    
    def move_memory(self, memory_id: str, source_folder: str, target_folder: str, 
                   source_status: Optional[str] = None, target_status: str = "cur",
                   flags: Optional[str] = None) -> Dict[str, Any]:
        """
        Move a memory from one folder to another
        
        Args:
            memory_id: Memory ID or filename
            source_folder: Source folder
            target_folder: Target folder
            source_status: Source status folder (default: auto-detect)
            target_status: Target status folder
            flags: New flags (optional)
            
        Returns:
            Result dictionary
        """
        data = {
            "source_folder": source_folder,
            "target_folder": target_folder,
            "target_status": target_status
        }
        
        if source_status:
            data["source_status"] = source_status
        if flags is not None:
            data["flags"] = flags
        
        return self._make_request("PUT", f"memories/{memory_id}", json=data)
    
    def update_flags(self, memory_id: str, flags: str, folder: str = "", status: Optional[str] = None) -> Dict[str, Any]:
        """
        Update memory flags
        
        Args:
            memory_id: Memory ID or filename
            flags: New flags
            folder: Memory folder
            status: Memory status folder (default: auto-detect)
            
        Returns:
            Result dictionary
        """
        data = {
            "source_folder": folder,
            "flags": flags
        }
        
        if status:
            data["source_status"] = status
        
        return self._make_request("PUT", f"memories/{memory_id}", json=data)
    
    def delete_memory(self, memory_id: str, folder: str = "") -> Dict[str, Any]:
        """
        Move a memory to trash
        
        Args:
            memory_id: Memory ID or filename
            folder: Memory folder
            
        Returns:
            Result dictionary
        """
        params = {}
        if folder:
            params["folder"] = folder
        
        return self._make_request("DELETE", f"memories/{memory_id}", params=params)
    
    def search(self, query: str, folder: Optional[str] = None, status: Optional[str] = None,
              limit: Optional[int] = None, offset: int = 0, 
              with_content: bool = False, debug: bool = False) -> Dict[str, Any]:
        """
        Search memories
        
        Args:
            query: Search query
            folder: Folder to search in (default: all folders)
            status: Status folder to search in (default: all statuses)
            limit: Maximum number of results
            offset: Offset for pagination
            with_content: Whether to include content
            debug: Whether to show debug information
            
        Returns:
            Result dictionary with count and results
        """
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
        
        return self._make_request("GET", "search", params=params)
    
    def list_folders(self) -> List[str]:
        """
        List all folders
        
        Returns:
            List of folder names
        """
        result = self._make_request("GET", "folders")
        return result["folders"]
    
    def create_folder(self, folder: str) -> Dict[str, Any]:
        """
        Create a new folder
        
        Args:
            folder: Folder name
            
        Returns:
            Result dictionary
        """
        data = {"folder": folder}
        return self._make_request("POST", "folders", json=data)
    
    def delete_folder(self, folder: str) -> Dict[str, Any]:
        """
        Delete a folder
        
        Args:
            folder: Folder name
            
        Returns:
            Result dictionary
        """
        return self._make_request("DELETE", f"folders/{folder}")
    
    def rename_folder(self, folder: str, new_name: str) -> Dict[str, Any]:
        """
        Rename a folder
        
        Args:
            folder: Original folder name
            new_name: New folder name
            
        Returns:
            Result dictionary
        """
        data = {"new_name": new_name}
        return self._make_request("PUT", f"folders/{folder}", json=data)
    
    def run_filters(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run memory filters
        
        Args:
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Result dictionary
        """
        data = {"dry_run": dry_run}
        return self._make_request("POST", "filters/run", json=data)
    
    def create_memory_from_conversation(self, subject: str, content: str, tags: str = "", 
                                      priority: str = "medium", folder: str = "") -> Dict[str, Any]:
        """
        Create a memory from assistant conversation
        
        Args:
            subject: Memory subject
            content: Memory content
            tags: Tags for the memory (comma-separated)
            priority: Memory priority (high, medium, low)
            folder: Target folder
            
        Returns:
            Result dictionary
        """
        # Create headers
        headers = {
            "Subject": subject,
            "Tags": tags,
            "Priority": priority,
            "Source": "FEI Assistant",
            "Date": datetime.now().isoformat()
        }
        
        # Create the memory
        return self.create_memory(content, headers, folder)

    def start_server_command(self) -> Dict[str, Any]:
        """
        Command to start the Memdir server
        
        Returns:
            Status dictionary
        """
        # Check if server is already running
        if self._is_port_in_use(self._port):
            # Port is in use, assume server is running
            return {"status": "already_running", "message": "Memdir server is already running"}
            
        # If we already have a process reference, check if it's still alive
        if MemdirConnector._server_process is not None:
            if MemdirConnector._server_process.poll() is None:
                # Process is still running
                return {"status": "already_running", "message": "Memdir server is already running"}
            else:
                # Process has terminated, clean up the reference
                MemdirConnector._server_process = None
                
        # Start the server
        success = self._start_server()
        
        # Wait a bit for the server to be ready
        import time
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
    
    def stop_server_command(self) -> Dict[str, Any]:
        """
        Command to stop the Memdir server
        
        Returns:
            Status dictionary
        """
        if MemdirConnector._server_process is None:
            return {"status": "not_running", "message": "Memdir server is not running"}
            
        self._stop_server()
        return {"status": "stopped", "message": "Memdir server stopped"}
    
    @classmethod
    def get_server_status(cls) -> Dict[str, Any]:
        """
        Get the current server status
        
        Returns:
            Status dictionary
        """
        is_running = cls._server_process is not None
        
        if is_running:
            # Check if process is actually still running
            if cls._server_process.poll() is not None:
                # Process has terminated
                cls._server_process = None
                is_running = False
                
        return {
            "status": "running" if is_running else "stopped",
            "message": "Memdir server is running" if is_running else "Memdir server is not running",
            "port": cls._port if is_running else None
        }

# Example usage
if __name__ == "__main__":
    # Simple test client
    connector = MemdirConnector()
    
    if not connector.check_connection():
        print("Cannot connect to Memdir server. Make sure it's running.")
        sys.exit(1)
    
    # List folders
    print("Memdir folders:")
    for folder in connector.list_folders():
        print(f"  {folder or 'Inbox'}")
    
    # List memories in the root folder
    print("\nMemories in the root folder:")
    memories = connector.list_memories()
    for memory in memories[:5]:  # Show first 5 only
        print(f"  {memory['metadata']['unique_id']} - {memory['headers'].get('Subject', 'No subject')}")
    
    # Search for memories
    print("\nSearch for 'python':")
    search_results = connector.search("python", limit=3)
    for memory in search_results["results"]:
        print(f"  {memory['metadata']['unique_id']} - {memory['headers'].get('Subject', 'No subject')}")