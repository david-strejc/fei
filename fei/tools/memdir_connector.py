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
    _server_process: Optional[subprocess.Popen] = None
    _managed_data_dir: Optional[str] = None # Store data dir used by managed process
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

    def _start_server(self, data_dir: Optional[str] = None) -> bool: # Added data_dir argument
        """
        Start the Memdir server if not already running

        Args:
            data_dir: Explicit data directory path to use (overrides config lookup).

        Returns:
            True if server was started successfully, False otherwise
        """
        # Check if the class already has a running server process
        if MemdirConnector._server_process is not None and MemdirConnector._server_process.poll() is None:
            logger.info("Managed server process already running. Skipping start.")
            return False # Indicate server was already running

        # REMOVED: Old checks for existing _server_process handle and port in use.
        # The check above handles the case where *this class* started a server.
        # We still attempt Popen otherwise and rely on the post-start health check.
        try:
            logger.info(f"Starting Memdir server on port {self._port}...")
            cmd = [
                sys.executable,
                "-m",
                "memdir_tools.run_server",
                "--port",
                str(self._port),
                # The server script should ideally read the API key from its environment
                # "--api-key", # Remove direct CLI arg for API key
                # self.api_key
            ]

            # Use passed data_dir argument if provided, otherwise try config (though config lookup seems unreliable here)
            if data_dir is None:
                config = get_config()
                data_dir = config.get("memdir", {}).get("data_dir")
            logger.info(f"Connector fetched data_dir from config: {data_dir}") # ADDED LOGGING
            if data_dir:
                cmd.extend(["--data-dir", data_dir])
                logger.info(f"Adding --data-dir {data_dir} to server command.")
            else:
                logger.warning("memdir.data_dir not found in config, server will use its default.")

            # Prepare environment for the subprocess (API key, URL, Data Dir)
            server_env = os.environ.copy()
            server_env["MEMDIR_API_KEY"] = self.api_key # Ensure the key is passed via env
            server_env["MEMDIR_SERVER_URL"] = self.server_url # Pass URL via env
            if data_dir: # data_dir was fetched from config earlier
                server_env["MEMDIR_DATA_DIR"] = data_dir # Explicitly add to env for subprocess
                logger.info(f"Adding MEMDIR_DATA_DIR={data_dir} to server subprocess environment.")
            else: # ADDED ELSE FOR LOGGING
                logger.warning("data_dir is None or empty, MEMDIR_DATA_DIR not added to server env.") # ADDED LOGGING

            # ADDED LOGGING FOR FULL ENV - Use debug level
            logger.debug(f"Full server_env passed to Popen includes MEMDIR_DATA_DIR: {server_env.get('MEMDIR_DATA_DIR')}")


            log_dir = os.path.join(os.path.expanduser("~"), ".memdir_logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"memdir_server_{self._port}.log") # Port-specific log
            f_log = open(log_file, 'a')

            logger.info(f"Starting server subprocess with command: {' '.join(cmd)}")
            logger.info(f"Server subprocess environment includes: MEMDIR_API_KEY='{server_env['MEMDIR_API_KEY']}', MEMDIR_SERVER_URL='{server_env['MEMDIR_SERVER_URL']}'")

            if os.name == 'nt':
                MemdirConnector._server_process = subprocess.Popen(
                    cmd,
                    stdout=f_log,
                    stderr=f_log,
                    env=server_env, # Pass the modified environment
                    creationflags=subprocess.DETACHED_PROCESS
                    # cwd=data_dir # REMOVED: Run from project root, not data dir
                )
            else:
                MemdirConnector._server_process = subprocess.Popen(
                    cmd,
                    stdout=f_log,
                    stderr=f_log,
                    env=server_env, # Pass the modified environment
                    preexec_fn=os.setpgrp
                    # cwd=data_dir # REMOVED: Run from project root, not data dir
                )

            # Wait longer for server to start (up to 5 seconds)
            for i in range(10):  # 10 attempts, 0.5s each = 5s total
                time.sleep(0.5)
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=0.5)
                    if response.status_code == 200:
                        logger.info(f"Memdir server started successfully after {i/2:.1f}s using data_dir: {data_dir}")
                        MemdirConnector._managed_data_dir = data_dir # Store on class
                        return True
                except Exception:
                    pass # Ignore connection errors during startup check
                continue # Continue loop to check again

            # If loop finishes, check process status
            if MemdirConnector._server_process.poll() is None:
                logger.info(f"Memdir server process started (using data_dir: {data_dir}) but not responding to health checks yet.") # Updated log message
                MemdirConnector._managed_data_dir = data_dir # Store on class
                return True # Assume it might become ready later
            else:
                logger.error("Memdir server process exited unexpectedly during startup.")
                MemdirConnector._managed_data_dir = None # Clear class variable
                return False

        except Exception as e:
            logger.error(f"Error starting Memdir server: {e}", exc_info=True)
            MemdirConnector._managed_data_dir = None # Clear class variable
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
                # Also clear the class variable for data directory
                MemdirConnector._managed_data_dir = None
    
    def check_connection(self, start_if_needed: bool = False) -> bool:
        """
        Check if the connection to the Memdir server is working
        
        Args:
            start_if_needed: Whether to start the server if not running
            
        Returns:
            True if connected, False otherwise
        """
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
        
        result = self._make_request("GET", "search", params=params)
        # The server returns a dict like {'count': N, 'query': Q, 'results': [...] }
        # We should return just the list of results
        return result.get("results", []) # Return empty list if 'results' key is missing
    
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
        return {"status": "stopped", "message": "Memdir server stopped successfully"}

    def get_server_status(self) -> Dict[str, Any]:
        """
        Get the status of the managed Memdir server process.

        Returns:
            Dictionary with status info: running (bool), pid (int|None),
            port (int), url (str), data_dir (str|None).
        """
        is_running = False
        pid = None
        data_dir = None # Cannot reliably determine data_dir after start easily

        # Check the class variable holding the process
        process = MemdirConnector._server_process
        if process is not None:
            if process.poll() is None:
                # Process is running
                is_running = True
                pid = process.pid
                # Retrieve data_dir from the class variable if running
                data_dir = MemdirConnector._managed_data_dir
                # Try to retrieve data_dir if stored during start (e.g., on self)
                # data_dir = getattr(self, '_last_started_data_dir', None)
            else:
                # Process has terminated, clear the class variable
                logger.info(f"Server process (PID: {process.pid}) found but has terminated. Clearing reference.")
                MemdirConnector._server_process = None

        # Construct the status dictionary in the format expected by tests
        status = {
            "running": is_running,
            "pid": pid,
            "port": self._port, # Return configured port
            "url": self.server_url,
            "data_dir": data_dir # Will be None for now
        }
        return status
            
        self._stop_server()
        return {"status": "stopped", "message": "Memdir server stopped"}
    
    # REMOVED duplicate @classmethod get_server_status

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
