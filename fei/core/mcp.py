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
import subprocess # Keep for types if needed
import urllib.parse
import signal
import ssl
import httpx # Use httpx for async HTTP
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

class MCPConsentDeniedError(Exception):
    """Exception raised when user denies MCP action consent"""
    pass

class ProcessManager:
    """Manager for child processes using asyncio"""

    def __init__(self):
        """Initialize process manager"""
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()
        # NOTE: Removed atexit registration for cleanup. Cleanup should be handled explicitly
        # by calling MCPManager.stop_all_servers() or MCPClient.close() on application exit.

    async def start_process(
        self,
        process_id: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> asyncio.subprocess.Process:
        """
        Start a process safely using asyncio.
        """
        # Validate command
        if not command or not isinstance(command, list):
            raise ValueError("Command must be a non-empty list")

        # Use lock to prevent race conditions
        async with self._lock:
            # Check if process already exists and is running
            process = self.processes.get(process_id)
            if process and process.returncode is None:
                logger.debug(f"Process {process_id} already running (PID: {process.pid})")
                return process

            # Set up process environment
            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            try:
                # Start the process using asyncio
                logger.debug(f"Starting process {process_id} with command: {' '.join(command)}")
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=process_env,
                    start_new_session=True # Keep this for cleanup
                )

                # Store the process
                self.processes[process_id] = process

                logger.info(f"Started process {process_id} (PID: {process.pid})")
                return process

            except (OSError, Exception) as e: # Catch broader exceptions
                logger.error(f"Failed to start process {process_id}: {e}", exc_info=True)
                raise OSError(f"Failed to start process {process_id}: {e}")

    async def stop_process(self, process_id: str, timeout: float = 3.0) -> bool:
        """
        Stop a process safely using asyncio.
        """
        async with self._lock:
            if process_id not in self.processes:
                logger.debug(f"Process {process_id} not found for stopping.")
                return False

            process = self.processes[process_id]

            # Check if already terminated (returncode is not None)
            if process.returncode is not None:
                logger.debug(f"Process {process_id} already terminated.")
                del self.processes[process_id]
                return True

            try:
                # Try to terminate gracefully using process group ID
                pgid = os.getpgid(process.pid)
                logger.debug(f"Attempting to terminate process {process_id} (PID: {process.pid}, PGID: {pgid})")
                os.killpg(pgid, signal.SIGTERM)

                # Wait for termination with timeout
                try:
                    await asyncio.wait_for(process.wait(), timeout=timeout)
                    logger.info(f"Process {process_id} terminated gracefully.")
                except asyncio.TimeoutError:
                    logger.warning(f"Process {process_id} did not terminate gracefully after {timeout}s, sending SIGKILL.")
                    os.killpg(pgid, signal.SIGKILL)
                    # Wait a short time after SIGKILL
                    try:
                       await asyncio.wait_for(process.wait(), timeout=1.0)
                       logger.info(f"Process {process_id} terminated after SIGKILL.")
                    except asyncio.TimeoutError:
                       logger.error(f"Process {process_id} failed to terminate even after SIGKILL.")
                       # Process might be defunct, remove from tracking anyway
                       if process_id in self.processes:
                           del self.processes[process_id]
                       return False

                # Clean up
                if process_id in self.processes:
                    del self.processes[process_id]
                return True

            except (OSError, ProcessLookupError, Exception) as e: # Catch ProcessLookupError if pgid is invalid
                logger.error(f"Error stopping process {process_id}: {e}", exc_info=True)

                # Still remove from our tracking if we can't manage it
                if process_id in self.processes:
                    del self.processes[process_id]

                return False

    def cleanup_all_sync(self):
        """Synchronous wrapper for cleanup, suitable for atexit."""
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                logger.warning("Event loop is running during atexit cleanup, cannot stop processes reliably.")
                # Attempt to schedule cleanup if possible, but might not run fully
                asyncio.ensure_future(self.cleanup_all_async())
                return
            else:
                 loop.run_until_complete(self.cleanup_all_async())
        except RuntimeError: # No event loop running
             logger.info("No running event loop found for sync cleanup, creating new one.")
             loop = asyncio.new_event_loop()
             asyncio.set_event_loop(loop)
             try:
                 loop.run_until_complete(self.cleanup_all_async())
             finally:
                 loop.close()
                 asyncio.set_event_loop(None) # Clean up loop association


    async def cleanup_all_async(self):
        """Asynchronously clean up all running processes"""
        logger.info("Cleaning up MCP processes...")
        # Get a copy of process IDs to avoid modification during iteration
        process_ids = list(self.processes.keys())
        if not process_ids:
            logger.info("No MCP processes to clean up.")
            return

        tasks = [self.stop_process(pid) for pid in process_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for pid, result in zip(process_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Error during async cleanup of process {pid}: {result}", exc_info=result)
            elif not result:
                 logger.warning(f"Failed to stop process {pid} during async cleanup.")
        logger.info("MCP process cleanup finished.")

    # Removed cleanup_all_sync method as atexit is no longer used.

from typing import Dict, List, Any, Optional, Union, Tuple, Set, Callable, Awaitable
from contextlib import asynccontextmanager

from fei.utils.logging import get_logger
from fei.utils.config import Config

logger = get_logger(__name__)

# Type alias for the consent handler function
ConsentHandler = Callable[[str, str, str, Dict[str, Any]], Awaitable[bool]]


class MCPServerConfigError(Exception):
    """Exception raised for MCP server configuration errors"""


class MCPClient:
    """Client for MCP servers"""

    def __init__(self, config: Optional[Config] = None, consent_handler: Optional[ConsentHandler] = None):
        """
        Initialize MCP client

        Args:
            config: Configuration object.
            consent_handler: An async function to call when user consent is needed.
                             Expected signature: async def handler(server_id, service, method, params) -> bool
        """
        self.config = config or Config()
        self.consent_handler = consent_handler # Store the consent handler

        # Initialize process manager
        self.process_manager = ProcessManager()

        # Get server configurations
        self.servers = self._load_servers()

        # Set default server
        self.default_server = self.config.get("mcp.default_server")

        # SSL context for secure requests
        self.ssl_context = self._create_ssl_context()

        # Async HTTP client session
        self.http_client = httpx.AsyncClient(timeout=30, verify=self.ssl_context)

        logger.debug(f"Initialized MCP client with {len(self.servers)} servers")

    async def close(self):
        """Close resources like the HTTP client and stop processes."""
        logger.info("Closing MCPClient resources...")
        await self.http_client.aclose()
        await self.process_manager.cleanup_all_async()
        logger.info("MCPClient resources closed.")

    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Create a secure SSL context for HTTPS requests
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
        """
        servers = {}

        # Load from config file first
        config_servers = self.config.get_dict("mcp.servers", {})
        if config_servers:
             for server_id, server_config in config_servers.items():
                 if isinstance(server_config, str): # Old URL-only format
                     if self._validate_url(server_config):
                         servers[server_id] = {"url": server_config, "type": "http"}
                     else:
                         logger.warning(f"Invalid URL in config for server {server_id}: {server_config}")
                 elif isinstance(server_config, dict): # New dict format
                     server_type = server_config.get("type", "http") # Default to http
                     if server_type == "http":
                         url = server_config.get("url")
                         if url and self._validate_url(url):
                             servers[server_id] = server_config
                         else:
                             logger.warning(f"Invalid or missing URL for HTTP server {server_id} in config.")
                     elif server_type == "stdio":
                         if "command" in server_config:
                             servers[server_id] = server_config
                         else:
                             logger.warning(f"Missing 'command' for stdio server {server_id} in config.")
                     else:
                         logger.warning(f"Unsupported server type '{server_type}' for server {server_id} in config.")
                 else:
                     logger.warning(f"Invalid config format for server {server_id}")


        # Load/Override from environment variables (FEI_MCP_SERVER_<ID>=<URL> or FEI_MCP_SERVER_<ID>_CONFIG=<JSON>)
        for env_var, value in os.environ.items():
            if env_var.startswith("FEI_MCP_SERVER_"):
                if env_var.endswith("_CONFIG"):
                    server_id = env_var[15:-7].lower()
                    try:
                        server_config = json.loads(value)
                        if isinstance(server_config, dict):
                             server_type = server_config.get("type", "http")
                             if server_type == "http":
                                 url = server_config.get("url")
                                 if url and self._validate_url(url):
                                     servers[server_id] = server_config
                                     logger.debug(f"Loaded server {server_id} config from env var {env_var}")
                                 else:
                                     logger.warning(f"Invalid or missing URL for HTTP server {server_id} in env var {env_var}.")
                             elif server_type == "stdio":
                                 if "command" in server_config:
                                     servers[server_id] = server_config
                                     logger.debug(f"Loaded server {server_id} config from env var {env_var}")
                                 else:
                                     logger.warning(f"Missing 'command' for stdio server {server_id} in env var {env_var}.")
                             else:
                                 logger.warning(f"Unsupported server type '{server_type}' for server {server_id} in env var {env_var}.")
                        else:
                             logger.warning(f"Invalid JSON format in env var {env_var}")
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON from env var {env_var}")
                elif "=" in value and not env_var.endswith(("_COMMAND", "_ARGS", "_ENV", "_TYPE")): # Assume URL if not a config component
                    server_id = env_var[15:].lower()
                    if self._validate_url(value):
                        servers[server_id] = {"url": value, "type": "http"}
                        logger.debug(f"Loaded server {server_id} URL from env var {env_var}")
                    else:
                         logger.warning(f"Invalid URL in env var {env_var}: {value}")


        # Add default brave-search server if not already defined and key exists
        if "brave-search" not in servers:
            brave_api_key = (
                self.config.get("brave.api_key") or
                os.environ.get("BRAVE_API_KEY")
            )
            if brave_api_key:
                servers["brave-search"] = {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-brave-search"],
                    "env": {
                        "BRAVE_API_KEY": brave_api_key
                    }
                }
                logger.debug("Added default brave-search stdio server config.")
            else:
                logger.debug("Brave API key not found, default brave-search server not added.")

        return servers

    def _validate_url(self, url: str) -> bool:
        """
        Validate a URL
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
        """
        result = []
        for server_id, config in self.servers.items():
            server_info = {"id": server_id, "type": config.get("type", "http")}

            if config.get("type") == "stdio":
                server_info["command"] = config.get("command")
                server_info["args"] = config.get("args", [])

                # Check process status directly
                process = self.process_manager.processes.get(server_id)
                server_info["status"] = "running" if process and process.returncode is None else "stopped"
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
        """
        if not server_id:
            server_id = self.default_server

        if not server_id or server_id not in self.servers:
            return None

        return self.servers[server_id]

    def add_server(self, server_id: str, config: Dict[str, Any]) -> bool:
        """
        Add or update an MCP server configuration.
        """
        # Validate server ID
        if not server_id or not server_id.isalnum():
            raise ValueError("Server ID must be alphanumeric")

        # Validate config structure
        server_type = config.get("type")
        if server_type == "http":
            if not config.get("url") or not self._validate_url(config["url"]):
                 raise ValueError(f"Invalid or missing URL for HTTP server {server_id}")
        elif server_type == "stdio":
             if not config.get("command"):
                 raise ValueError(f"Missing 'command' for stdio server {server_id}")
        else:
             raise ValueError(f"Unsupported server type: {server_type}")

        # Add/Update server config
        self.servers[server_id] = config
        logger.info(f"Added/Updated server config for {server_id}")

        # Save to config file (only saves the whole section)
        all_servers_config = self.config.get_dict("mcp.servers", {})
        all_servers_config[server_id] = config
        self.config.set("mcp.servers", all_servers_config) # Use set for dict

        return True

    async def remove_server(self, server_id: str) -> bool: # Changed to async def
        """
        Remove an MCP server (asynchronously)
        """
        if server_id not in self.servers:
            return False

        # Stop server if running (await is now valid)
        if server_id in self.process_manager.processes:
            stopped = await self.process_manager.stop_process(server_id)
            if not stopped:
                 logger.warning(f"Could not cleanly stop process {server_id} during removal.")

        # Remove from servers dict
        del self.servers[server_id]

        # Remove from config file
        all_servers_config = self.config.get_dict("mcp.servers", {})
        if server_id in all_servers_config:
            del all_servers_config[server_id]
            self.config.set("mcp.servers", all_servers_config) # Use set for dict

        logger.info(f"Removed server {server_id}")
        return True

    def set_default_server(self, server_id: str) -> bool:
        """
        Set default MCP server
        """
        if server_id not in self.servers:
            return False

        self.default_server = server_id
        self.config.set("mcp.default_server", server_id)

        return True

    async def stop_server(self, server_id: str) -> bool:
        """
        Stop a running MCP server
        """
        return await self.process_manager.stop_process(server_id)

    async def _start_stdio_server(self, server_id: str, config: Dict[str, Any]) -> None:
        """
        Start a stdio-based MCP server
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
        READY_SIGNAL = "MCP_SERVER_READY"
        READINESS_TIMEOUT = 5.0 # Seconds to wait for the ready signal

        try:
            process = await self.process_manager.start_process(server_id, cmd, env=env_vars)

            # --- Robust Readiness Check ---
            logger.debug(f"Waiting for '{READY_SIGNAL}' from {server_id} (timeout: {READINESS_TIMEOUT}s)")
            ready = False
            stderr_output = ""
            try:
                # Read stdout line by line until ready signal or timeout
                while True: # Loop until break, timeout, or error
                    try:
                        # Read a line with timeout
                        line_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=READINESS_TIMEOUT)

                        if not line_bytes: # EOF reached - process likely exited
                            logger.warning(f"EOF reached on stdout for {server_id} before ready signal.")
                            break # Exit loop, check return code below

                        line = line_bytes.decode('utf-8', errors='ignore').strip()
                        logger.debug(f"[{server_id} stdout] {line}") # Log server output during startup

                        if READY_SIGNAL in line:
                            logger.info(f"MCP server {server_id} reported ready (PID: {process.pid}).")
                            ready = True
                            break # Exit loop, server is ready

                    except asyncio.TimeoutError:
                        logger.error(f"Timeout waiting for ready signal '{READY_SIGNAL}' from {server_id}.")
                        break # Exit loop on timeout

                # Check if process exited prematurely after the loop
                if process.returncode is not None:
                     try:
                         # Try reading stderr if process exited
                         stderr_bytes = await asyncio.wait_for(process.stderr.read(), timeout=0.5)
                         stderr_output = stderr_bytes.decode('utf-8', errors='ignore')
                     except asyncio.TimeoutError: pass
                     except Exception: pass # Ignore errors reading stderr on exit
                     raise MCPServerConfigError(f"MCP server {server_id} exited prematurely (code: {process.returncode}) before signaling ready. Stderr: {stderr_output[:500]}")

                if not ready:
                     # If loop finished without ready signal and process hasn't exited (timeout case)
                     raise MCPServerConfigError(f"MCP server {server_id} did not signal readiness within {READINESS_TIMEOUT}s.")

            except Exception as readiness_err:
                 # Catch any other errors during readiness check
                 logger.error(f"Error during readiness check for {server_id}: {readiness_err}", exc_info=True)
                 # Attempt to stop the process if it's still running
                 await self.process_manager.stop_process(server_id)
                 raise MCPServerConfigError(f"Failed readiness check for {server_id}: {readiness_err}")
            # --- End Robust Readiness Check ---

        except OSError as e:
            logger.error(f"Error starting MCP server {server_id}: {e}", exc_info=True)
            raise MCPServerConfigError(f"Error starting MCP server {server_id}: {e}")

    @asynccontextmanager
    async def _ensure_server_running(self, server_id: str):
        """
        Ensure a server is running before making a request (async context manager).
        """
        server = self.get_server(server_id)
        if not server:
            raise MCPServerConfigError(f"MCP server not found: {server_id}")

        server_type = server.get("type", "http")

        if server_type == "stdio":
            # For stdio servers, ensure process is running
            process = self.process_manager.processes.get(server_id)
            # Check returncode instead of poll()
            if not process or process.returncode is not None:
                logger.info(f"Stdio server {server_id} not running or exited, attempting to start.")
                await self._start_stdio_server(server_id, server)
                # Re-fetch process after starting
                process = self.process_manager.processes.get(server_id)
                if not process or process.returncode is not None:
                     raise MCPConnectionError(f"Failed to start or keep stdio server {server_id} running.")

        try:
            yield server # Provide server config to the caller context
        except Exception as e:
            logger.error(f"Error during server operation for {server_id}: {e}", exc_info=True)
            raise # Re-raise the exception

    async def _call_stdio_service(
        self,
        server_id: str,
        service: str,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP service method via stdio using asyncio streams.
        """
        async with self._ensure_server_running(server_id) as _:
            process = self.process_manager.processes.get(server_id)
            if not process or process.stdin is None or process.stdout is None:
                raise MCPConnectionError(f"Process or streams not available for server: {server_id}")

            # Build request payload
            request_id = str(time.time_ns()) # Use a unique ID
            payload = {
                "jsonrpc": "2.0",
                "method": f"{service}.{method}",
                "params": params or {},
                "id": request_id
            }

            payload_bytes = (json.dumps(payload) + "\n").encode('utf-8')

            logger.debug(f"Calling stdio MCP service: {service}.{method} on {server_id}")

            try:
                # Send request
                process.stdin.write(payload_bytes)
                await process.stdin.drain()

                # Read response with timeout
                try:
                    response_bytes = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=30.0 # Configurable timeout?
                    )
                except asyncio.TimeoutError:
                    raise MCPConnectionError(f"Timeout waiting for response from MCP server: {server_id}")

                if not response_bytes:
                    # Server closed stdout?
                    raise MCPConnectionError(f"Received empty response (EOF?) from MCP server: {server_id}")

                response_str = response_bytes.decode('utf-8')

                # Parse response
                try:
                    result = json.loads(response_str)
                except json.JSONDecodeError as e:
                    raise MCPConnectionError(f"Invalid JSON response from MCP server {server_id}: {e}. Response: {response_str[:200]}")

                # Check for JSON-RPC error
                if "error" in result:
                    error = result["error"]
                    error_code = error.get("code", "N/A")
                    error_msg = error.get("message", "Unknown error")
                    logger.error(f"MCP service error from {server_id}: Code {error_code}, Msg: {error_msg}")
                    raise MCPExecutionError(f"MCP service error ({error_code}): {error_msg}")

                # Check if the response ID matches the request ID (optional but good practice)
                if result.get("id") != request_id:
                     logger.warning(f"MCP response ID mismatch for {server_id}. Req: {request_id}, Resp: {result.get('id')}")


                return result.get("result", {})

            except (BrokenPipeError, ConnectionResetError) as e:
                 logger.error(f"Connection error with stdio server {server_id}: {e}", exc_info=True)
                 # Attempt to stop the potentially broken process
                 await self.process_manager.stop_process(server_id)
                 raise MCPConnectionError(f"Connection error with stdio server {server_id}: {e}")
            except (json.JSONDecodeError, MCPConnectionError, MCPExecutionError) as e:
                # Re-raise specific exceptions
                raise
            except Exception as e:
                logger.error(f"Unexpected error in stdio MCP service call to {server_id}: {e}", exc_info=True)
                raise MCPConnectionError(f"Unexpected error in stdio MCP service call: {e}")

    async def call_service(
        self,
        service: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        server_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Call an MCP service method (stdio or http).
        """
        target_server_id = server_id or self.default_server
        if not target_server_id:
            raise MCPServerConfigError("No MCP server specified and no default server set")

        # --- Consent Check Placeholder ---
        # TODO: Implement actual consent check logic here.
        # This should:
        # 1. Check configuration for consent requirements (allow, deny, ask).
        # 2. If 'ask', call a handler (passed during MCPClient init or set later)
        #    to prompt the user for confirmation.
        # 3. The handler should receive details: target_server_id, service, method, params.
        # 4. If consent is denied by the handler/user, raise MCPConsentDeniedError.
        # Example:
        # consent_required = self._check_consent_config(target_server_id, service, method)
        # if consent_required == "ask" and self.consent_handler:
        #     approved = await self.consent_handler(target_server_id, service, method, params)
        #     if not approved:
        #         raise MCPConsentDeniedError(f"User denied consent for {service}.{method} on {target_server_id}")
        # elif consent_required == "deny":
        #      raise MCPConsentDeniedError(f"Action {service}.{method} on {target_server_id} is denied by configuration.")
        logger.debug(f"Proceeding with MCP call to {target_server_id} for {service}.{method} (Consent check placeholder)")
        # --- Consent Check ---
        consent_policy = self._get_consent_policy(target_server_id, service, method)
        logger.debug(f"MCP Consent policy for {target_server_id}.{service}.{method}: {consent_policy}")

        if consent_policy == "deny":
            raise MCPConsentDeniedError(f"Action {service}.{method} on {target_server_id} is denied by configuration.")
        elif consent_policy == "ask":
            if not self.consent_handler:
                raise MCPServerConfigError(f"Consent policy is 'ask' for {service}.{method} on {target_server_id}, but no consent handler is configured.")

            # Call the provided async consent handler
            try:
                approved = await self.consent_handler(target_server_id, service, method, params or {})
                if not approved:
                    raise MCPConsentDeniedError(f"User denied consent for {service}.{method} on {target_server_id}")
                logger.debug(f"User approved MCP action: {target_server_id}.{service}.{method}")
            except Exception as e:
                 logger.error(f"Error during consent handling for {target_server_id}.{service}.{method}: {e}", exc_info=True)
                 # Treat errors in consent handling as denial for safety
                 raise MCPConsentDeniedError(f"Error during consent handling for {service}.{method} on {target_server_id}: {e}")

        # If policy is 'allow' or 'ask' was approved, proceed.
        # --- End Consent Check ---


        async with self._ensure_server_running(target_server_id) as server:
            server_type = server.get("type", "http")

            if server_type == "stdio":
                return await self._call_stdio_service(target_server_id, service, method, params or {})
            else:
                # HTTP server
                url = server.get("url")
                if not url:
                    raise MCPServerConfigError(f"URL not specified for HTTP MCP server: {target_server_id}")

                # Build request payload
                request_id = str(time.time_ns())
                payload = {
                    "jsonrpc": "2.0",
                    "method": f"{service}.{method}",
                    "params": params or {},
                    "id": request_id
                }

                logger.debug(f"Calling HTTP MCP service: {service}.{method} on {target_server_id} ({url})")

                try:
                    # Make async request using httpx client
                    response = await self.http_client.post(url, json=payload)

                    # Raise for HTTP errors (4xx, 5xx)
                    response.raise_for_status()

                    # Parse response
                    try:
                        result = response.json()
                    except json.JSONDecodeError as e:
                        raise MCPConnectionError(f"Invalid JSON response from MCP server {target_server_id}: {e}. Response: {response.text[:200]}")

                    # Check for JSON-RPC error
                    if "error" in result:
                        error = result["error"]
                        error_code = error.get("code", "N/A")
                        error_msg = error.get("message", "Unknown error")
                        logger.error(f"MCP service error from {target_server_id}: Code {error_code}, Msg: {error_msg}")
                        raise MCPExecutionError(f"MCP service error ({error_code}): {error_msg}")

                    # Check response ID
                    if result.get("id") != request_id:
                         logger.warning(f"MCP response ID mismatch for {target_server_id}. Req: {request_id}, Resp: {result.get('id')}")

                    return result.get("result", {})

                except httpx.RequestError as e:
                    logger.error(f"HTTP request error calling MCP server {target_server_id}: {e}", exc_info=True)
                    raise MCPConnectionError(f"MCP service request error to {target_server_id}: {e}")
                except httpx.HTTPStatusError as e:
                     logger.error(f"HTTP status error from MCP server {target_server_id}: {e.response.status_code} - {e.response.text[:200]}", exc_info=True)
                     raise MCPConnectionError(f"MCP server {target_server_id} returned status {e.response.status_code}")
                except (json.JSONDecodeError, MCPConnectionError, MCPExecutionError) as e:
                    # Re-raise specific exceptions
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error in HTTP MCP service call to {target_server_id}: {e}", exc_info=True)
                    raise MCPConnectionError(f"Unexpected error in HTTP MCP service call: {e}")

    def _get_consent_policy(self, server_id: str, service: str, method: str) -> str:
        """
        Determine the consent policy based on configuration rules.

        Checks rules in order of specificity:
        1. server.service.method
        2. server.service.*
        3. server.*.*
        4. *.service.method
        5. *.service.*
        6. Default policy

        Returns:
            'allow', 'deny', or 'ask'
        """
        # Use the updated configuration keys
        rules = self.config.get_dict("mcp.consent_rules", {})
        default_policy = self.config.get_string("mcp.consent_default_policy", "ask")

        # Define rule patterns to check in order of specificity
        patterns_to_check = [
            f"{server_id}.{service}.{method}",
            f"{server_id}.{service}.*",
            f"{server_id}.*.*",
            f"*.{service}.{method}",
            f"*.{service}.*",
        ]

        for pattern in patterns_to_check:
            if pattern in rules:
                policy = rules[pattern]
                if policy in ["allow", "deny", "ask"]:
                    return policy
                else:
                    logger.warning(f"Invalid consent policy '{policy}' found for rule '{pattern}'. Using default.")
                    # Fall through to default if rule value is invalid

        # No specific rule matched, use default
        return default_policy


# Base class for all MCP services
class MCPBaseService:
    """Base class for MCP services"""

    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
        """
        Initialize MCP service
        """
        self.client = client
        self.server_id = server_id # Specific server for this service instance, or None for default
        # Derive service name from class name (e.g., MCPMemoryService -> memory)
        self.service_name = self.__class__.__name__.replace("MCP", "").replace("Service", "").lower()

    async def call_method(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a service method on the configured server (or default).
        """
        return await self.client.call_service(
            self.service_name,
            method,
            params,
            self.server_id # Pass specific server_id if set for this instance
        )


class MCPMemoryService(MCPBaseService):
    """MCP Memory service client"""
    # Service name is automatically derived as "memory"

    async def create_entities(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.call_method("create_entities", {"entities": entities})

    async def create_relations(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.call_method("create_relations", {"relations": relations})

    async def add_observations(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.call_method("add_observations", {"observations": observations})

    async def delete_entities(self, entity_names: List[str]) -> Dict[str, Any]:
        return await self.call_method("delete_entities", {"entityNames": entity_names})

    async def delete_observations(self, deletions: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.call_method("delete_observations", {"deletions": deletions})

    async def delete_relations(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        return await self.call_method("delete_relations", {"relations": relations})

    async def read_graph(self) -> Dict[str, Any]:
        return await self.call_method("read_graph", {})

    async def search_nodes(self, query: str) -> Dict[str, Any]:
        return await self.call_method("search_nodes", {"query": query})

    async def open_nodes(self, names: List[str]) -> Dict[str, Any]:
        return await self.call_method("open_nodes", {"names": names})


class MCPFetchService(MCPBaseService):
    """MCP Fetch service client"""
    # Service name is automatically derived as "fetch"

    async def fetch(
        self,
        url: str,
        max_length: int = 5000,
        raw: bool = False,
        start_index: int = 0
    ) -> Dict[str, Any]:
        """
        Fetch a URL via MCP service.
        """
        # Basic URL validation (moved from original call_method)
        if not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL scheme: {url}")
        dangerous_patterns = ["file://", "ftp://", "data:"]
        if any(pattern in url for pattern in dangerous_patterns):
            raise ValueError(f"Rejected potentially dangerous URL: {url}")

        return await self.call_method("fetch", {
            "url": url,
            "max_length": max_length,
            "raw": raw,
            "start_index": start_index
        })


class MCPBraveSearchService(MCPBaseService):
    """MCP Brave Search service client"""
    # Service name is automatically derived as "bravesearch" - needs adjustment if service name is different
    def __init__(self, client: MCPClient, server_id: Optional[str] = None):
         super().__init__(client, server_id)
         self.service_name = "brave-search" # Explicitly set if derivation is wrong

    async def brave_web_search(self, query: str, count: int = 10, offset: int = 0) -> Dict[str, Any]:
        """
        Perform a web search via MCP service, with direct fallback.
        """
        # Validate parameters
        if not query:
            raise ValueError("Search query cannot be empty")
        count = min(max(1, count), 20)
        offset = max(0, offset)

        try:
            # Try via MCP service first
            return await self.call_method("brave_web_search", {
                "query": query,
                "count": count,
                "offset": offset
            })
        except (MCPConnectionError, MCPExecutionError, MCPServerConfigError) as e:
            logger.warning(f"MCP brave_web_search failed ({type(e).__name__}), falling back to direct API: {e}")
            # Fallback to direct API call using httpx
            return await self._direct_brave_api_call(query, count, offset)
        except Exception as e:
             logger.error(f"Unexpected error during MCP brave_web_search: {e}", exc_info=True)
             raise # Re-raise unexpected errors

    async def _direct_brave_api_call(self, query: str, count: int, offset: int) -> Dict[str, Any]:
        """
        Make a direct API call to Brave Search using httpx.
        """
        # Get API key
        brave_api_key = (
            self.client.config.get("brave.api_key") or
            os.environ.get("BRAVE_API_KEY")
        )
        if not brave_api_key:
            raise MCPExecutionError("Brave API key not found for direct fallback")

        # Prepare request
        headers = {"X-Subscription-Token": brave_api_key, "Accept": "application/json"}
        params = {"q": query, "count": count, "offset": offset}
        url = "https://api.search.brave.com/res/v1/web/search"

        try:
            # Make async request using the client's httpx instance
            response = await self.client.http_client.get(url, headers=headers, params=params)
            response.raise_for_status() # Raise for HTTP errors
            return response.json()

        except httpx.RequestError as e:
            raise MCPConnectionError(f"Direct Brave Search API request error: {e}")
        except httpx.HTTPStatusError as e:
            raise MCPConnectionError(f"Direct Brave Search API returned status {e.response.status_code}")
        except json.JSONDecodeError as e:
            raise MCPConnectionError(f"Invalid JSON response from direct Brave Search API: {e}")
        except Exception as e:
            raise MCPExecutionError(f"Unexpected error calling direct Brave Search API: {e}")

    # Alias for compatibility
    async def search(self, query: str, count: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Alias for brave_web_search"""
        return await self.brave_web_search(query, count, offset)

    async def local_search(self, query: str, count: int = 5) -> Dict[str, Any]:
        """
        Perform a local search via MCP service, falling back to web search.
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
        except (MCPConnectionError, MCPExecutionError, MCPServerConfigError) as e:
            logger.warning(f"MCP local_search failed ({type(e).__name__}), falling back to web search: {e}")
            # Fallback to regular web search
            return await self.brave_web_search(query, count)
        except Exception as e:
             logger.error(f"Unexpected error during MCP local_search: {e}", exc_info=True)
             raise


class MCPGitHubService(MCPBaseService):
    """MCP GitHub service client"""
    # Service name is automatically derived as "github"

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
        Create or update a file via MCP service.
        """
        # Validate parameters
        if not all([owner, repo, path, message, branch]):
            raise ValueError("Missing required parameters for create_or_update_file")

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

    def __init__(self, config: Optional[Config] = None, consent_handler: Optional[ConsentHandler] = None):
        """
        Initialize MCP manager

        Args:
            config: Configuration object.
            consent_handler: An async function to call when user consent is needed.
        """
        self.config = config or Config()
        # Pass the consent_handler down to the MCPClient
        self.client = MCPClient(self.config, consent_handler=consent_handler)

        # Initialize services (pass the client)
        self.memory = MCPMemoryService(self.client)
        self.fetch = MCPFetchService(self.client)
        self.brave_search = MCPBraveSearchService(self.client)
        self.github = MCPGitHubService(self.client)

    def list_servers(self) -> List[Dict[str, Any]]:
        """
        List available MCP servers
        """
        return self.client.list_servers()

    def add_server(self, server_id: str, config: Dict[str, Any]) -> bool:
        """
        Add or update an MCP server configuration.
        """
        return self.client.add_server(server_id, config)

    async def remove_server(self, server_id: str) -> bool: # Changed to async
        """
        Remove an MCP server (asynchronously).
        """
        return await self.client.remove_server(server_id) # await the client method

    def set_default_server(self, server_id: str) -> bool:
        """
        Set default MCP server
        """
        return self.client.set_default_server(server_id)

    async def stop_server(self, server_id: str) -> bool:
        """
        Stop a running MCP server
        """
        return await self.client.stop_server(server_id)

    async def stop_all_servers(self) -> None:
        """Stop all running MCP servers"""
        await self.client.close() # Use the client's close method which handles cleanup

    # Expose call_service directly for flexibility?
    async def call_service(
        self,
        service: str,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        server_id: Optional[str] = None
    ) -> Dict[str, Any]:
         """Directly call a service method on a specified or default server."""
         return await self.client.call_service(service, method, params, server_id)
