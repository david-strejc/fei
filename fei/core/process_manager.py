"""
Process manager for Fei.

This module provides a comprehensive system for managing processes spawned by Fei,
including interactive commands, long-running tasks, and proper process management.
"""

import os
import sys
import time
import signal
import tempfile
import threading
import subprocess
import select
import fcntl
import atexit
import shlex
import json
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable, BinaryIO, TextIO
from pathlib import Path
from datetime import datetime

from fei.utils.logging import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_TIMEOUT = 3600  # 1 hour
MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_OUTPUT_LINES = 1000
BUFFER_SIZE = 4096
OUTPUT_POLL_INTERVAL = 0.1  # seconds


class ProcessError(Exception):
    """Exception raised for process-related errors."""
    pass


class ProcessNotFoundError(ProcessError):
    """Exception raised when a process is not found."""
    pass


class ProcessManager:
    """
    Manages processes spawned by Fei.
    
    This class provides a comprehensive system for managing processes,
    including interactive commands, long-running tasks, and proper
    process management.
    """
    
    def __init__(
        self,
        temp_dir: Optional[str] = None,
        max_output_size: int = MAX_OUTPUT_SIZE,
        output_poll_interval: float = OUTPUT_POLL_INTERVAL
    ):
        """
        Initialize the process manager.
        
        Args:
            temp_dir: Directory for temporary files (defaults to system temp dir)
            max_output_size: Maximum size of output files in bytes
            output_poll_interval: Interval for polling output in seconds
        """
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_output_size = max_output_size
        self.output_poll_interval = output_poll_interval
        
        # Process tracking
        self.processes: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        # Output readers
        self.output_readers: Dict[int, threading.Thread] = {}
        self._stop_event = threading.Event()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _create_temp_file(self, prefix: str, suffix: str) -> str:
        """
        Create a temporary file.
        
        Args:
            prefix: Prefix for the file name
            suffix: Suffix for the file name
            
        Returns:
            Path to the temporary file
        """
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.temp_dir)
        os.close(fd)  # Close the file descriptor
        return path
    
    def _make_non_blocking(self, file_obj: Union[TextIO, BinaryIO]) -> None:
        """
        Make a file object non-blocking.
        
        Args:
            file_obj: File object to make non-blocking
        """
        fd = file_obj.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    
    def _read_non_blocking(self, file_obj: Union[TextIO, BinaryIO], size: int = BUFFER_SIZE) -> str:
        """
        Read from a non-blocking file object.
        
        Args:
            file_obj: File object to read from
            size: Maximum number of bytes to read
            
        Returns:
            Data read from the file object
        """
        try:
            return file_obj.read(size) or ""
        except (IOError, BlockingIOError):
            return ""
    
    def _output_reader_thread(self, pid: int) -> None:
        """
        Thread function for reading process output.
        
        Args:
            pid: Process ID
        """
        try:
            process_info = self.processes.get(pid)
            if not process_info:
                logger.warning(f"Process {pid} not found for output reader")
                return
            
            process = process_info["process"]
            stdout_file = open(process_info["stdout_file"], "a")
            stderr_file = open(process_info["stderr_file"], "a")
            
            # Make stdout and stderr non-blocking
            self._make_non_blocking(process.stdout)
            self._make_non_blocking(process.stderr)
            
            # Read output until process exits or stop event is set
            while not self._stop_event.is_set():
                # Check if process has exited
                if process.poll() is not None:
                    # Read any remaining output
                    stdout_data = self._read_non_blocking(process.stdout)
                    stderr_data = self._read_non_blocking(process.stderr)
                    
                    if stdout_data:
                        stdout_file.write(stdout_data)
                        stdout_file.flush()
                    
                    if stderr_data:
                        stderr_file.write(stderr_data)
                        stderr_file.flush()
                    
                    # Update process info
                    with self._lock:
                        if pid in self.processes:
                            self.processes[pid]["status"] = "exited"
                            self.processes[pid]["exit_code"] = process.returncode
                            self.processes[pid]["end_time"] = time.time()
                    
                    # Close files
                    stdout_file.close()
                    stderr_file.close()
                    
                    logger.info(f"Process {pid} exited with code {process.returncode}")
                    return
                
                # Read output
                stdout_data = self._read_non_blocking(process.stdout)
                stderr_data = self._read_non_blocking(process.stderr)
                
                if stdout_data:
                    stdout_file.write(stdout_data)
                    stdout_file.flush()
                
                if stderr_data:
                    stderr_file.write(stderr_data)
                    stderr_file.flush()
                
                # Sleep briefly to avoid high CPU usage
                time.sleep(self.output_poll_interval)
        
        except Exception as e:
            logger.error(f"Error in output reader for process {pid}: {e}", exc_info=True)
            
            # Update process info
            with self._lock:
                if pid in self.processes:
                    self.processes[pid]["status"] = "error"
                    self.processes[pid]["error"] = str(e)
    
    def spawn_process(
        self,
        command: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: Optional[int] = None,
        is_interactive: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Spawn a new process and track it.
        
        Args:
            command: Command to execute
            env: Environment variables
            cwd: Working directory
            timeout: Timeout in seconds
            is_interactive: Whether the command is interactive
            
        Returns:
            Process information
        """
        # Create temporary files for output
        prefix = f"fei_process_{int(time.time())}_"
        stdout_file = self._create_temp_file(prefix, "_stdout.txt")
        stderr_file = self._create_temp_file(prefix, "_stderr.txt")
        input_file = self._create_temp_file(prefix, "_input.txt")
        
        # Create process
        try:
            # Use shell=True for complex commands
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                cwd=cwd,
                env=env or os.environ.copy(),
                start_new_session=True  # Create a new process group
            )
            
            # Get process ID
            pid = process.pid
            
            # Create process info
            process_info = {
                "pid": pid,
                "command": command,
                "start_time": time.time(),
                "process": process,
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "input_file": input_file,
                "timeout": timeout or DEFAULT_TIMEOUT,
                "status": "running",
                "exit_code": None,
                "end_time": None,
                "is_interactive": is_interactive,
                "cwd": cwd or os.getcwd(),
                "env": env or {},
                "error": None
            }
            
            # Store process info
            with self._lock:
                self.processes[pid] = process_info
            
            # Start output reader thread
            reader_thread = threading.Thread(
                target=self._output_reader_thread,
                args=(pid,),
                daemon=True
            )
            reader_thread.start()
            self.output_readers[pid] = reader_thread
            
            # Start timeout thread if timeout is specified
            if timeout:
                def terminate_after_timeout():
                    time.sleep(timeout)
                    with self._lock:
                        if pid in self.processes and self.processes[pid]["status"] == "running":
                            logger.info(f"Terminating process {pid} after timeout")
                            self.terminate_process(pid, force=True)
                
                timeout_thread = threading.Thread(
                    target=terminate_after_timeout,
                    daemon=True
                )
                timeout_thread.start()
            
            logger.info(f"Spawned process {pid}: {command}")
            
            # Return process info (without the actual process object)
            return {
                "pid": pid,
                "command": command,
                "start_time": process_info["start_time"],
                "stdout_file": stdout_file,
                "stderr_file": stderr_file,
                "input_file": input_file,
                "timeout": timeout or DEFAULT_TIMEOUT,
                "status": "running",
                "is_interactive": is_interactive
            }
        
        except Exception as e:
            logger.error(f"Error spawning process: {e}", exc_info=True)
            
            # Clean up temporary files
            for file_path in [stdout_file, stderr_file, input_file]:
                try:
                    os.unlink(file_path)
                except:
                    pass
            
            raise ProcessError(f"Failed to spawn process: {e}")
    
    def get_process_output(
        self,
        pid: int,
        max_lines: int = MAX_OUTPUT_LINES,
        include_stderr: bool = True
    ) -> Dict[str, Any]:
        """
        Get the output of a process.
        
        Args:
            pid: Process ID
            max_lines: Maximum number of lines to return
            include_stderr: Whether to include stderr
            
        Returns:
            Process output information
            
        Raises:
            ProcessNotFoundError: If the process is not found
        """
        with self._lock:
            process_info = self.processes.get(pid)
            if not process_info:
                raise ProcessNotFoundError(f"Process {pid} not found")
            
            # Check if stdout file exists
            if not os.path.exists(process_info["stdout_file"]):
                return {
                    "pid": pid,
                    "stdout": "",
                    "stderr": "",
                    "status": process_info["status"],
                    "exit_code": process_info["exit_code"],
                    "error": "Output files not found"
                }
            
            # Read stdout
            try:
                with open(process_info["stdout_file"], "r") as f:
                    stdout_lines = f.readlines()
                    
                    # Limit number of lines
                    if max_lines > 0 and len(stdout_lines) > max_lines:
                        stdout_lines = stdout_lines[-max_lines:]
                    
                    stdout = "".join(stdout_lines)
            except Exception as e:
                logger.error(f"Error reading stdout for process {pid}: {e}", exc_info=True)
                stdout = f"Error reading stdout: {e}"
            
            # Read stderr if requested
            stderr = ""
            if include_stderr:
                try:
                    with open(process_info["stderr_file"], "r") as f:
                        stderr_lines = f.readlines()
                        
                        # Limit number of lines
                        if max_lines > 0 and len(stderr_lines) > max_lines:
                            stderr_lines = stderr_lines[-max_lines:]
                        
                        stderr = "".join(stderr_lines)
                except Exception as e:
                    logger.error(f"Error reading stderr for process {pid}: {e}", exc_info=True)
                    stderr = f"Error reading stderr: {e}"
            
            # Get process status
            status = process_info["status"]
            exit_code = process_info["exit_code"]
            
            return {
                "pid": pid,
                "stdout": stdout,
                "stderr": stderr,
                "status": status,
                "exit_code": exit_code
            }
    
    def send_process_input(self, pid: int, input_str: str) -> Dict[str, Any]:
        """
        Send input to a process.
        
        Args:
            pid: Process ID
            input_str: Input string
            
        Returns:
            Result information
            
        Raises:
            ProcessNotFoundError: If the process is not found
        """
        with self._lock:
            process_info = self.processes.get(pid)
            if not process_info:
                raise ProcessNotFoundError(f"Process {pid} not found")
            
            # Check if process is still running
            process = process_info["process"]
            if process.poll() is not None:
                return {
                    "success": False,
                    "error": f"Process {pid} has already exited (code: {process.poll()})"
                }
            
            # Check if stdin is available
            if not process.stdin or process.stdin.closed:
                return {
                    "success": False,
                    "error": f"Stdin for process {pid} is not available or closed"
                }
            
            # Append input to input file for record keeping
            try:
                with open(process_info["input_file"], "a") as f:
                    f.write(f"{datetime.now().isoformat()}: {input_str}\n")
            except Exception as e:
                logger.error(f"Error writing to input file for process {pid}: {e}", exc_info=True)
            
            # Send input to process
            try:
                # Add newline if not present
                if not input_str.endswith('\n'):
                    input_str += '\n'
                
                process.stdin.write(input_str)
                process.stdin.flush()
                
                return {
                    "success": True,
                    "message": f"Input sent to process {pid}"
                }
            except Exception as e:
                logger.error(f"Error sending input to process {pid}: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to send input: {e}"
                }
    
    def check_process_status(self, pid: int) -> Dict[str, Any]:
        """
        Check if a process is running.
        
        Args:
            pid: Process ID
            
        Returns:
            Process status information
        """
        # Check if it's a managed process
        with self._lock:
            process_info = self.processes.get(pid)
            if process_info:
                # Get process status
                process = process_info["process"]
                exit_code = process.poll()
                
                # Update status if needed
                if exit_code is not None and process_info["status"] == "running":
                    process_info["status"] = "exited"
                    process_info["exit_code"] = exit_code
                    process_info["end_time"] = time.time()
                
                return {
                    "pid": pid,
                    "status": process_info["status"],
                    "exit_code": process_info["exit_code"],
                    "command": process_info["command"],
                    "start_time": process_info["start_time"],
                    "end_time": process_info["end_time"],
                    "managed": True
                }
        
        # Check if it's an unmanaged process
        try:
            # Send signal 0 to check if process exists
            os.kill(pid, 0)
            
            # Process exists but is not managed
            return {
                "pid": pid,
                "status": "running",
                "exit_code": None,
                "managed": False,
                "message": "Process exists but is not managed by Fei"
            }
        except ProcessLookupError:
            # Process does not exist
            return {
                "pid": pid,
                "status": "not_found",
                "exit_code": None,
                "managed": False,
                "message": "Process not found or already terminated"
            }
        except PermissionError:
            # Process exists but we don't have permission to check it
            return {
                "pid": pid,
                "status": "running",
                "exit_code": None,
                "managed": False,
                "message": "Process exists but permission denied to check status"
            }
        except Exception as e:
            logger.error(f"Error checking process status for {pid}: {e}", exc_info=True)
            return {
                "pid": pid,
                "status": "error",
                "exit_code": None,
                "managed": False,
                "error": str(e)
            }
    
    def terminate_process(
        self,
        pid: int,
        force: bool = False,
        timeout: float = 3.0
    ) -> Dict[str, Any]:
        """
        Terminate a process.
        
        Args:
            pid: Process ID
            force: Whether to force termination (SIGKILL)
            timeout: Timeout in seconds before sending SIGKILL
            
        Returns:
            Termination result information
        """
        # Check if it's a managed process
        with self._lock:
            process_info = self.processes.get(pid)
            if not process_info:
                # Try to terminate unmanaged process
                try:
                    # Check if process exists
                    os.kill(pid, 0)
                    
                    # Try to terminate process group
                    try:
                        pgid = os.getpgid(pid)
                        if force:
                            os.killpg(pgid, signal.SIGKILL)
                        else:
                            os.killpg(pgid, signal.SIGTERM)
                        
                        return {
                            "success": True,
                            "message": f"Sent {'SIGKILL' if force else 'SIGTERM'} to unmanaged process group {pgid}"
                        }
                    except:
                        # Fall back to terminating process directly
                        if force:
                            os.kill(pid, signal.SIGKILL)
                        else:
                            os.kill(pid, signal.SIGTERM)
                        
                        return {
                            "success": True,
                            "message": f"Sent {'SIGKILL' if force else 'SIGTERM'} to unmanaged process {pid}"
                        }
                except ProcessLookupError:
                    return {
                        "success": False,
                        "error": f"Process {pid} not found"
                    }
                except PermissionError:
                    return {
                        "success": False,
                        "error": f"Permission denied to terminate process {pid}"
                    }
                except Exception as e:
                    logger.error(f"Error terminating unmanaged process {pid}: {e}", exc_info=True)
                    return {
                        "success": False,
                        "error": f"Failed to terminate process: {e}"
                    }
            
            # Get process
            process = process_info["process"]
            
            # Check if process has already exited
            if process.poll() is not None:
                # Update status if needed
                if process_info["status"] == "running":
                    process_info["status"] = "exited"
                    process_info["exit_code"] = process.returncode
                    process_info["end_time"] = time.time()
                
                return {
                    "success": True,
                    "message": f"Process {pid} already exited with code {process.returncode}"
                }
            
            # Try to terminate process group
            try:
                pgid = os.getpgid(pid)
                
                # Send SIGTERM first
                os.killpg(pgid, signal.SIGTERM)
                
                # Wait for process to exit
                for _ in range(int(timeout * 10)):
                    if process.poll() is not None:
                        # Process exited
                        process_info["status"] = "exited"
                        process_info["exit_code"] = process.returncode
                        process_info["end_time"] = time.time()
                        
                        return {
                            "success": True,
                            "message": f"Process {pid} terminated with code {process.returncode}"
                        }
                    
                    time.sleep(0.1)
                
                # Force kill if requested and process is still running
                if force and process.poll() is None:
                    os.killpg(pgid, signal.SIGKILL)
                    
                    # Wait briefly for process to exit
                    for _ in range(5):
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    
                    # Update status
                    if process.poll() is not None:
                        process_info["status"] = "exited"
                        process_info["exit_code"] = process.returncode
                        process_info["end_time"] = time.time()
                        
                        return {
                            "success": True,
                            "message": f"Process {pid} forcefully terminated with code {process.returncode}"
                        }
                    else:
                        return {
                            "success": False,
                            "error": f"Failed to forcefully terminate process {pid}"
                        }
                elif process.poll() is None:
                    return {
                        "success": False,
                        "error": f"Process {pid} did not terminate within {timeout} seconds and force=False"
                    }
            except Exception as e:
                logger.error(f"Error terminating process {pid}: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": f"Failed to terminate process: {e}"
                }
    
    def list_processes(self) -> List[Dict[str, Any]]:
        """
        List all tracked processes.
        
        Returns:
            List of process information
        """
        with self._lock:
            result = []
            
            for pid, process_info in self.processes.items():
                # Check if process has exited
                process = process_info["process"]
                exit_code = process.poll()
                
                # Update status if needed
                if exit_code is not None and process_info["status"] == "running":
                    process_info["status"] = "exited"
                    process_info["exit_code"] = exit_code
                    process_info["end_time"] = time.time()
                
                # Add process info to result (without the actual process object)
                result.append({
                    "pid": pid,
                    "command": process_info["command"],
                    "status": process_info["status"],
                    "exit_code": process_info["exit_code"],
                    "start_time": process_info["start_time"],
                    "end_time": process_info["end_time"],
                    "is_interactive": process_info["is_interactive"]
                })
            
            return result
    
    def cleanup(self) -> None:
        """Clean up all tracked processes."""
        logger.info("Cleaning up processes")
        
        # Signal threads to stop
        self._stop_event.set()
        
        # Terminate all processes
        with self._lock:
            for pid, process_info in list(self.processes.items()):
                try:
                    # Check if process has already exited
                    process = process_info["process"]
                    if process.poll() is None:
                        # Try to terminate process group
                        try:
                            pgid = os.getpgid(pid)
                            logger.info(f"Terminating process group {pgid} (PID {pid})")
                            os.killpg(pgid, signal.SIGTERM)
                            
                            # Wait briefly for termination
                            for _ in range(5):
                                if process.poll() is not None:
                                    break
                                time.sleep(0.1)
                            
                            # Force kill if still running
                            if process.poll() is None:
                                logger.info(f"Force killing process group {pgid} (PID {pid})")
                                os.killpg(pgid, signal.SIGKILL)
                        except:
                            # Fall back to terminating process directly
                            logger.info(f"Terminating process {pid}")
                            process.terminate()
                            
                            # Wait briefly for termination
                            for _ in range(5):
                                if process.poll() is not None:
                                    break
                                time.sleep(0.1)
                            
                            # Force kill if still running
                            if process.poll() is None:
                                logger.info(f"Force killing process {pid}")
                                process.kill()
                except Exception as e:
                    logger.error(f"Error terminating process {pid}: {e}", exc_info=True)
            
            # Wait for reader threads to finish
            for pid, thread in list(self.output_readers.items()):
                try:
                    thread.join(1.0)
                except:
                    pass
            
            # Clean up temporary files
            for pid, process_info in list(self.processes.items()):
                try:
                    # Remove temporary files
                    for file_path in [
                        process_info["stdout_file"],
                        process_info["stderr_file"],
                        process_info["input_file"]
                    ]:
                        try:
                            if os.path.exists(file_path):
                                os.unlink(file_path)
                        except:
                            pass
                except Exception as e:
                    logger.error(f"Error cleaning up files for process {pid}: {e}", exc_info=True)
            
            # Clear process tracking
            self.processes.clear()
            self.output_readers.clear()


# Global instance
process_manager = ProcessManager()


def initialize_process_manager(temp_dir: Optional[str] = None) -> None:
    """
    Initialize or update the process manager.
    
    Args:
        temp_dir: Directory for temporary files
    """
    global process_manager
    process_manager = ProcessManager(temp_dir=temp_dir)
