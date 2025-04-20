"""
Process management tool handlers for Fei.

This module provides handlers for process management tools, including
viewing process output, sending input to processes, checking process status,
and terminating processes.
"""

import os
import signal
import time
from typing import Dict, List, Optional, Any

from fei.utils.logging import get_logger
from fei.core.process_manager import process_manager
from fei.tools.shell_runner import shell_runner

logger = get_logger(__name__)


def view_process_output_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for view_process_output tool.
    
    Args:
        args: Tool arguments
            - pid: Process ID
            - max_lines: Maximum number of lines to return (default: 50)
            - include_stderr: Whether to include stderr (default: True)
            
    Returns:
        Process output information
    """
    pid = args.get("pid")
    max_lines = args.get("max_lines", 50)
    include_stderr = args.get("include_stderr", True)

    if pid is None:
        return {"error": "PID is required"}

    try:
        # Convert pid to integer
        pid = int(pid)
        
        # Get process output
        output_info = process_manager.get_process_output(
            pid=pid,
            max_lines=max_lines,
            include_stderr=include_stderr
        )
        
        return output_info
    except Exception as e:
        logger.error(f"Error in view_process_output_handler for PID {pid}: {e}", exc_info=True)
        return {"error": str(e)}


def send_process_input_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for send_process_input tool.
    
    Args:
        args: Tool arguments
            - pid: Process ID
            - input_string: Input string to send
            
    Returns:
        Result information
    """
    pid = args.get("pid")
    input_string = args.get("input_string")

    if pid is None:
        return {"error": "PID is required"}
    if input_string is None:
        return {"error": "input_string is required"}

    try:
        # Convert pid to integer
        pid = int(pid)
        
        # Send input to process
        result = process_manager.send_process_input(
            pid=pid,
            input_str=input_string
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in send_process_input_handler for PID {pid}: {e}", exc_info=True)
        return {"error": str(e)}


def check_process_status_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for check_process_status tool.
    
    Args:
        args: Tool arguments
            - pid: Process ID
            
    Returns:
        Process status information
    """
    pid = args.get("pid")

    if pid is None:
        return {"error": "PID is required"}

    try:
        # Convert pid to integer
        pid = int(pid)
        
        # Check process status
        status_info = process_manager.check_process_status(pid=pid)
        
        return status_info
    except Exception as e:
        logger.error(f"Error in check_process_status_handler for PID {pid}: {e}", exc_info=True)
        return {"error": str(e)}


def terminate_process_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for terminate_process tool.
    
    Args:
        args: Tool arguments
            - pid: Process ID
            - force: Whether to force termination (default: False)
            - timeout: Timeout in seconds before sending SIGKILL (default: 3.0)
            
    Returns:
        Termination result information
    """
    pid = args.get("pid")
    force = args.get("force", False)
    timeout = args.get("timeout", 3.0)

    if pid is None:
        return {"error": "PID is required"}

    try:
        # Convert pid to integer
        pid = int(pid)
        
        # Terminate process
        result = process_manager.terminate_process(
            pid=pid,
            force=force,
            timeout=timeout
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in terminate_process_handler for PID {pid}: {e}", exc_info=True)
        return {"error": str(e)}


def list_processes_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for list_processes tool.
    
    Args:
        args: Tool arguments (none required)
            
    Returns:
        List of process information
    """
    try:
        # List processes
        processes = process_manager.list_processes()
        
        return {
            "processes": processes,
            "count": len(processes)
        }
    except Exception as e:
        logger.error(f"Error in list_processes_handler: {e}", exc_info=True)
        return {"error": str(e)}
