#!/usr/bin/env python3
"""
Example demonstrating the process management system in Fei.

This script shows how to use the process management tools to run and interact with
background processes.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the parent directory to the path so we can import fei
sys.path.insert(0, str(Path(__file__).parent.parent))

from fei.utils.logging import get_logger, setup_logging
from fei.core.process_manager import initialize_process_manager
from fei.tools.shell_runner import initialize_shell_runner
from fei.tools.process_handlers import (
    view_process_output_handler,
    send_process_input_handler,
    check_process_status_handler,
    terminate_process_handler,
    list_processes_handler
)


def main():
    """Main function to demonstrate the process management system."""
    parser = argparse.ArgumentParser(description="Demonstrate Fei's process management system")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger = get_logger(__name__)
    
    # Initialize process manager and shell runner
    initialize_process_manager()
    initialize_shell_runner()
    
    # Run a simple interactive command (Python REPL)
    print("\n=== Starting Python REPL in background ===")
    result = shell_handler({
        "command": "python -i",
        "background": True
    })
    
    if not result.get("success", False):
        print(f"Error: {result.get('error', 'Unknown error')}")
        return
    
    pid = result.get("pid")
    print(f"Process started with PID: {pid}")
    
    # Wait a moment for the process to start
    time.sleep(1)
    
    # Check process status
    print("\n=== Checking process status ===")
    status = check_process_status_handler({"pid": pid})
    print(f"Process status: {status}")
    
    # Send input to the process
    print("\n=== Sending input to process ===")
    input_result = send_process_input_handler({
        "pid": pid,
        "input_string": "print('Hello from Fei!')\n"
    })
    print(f"Input result: {input_result}")
    
    # Wait a moment for the process to process the input
    time.sleep(1)
    
    # View process output
    print("\n=== Viewing process output ===")
    output = view_process_output_handler({
        "pid": pid,
        "max_lines": 10
    })
    print(f"Process output:\n{output.get('stdout', '')}")
    
    # Send more input
    print("\n=== Sending more input to process ===")
    input_result = send_process_input_handler({
        "pid": pid,
        "input_string": "import sys\nprint(sys.version)\n"
    })
    print(f"Input result: {input_result}")
    
    # Wait a moment for the process to process the input
    time.sleep(1)
    
    # View process output again
    print("\n=== Viewing process output again ===")
    output = view_process_output_handler({
        "pid": pid,
        "max_lines": 10
    })
    print(f"Process output:\n{output.get('stdout', '')}")
    
    # List all processes
    print("\n=== Listing all processes ===")
    processes = list_processes_handler({})
    print(f"Processes: {processes}")
    
    # Terminate the process
    print("\n=== Terminating process ===")
    terminate_result = terminate_process_handler({
        "pid": pid,
        "force": False
    })
    print(f"Terminate result: {terminate_result}")
    
    # Check process status again
    print("\n=== Checking process status again ===")
    status = check_process_status_handler({"pid": pid})
    print(f"Process status: {status}")
    
    print("\nProcess management example completed.")


def shell_handler(args):
    """Simple shell handler for the example."""
    from fei.tools.shell_runner import shell_runner
    
    command = args.get("command")
    timeout = args.get("timeout", 60)
    current_dir = args.get("current_dir")
    background = args.get("background", False)
    
    if not command:
        return {"error": "Command is required"}
    
    return shell_runner.run_command(
        command=command,
        timeout=timeout,
        background=background,
        env=os.environ.copy(),
        do_as_i_say=False
    )


if __name__ == "__main__":
    main()
