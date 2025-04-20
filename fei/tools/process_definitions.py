"""
Process management tool definitions for Fei.

This module provides definitions for process management tools.
"""

from typing import Dict, List, Any

# View Process Output tool definition
VIEW_PROCESS_OUTPUT_TOOL = {
    "name": "ViewProcessOutput",
    "description": "View the output of a background process. Use this to see what a process is printing to stdout/stderr.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Process ID (PID) of the process to view"
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return (default: 50)"
            },
            "include_stderr": {
                "type": "boolean",
                "description": "Whether to include stderr output (default: true)"
            }
        },
        "required": ["pid"]
    }
}

# Send Process Input tool definition
SEND_PROCESS_INPUT_TOOL = {
    "name": "SendProcessInput",
    "description": "Send input to a running interactive process. Use this to interact with processes that require user input.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Process ID (PID) of the process to send input to"
            },
            "input_string": {
                "type": "string",
                "description": "Input string to send to the process"
            }
        },
        "required": ["pid", "input_string"]
    }
}

# Check Process Status tool definition
CHECK_PROCESS_STATUS_TOOL = {
    "name": "CheckProcessStatus",
    "description": "Check the status of a process. Use this to see if a process is still running.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Process ID (PID) of the process to check"
            }
        },
        "required": ["pid"]
    }
}

# Terminate Process tool definition
TERMINATE_PROCESS_TOOL = {
    "name": "TerminateProcess",
    "description": "Terminate a running process. Use this to stop a process that is no longer needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pid": {
                "type": "integer",
                "description": "Process ID (PID) of the process to terminate"
            },
            "force": {
                "type": "boolean",
                "description": "Whether to force termination with SIGKILL (default: false)"
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds before sending SIGKILL if force=true (default: 3.0)"
            }
        },
        "required": ["pid"]
    }
}

# List Processes tool definition
LIST_PROCESSES_TOOL = {
    "name": "ListProcesses",
    "description": "List all processes managed by Fei. Use this to see what processes are currently running.",
    "input_schema": {
        "type": "object",
        "properties": {}
    }
}

# List of all process tool definitions
PROCESS_TOOL_DEFINITIONS = [
    VIEW_PROCESS_OUTPUT_TOOL,
    SEND_PROCESS_INPUT_TOOL,
    CHECK_PROCESS_STATUS_TOOL,
    TERMINATE_PROCESS_TOOL,
    LIST_PROCESSES_TOOL
]
