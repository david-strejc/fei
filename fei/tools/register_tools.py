"""
Tool registration for Fei.

This module provides functions for registering tools with the tool registry.
"""

from typing import Optional

from fei.tools.unified_registry import UnifiedToolRegistry
from fei.tools.process_handlers import (
    view_process_output_handler,
    send_process_input_handler,
    check_process_status_handler,
    terminate_process_handler,
    list_processes_handler
)
from fei.tools.process_definitions import (
    VIEW_PROCESS_OUTPUT_TOOL,
    SEND_PROCESS_INPUT_TOOL,
    CHECK_PROCESS_STATUS_TOOL,
    TERMINATE_PROCESS_TOOL,
    LIST_PROCESSES_TOOL
)


def register_process_tools(registry: Optional[UnifiedToolRegistry] = None) -> None:
    """
    Register process management tools with the registry.
    
    Args:
        registry: Tool registry instance
    """
    if registry is None:
        return
    
    # Register process management tools
    registry.register_tool(
        name=VIEW_PROCESS_OUTPUT_TOOL["name"],
        description=VIEW_PROCESS_OUTPUT_TOOL["description"],
        input_schema=VIEW_PROCESS_OUTPUT_TOOL["input_schema"],
        handler_func=view_process_output_handler,
        tags=["process", "utility"],
        category="process"
    )
    
    registry.register_tool(
        name=SEND_PROCESS_INPUT_TOOL["name"],
        description=SEND_PROCESS_INPUT_TOOL["description"],
        input_schema=SEND_PROCESS_INPUT_TOOL["input_schema"],
        handler_func=send_process_input_handler,
        tags=["process", "utility"],
        category="process"
    )
    
    registry.register_tool(
        name=CHECK_PROCESS_STATUS_TOOL["name"],
        description=CHECK_PROCESS_STATUS_TOOL["description"],
        input_schema=CHECK_PROCESS_STATUS_TOOL["input_schema"],
        handler_func=check_process_status_handler,
        tags=["process", "utility"],
        category="process"
    )
    
    registry.register_tool(
        name=TERMINATE_PROCESS_TOOL["name"],
        description=TERMINATE_PROCESS_TOOL["description"],
        input_schema=TERMINATE_PROCESS_TOOL["input_schema"],
        handler_func=terminate_process_handler,
        tags=["process", "utility"],
        category="process"
    )
    
    registry.register_tool(
        name=LIST_PROCESSES_TOOL["name"],
        description=LIST_PROCESSES_TOOL["description"],
        input_schema=LIST_PROCESSES_TOOL["input_schema"],
        handler_func=list_processes_handler,
        tags=["process", "utility"],
        category="process"
    )
