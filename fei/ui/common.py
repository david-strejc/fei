"""
Common UI utilities for Fei.

This module provides common utilities for both CLI and Textual UIs.
"""

import os
from typing import Dict, List, Optional, Any, Union

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.core.process_manager import initialize_process_manager
from fei.tools.shell_runner import initialize_shell_runner
from fei.tools.register_tools import register_process_tools

logger = get_logger(__name__)


def initialize_tools(do_as_i_say: bool = False, temp_dir: Optional[str] = None) -> None:
    """
    Initialize tools with command-line arguments.
    
    Args:
        do_as_i_say: Whether to bypass all command checks (DANGEROUS!)
        temp_dir: Directory for temporary files
    """
    # Initialize process manager
    initialize_process_manager(temp_dir=temp_dir)
    
    # Initialize shell runner
    initialize_shell_runner(do_as_i_say=do_as_i_say)
    
    logger.info("Tools initialized")


def setup_tool_registry(registry: Any) -> None:
    """
    Set up tool registry with all available tools.
    
    Args:
        registry: Tool registry instance
    """
    # Register code tools
    from fei.tools.code import create_code_tools
    create_code_tools(registry)
    
    # Register memory tools
    from fei.tools.memory_tools import create_memory_tools
    create_memory_tools(registry)
    
    # Register process tools
    register_process_tools(registry)
    
    logger.info("Tool registry set up")
