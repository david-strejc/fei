"""
Tool lifecycle management for Fei.
"""

from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)


class ToolLifecycleManager:
    """
    Manages the lifecycle of tools.
    
    This includes initialization, cleanup, and state management.
    """
    
    def __init__(self):
        """Initialize the tool lifecycle manager."""
        self.initializers: Dict[str, Callable] = {}
        self.cleanups: Dict[str, Callable] = {}
        self.states: Dict[str, Dict[str, Any]] = {}
    
    def register_initializer(self, tool_name: str, initializer: Callable) -> None:
        """
        Register an initializer for a tool.
        
        Args:
            tool_name: Name of the tool
            initializer: Initializer function
        """
        self.initializers[tool_name] = initializer
    
    def register_cleanup(self, tool_name: str, cleanup: Callable) -> None:
        """
        Register a cleanup function for a tool.
        
        Args:
            tool_name: Name of the tool
            cleanup: Cleanup function
        """
        self.cleanups[tool_name] = cleanup
    
    def initialize_tool(self, tool_name: str, **kwargs) -> bool:
        """
        Initialize a tool.
        
        Args:
            tool_name: Name of the tool
            **kwargs: Additional arguments for the initializer
            
        Returns:
            True if initialization was successful, False otherwise
        """
        if tool_name not in self.initializers:
            # No initializer registered, consider it successful
            return True
        
        try:
            initializer = self.initializers[tool_name]
            state = initializer(**kwargs)
            if state is not None:
                self.states[tool_name] = state
            return True
        except Exception as e:
            logger.error(f"Error initializing tool '{tool_name}': {e}")
            return False
    
    def cleanup_tool(self, tool_name: str) -> bool:
        """
        Clean up a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        if tool_name not in self.cleanups:
            # No cleanup registered, consider it successful
            return True
        
        try:
            cleanup = self.cleanups[tool_name]
            state = self.states.get(tool_name, {})
            cleanup(state)
            if tool_name in self.states:
                del self.states[tool_name]
            return True
        except Exception as e:
            logger.error(f"Error cleaning up tool '{tool_name}': {e}")
            return False
    
    def get_tool_state(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the state of a tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool state or None if not found
        """
        return self.states.get(tool_name)
    
    def set_tool_state(self, tool_name: str, state: Dict[str, Any]) -> None:
        """
        Set the state of a tool.
        
        Args:
            tool_name: Name of the tool
            state: Tool state
        """
        self.states[tool_name] = state
    
    def initialize_all_tools(self, tool_names: List[str], **kwargs) -> Dict[str, bool]:
        """
        Initialize multiple tools.
        
        Args:
            tool_names: List of tool names
            **kwargs: Additional arguments for initializers
            
        Returns:
            Dictionary of tool names and initialization results
        """
        results = {}
        for tool_name in tool_names:
            results[tool_name] = self.initialize_tool(tool_name, **kwargs)
        return results
    
    def cleanup_all_tools(self, tool_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Clean up multiple tools.
        
        Args:
            tool_names: Optional list of tool names, defaults to all tools with state
            
        Returns:
            Dictionary of tool names and cleanup results
        """
        if tool_names is None:
            tool_names = list(self.states.keys())
        
        results = {}
        for tool_name in tool_names:
            results[tool_name] = self.cleanup_tool(tool_name)
        return results
