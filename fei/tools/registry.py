#!/usr/bin/env python3
"""
Tool registry for Fei code assistant

This module provides a registry for Claude Universal Assistant tools
with proper error handling and extensibility.
"""

import os
import sys
import traceback
import asyncio
import concurrent.futures
from typing import Dict, List, Any, Optional, Callable, Union, Set, TypeVar, Generic, Tuple
import json
import inspect
import functools
import logging

from fei.utils.logging import get_logger

logger = get_logger(__name__)

# Type for tool handler functions
ToolHandlerType = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncToolHandlerType = Callable[[Dict[str, Any]], Any]
AnyToolHandlerType = Union[ToolHandlerType, AsyncToolHandlerType]

class ToolError(Exception):
    """Base class for tool-related exceptions"""
    pass

class ToolNotFoundError(ToolError):
    """Exception raised when a tool is not found"""
    pass

class ToolExecutionError(ToolError):
    """Exception raised when a tool execution fails"""
    pass

class ToolValidationError(ToolError):
    """Exception raised when tool arguments are invalid"""
    pass

class ToolRegistryError(ToolError):
    """Exception raised for tool registry errors"""
    pass

class Tool:
    """Represents a registered tool with metadata and handler"""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler_func: AnyToolHandlerType,
        tags: Optional[List[str]] = None,
        is_async: bool = False
    ):
        """
        Initialize a tool
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: Tool input schema (JSON Schema format)
            handler_func: Function to handle tool execution
            tags: Optional tags for categorization
            is_async: Whether the handler is asynchronous
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler_func = handler_func
        self.tags = tags or []
        self.is_async = is_async
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tool to dictionary format
        
        Returns:
            Dictionary representation of tool
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate arguments against input schema
        
        Args:
            arguments: Tool arguments to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        required = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})
        
        # Check required properties
        for prop in required:
            if prop not in arguments:
                return False, f"Missing required property: {prop}"
        
        # Validate property types (basic validation)
        for prop, value in arguments.items():
            if prop in properties:
                prop_type = properties[prop].get("type")
                
                # Skip if no type information
                if not prop_type:
                    continue
                
                # Handle array type
                if prop_type == "array" and not isinstance(value, list):
                    return False, f"Property {prop} must be an array"
                
                # Handle object type
                if prop_type == "object" and not isinstance(value, dict):
                    return False, f"Property {prop} must be an object"
                
                # Handle number types
                if prop_type == "number" and not isinstance(value, (int, float)):
                    return False, f"Property {prop} must be a number"
                
                # Handle integer type
                if prop_type == "integer" and not isinstance(value, int):
                    try:
                        # Try to convert strings to integers
                        if isinstance(value, str):
                            int(value)
                    except ValueError:
                        return False, f"Property {prop} must be an integer"
                
                # Handle string type
                if prop_type == "string" and not isinstance(value, str):
                    return False, f"Property {prop} must be a string"
                
                # Handle boolean type
                if prop_type == "boolean" and not isinstance(value, bool):
                    # Try to convert string booleans
                    if isinstance(value, str):
                        if value.lower() not in ["true", "false", "1", "0", "yes", "no"]:
                            return False, f"Property {prop} must be a boolean"
                    else:
                        return False, f"Property {prop} must be a boolean"
        
        return True, None


class ToolRegistry:
    """Registry for Claude Universal Assistant tools with enhanced error handling"""
    
    def __init__(self):
        """Initialize tool registry"""
        self.tools: Dict[str, Tool] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    
    def register_tool(
        self, 
        name: str, 
        description: str, 
        input_schema: Dict[str, Any], 
        handler_func: AnyToolHandlerType,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Register a new tool
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: Tool input schema (JSON Schema format)
            handler_func: Function to handle tool execution
            tags: Optional tags for categorization
            
        Raises:
            ToolRegistryError: If a tool with the same name is already registered
        """
        if name in self.tools:
            raise ToolRegistryError(f"Tool already registered: {name}")
        
        # Detect if handler is async
        is_async = asyncio.iscoroutinefunction(handler_func)
        
        # Register tool
        self.tools[name] = Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler_func=handler_func,
            tags=tags,
            is_async=is_async
        )
        
        logger.debug(f"Registered tool: {name}")
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all registered tools
        
        Returns:
            List of tool definitions
        """
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get tools by tag
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of matching tool definitions
        """
        return [tool.to_dict() for tool in self.tools.values() if tag in tool.tags]
    
    def get_tool(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific tool definition by name
        
        Args:
            name: Tool name
            
        Returns:
            Tool definition or None if not found
        """
        tool = self.tools.get(name)
        return tool.to_dict() if tool else None
    
    def get_handler(self, name: str) -> Optional[AnyToolHandlerType]:
        """
        Get a specific tool handler by name
        
        Args:
            name: Tool name
            
        Returns:
            Tool handler function or None if not found
        """
        tool = self.tools.get(name)
        return tool.handler_func if tool else None
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with proper error handling
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ToolNotFoundError: If the tool is not found
            ToolValidationError: If arguments are invalid
            ToolExecutionError: If tool execution fails
        """
        # Special handling for MCP tools
        if name == "brave_web_search" or name.startswith("mcp_"):
            return self._handle_mcp_tool(name, arguments)
        
        # Get tool
        tool = self.tools.get(name)
        if not tool:
            logger.error(f"Tool not found: {name}")
            return {"error": f"Tool not found: {name}"}
        
        # Validate arguments
        is_valid, error = tool.validate_arguments(arguments)
        if not is_valid:
            logger.error(f"Invalid arguments for {name}: {error}")
            return {"error": f"Invalid arguments for {name}: {error}"}
        
        # Execute tool
        try:
            if tool.is_async:
                # For async handlers, we need to run in an event loop
                return self._execute_async_tool(tool, arguments)
            else:
                # For sync handlers, just call directly
                return tool.handler_func(arguments)
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error executing tool {name}: {str(e)}")
            logger.debug(error_details)
            return {
                "error": f"Error executing tool {name}: {str(e)}",
                "details": error_details
            }
    
    def _execute_async_tool(self, tool: Tool, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an async tool in the current event loop or a new one
        
        Args:
            tool: Tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_running():
                # If no loop is running, use run_until_complete
                return loop.run_until_complete(tool.handler_func(arguments))
            else:
                # If we're already in a loop, ensure we don't create nested loops
                # Instead, use run_in_executor to run a new event loop in a separate thread
                def run_async_in_thread():
                    # Create a new event loop for this thread
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)
                    try:
                        return thread_loop.run_until_complete(tool.handler_func(arguments))
                    finally:
                        thread_loop.close()
                
                # Run in a separate thread
                future = self.executor.submit(run_async_in_thread)
                return future.result()
                
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error executing async tool {tool.name}: {str(e)}")
            logger.debug(error_details)
            return {
                "error": f"Error executing tool {tool.name}: {str(e)}",
                "details": error_details
            }
    
    def _handle_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tools with proper error handling
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            # Import here to avoid circular imports
            from fei.core.mcp import MCPManager, MCPConnectionError, MCPExecutionError
            
            # Create MCP manager on demand
            mcp_manager = MCPManager()
            
            # Handle specific MCP tools
            if name == "brave_web_search":
                # Handle Brave Search
                query = arguments.get("query", "")
                count = arguments.get("count", 10)
                offset = arguments.get("offset", 0)
                
                if not query:
                    return {"error": "Search query is required"}
                
                # Validate count and offset
                try:
                    count = int(count)
                    offset = int(offset)
                except (ValueError, TypeError):
                    return {"error": "Count and offset must be integers"}
                
                # Limit count
                count = min(max(1, count), 20)
                offset = max(0, offset)
                
                # Use asyncio to run the search
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        # If no loop is running, use run_until_complete
                        return loop.run_until_complete(
                            mcp_manager.brave_search.brave_web_search(query, count, offset)
                        )
                    else:
                        # If we're already in a loop, ensure we don't create nested loops
                        # Instead, use run_in_executor to run a new event loop in a separate thread
                        def run_async_in_thread():
                            # Create a new event loop for this thread
                            thread_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(thread_loop)
                            try:
                                return thread_loop.run_until_complete(
                                    mcp_manager.brave_search.brave_web_search(query, count, offset)
                                )
                            finally:
                                thread_loop.close()
                        
                        # Run in a separate thread
                        future = self.executor.submit(run_async_in_thread)
                        return future.result()
                except Exception as e:
                    logger.error(f"Error executing Brave Search: {str(e)}")
                    return {"error": f"Error executing Brave Search: {str(e)}"}
            
            # For other MCP tools (future expansion)
            if name.startswith("mcp_"):
                service_name = name[4:].split("_")[0]  # Extract service name after "mcp_"
                method_name = "_".join(name[4:].split("_")[1:])  # Extract method name
                
                # Check if the service exists
                service = getattr(mcp_manager, service_name, None)
                if not service:
                    return {"error": f"MCP service not found: {service_name}"}
                
                # Check if the method exists
                method = getattr(service, method_name, None)
                if not method:
                    return {"error": f"MCP method not found: {method_name} in {service_name}"}
                
                # Call the method
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_running():
                        # If no loop is running, use run_until_complete
                        return loop.run_until_complete(method(**arguments))
                    else:
                        # If we're already in a loop, ensure we don't create nested loops
                        # Instead, use run_in_executor to run a new event loop in a separate thread
                        def run_async_in_thread():
                            # Create a new event loop for this thread
                            thread_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(thread_loop)
                            try:
                                return thread_loop.run_until_complete(method(**arguments))
                            finally:
                                thread_loop.close()
                        
                        # Run in a separate thread
                        future = self.executor.submit(run_async_in_thread)
                        return future.result()
                except MCPConnectionError as e:
                    logger.error(f"MCP connection error: {e}")
                    return {"error": f"MCP connection error: {str(e)}"}
                except MCPExecutionError as e:
                    logger.error(f"MCP execution error: {e}")
                    return {"error": f"MCP execution error: {str(e)}"}
                except Exception as e:
                    logger.error(f"Error executing MCP tool {name}: {str(e)}")
                    return {"error": f"Error executing MCP tool {name}: {str(e)}"}
            
            # Unsupported MCP tool
            return {"error": f"Unsupported MCP tool: {name}"}
            
        except ImportError as e:
            logger.error(f"Error importing MCP: {e}")
            return {"error": f"MCP not available: {str(e)}"}
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error handling MCP tool {name}: {str(e)}")
            logger.debug(error_details)
            return {
                "error": f"Error handling MCP tool {name}: {str(e)}",
                "details": error_details
            }
    
    def list_tool_names(self) -> List[str]:
        """
        Get a list of all registered tool names
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def list_tags(self) -> Set[str]:
        """
        Get a list of all unique tags across tools
        
        Returns:
            Set of unique tags
        """
        tags = set()
        for tool in self.tools.values():
            tags.update(tool.tags)
        return tags
        
    def invoke_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke a tool (alias for execute_tool)
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        return self.execute_tool(name, arguments)

    def register_class_methods(self, cls: Any, prefix: str = "", tags: Optional[List[str]] = None) -> None:
        """
        Register all methods of a class as tools
        
        Args:
            cls: Class instance
            prefix: Prefix for tool names
            tags: Tags to apply to all registered tools
            
        Example:
            ```python
            class MathTools:
                def add(self, a: int, b: int) -> int:
                    \"\"\"Add two numbers\"\"\"
                    return a + b
                    
                def subtract(self, a: int, b: int) -> int:
                    \"\"\"Subtract b from a\"\"\"
                    return a - b
                    
            math_tools = MathTools()
            registry.register_class_methods(math_tools, prefix="math_", tags=["math"])
            ```
            
            This will register tools named "math_add" and "math_subtract".
        """
        # Get all methods of the class
        methods = [
            (name, method) for name, method in inspect.getmembers(cls, predicate=inspect.ismethod)
            if not name.startswith("_")
        ]
        
        for name, method in methods:
            # Get method signature and docstring
            sig = inspect.signature(method)
            doc = inspect.getdoc(method) or f"Execute {name} method"
            
            # Create input schema from signature
            properties = {}
            required = []
            
            for param_name, param in sig.parameters.items():
                # Skip 'self' parameter
                if param_name == "self":
                    continue
                
                # Get parameter type annotation
                param_type = param.annotation
                
                # Default to "string" if no annotation or annotation is not a supported type
                schema_type = "string"
                
                # Map Python types to JSON Schema types
                if param_type is int:
                    schema_type = "integer"
                elif param_type is float:
                    schema_type = "number"
                elif param_type is bool:
                    schema_type = "boolean"
                elif param_type is str:
                    schema_type = "string"
                elif param_type is list or getattr(param_type, "__origin__", None) is list:
                    schema_type = "array"
                elif param_type is dict or getattr(param_type, "__origin__", None) is dict:
                    schema_type = "object"
                
                # Add parameter to properties
                properties[param_name] = {
                    "type": schema_type,
                    "description": f"Parameter {param_name}"
                }
                
                # Add to required if no default value
                if param.default is inspect.Parameter.empty:
                    required.append(param_name)
            
            # Create input schema
            input_schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }
            
            # Create handler function
            def create_handler(method):
                def handler(args):
                    try:
                        return method(**args)
                    except Exception as e:
                        return {"error": str(e)}
                return handler
            
            # Register tool
            tool_name = f"{prefix}{name}" if prefix else name
            self.register_tool(
                name=tool_name,
                description=doc,
                input_schema=input_schema,
                handler_func=create_handler(method),
                tags=tags
            )
    
    def __del__(self):
        """Cleanup when registry is deleted"""
        # Shut down executor to clean up threads
        self.executor.shutdown(wait=False)