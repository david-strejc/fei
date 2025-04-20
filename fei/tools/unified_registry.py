"""
Unified tool registry for Fei code assistant.

This module contains the UnifiedToolRegistry class, which combines
the features of both the standard and enhanced tool registries.
"""

import asyncio
import concurrent.futures
import inspect
import json
import traceback
from typing import Dict, List, Optional, Any, Union, Callable, Set, Tuple

from fei.utils.logging import get_logger
from fei.core.mediator import Mediator
from fei.tools.category import ToolCategoryRegistry
from fei.tools.lifecycle import ToolLifecycleManager

logger = get_logger(__name__)

# Type for tool handler functions
ToolHandlerType = Callable[[Dict[str, Any]], Dict[str, Any]]
AsyncToolHandlerType = Callable[[Dict[str, Any]], Any]
AnyToolHandlerType = Union[ToolHandlerType, AsyncToolHandlerType]


class ToolRegistryError(Exception):
    """Exception raised for tool registry errors."""
    pass


class ToolNotFoundError(ToolRegistryError):
    """Exception raised when a tool is not found."""
    pass


class ToolValidationError(ToolRegistryError):
    """Exception raised when tool arguments are invalid."""
    pass


class ToolExecutionError(ToolRegistryError):
    """Exception raised when tool execution fails."""
    pass


class Tool:
    """Represents a registered tool with metadata and handler."""
    
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler_func: AnyToolHandlerType,
        tags: Optional[List[str]] = None,
        is_async: bool = False,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a tool.
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: Tool input schema (JSON Schema format)
            handler_func: Function to handle tool execution
            tags: Optional tags for categorization
            is_async: Whether the handler is asynchronous
            version: Tool version
            metadata: Additional metadata
        """
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.handler_func = handler_func
        self.tags = tags or []
        self.is_async = is_async
        self.version = version
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tool to dictionary format for LLM.
        
        Returns:
            Tool definition dictionary
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate tool arguments against schema.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation for required fields
        required = self.input_schema.get("required", [])
        for field in required:
            if field not in arguments:
                return False, f"Missing required field: {field}"
        
        # TODO: Add more comprehensive JSON Schema validation
        
        return True, None
    
    def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with arguments.
        
        Args:
            arguments: Tool arguments
            
        Returns:
            Tool execution result
            
        Raises:
            ToolValidationError: If arguments are invalid
            ToolExecutionError: If execution fails
        """
        # Validate arguments
        is_valid, error = self.validate_arguments(arguments)
        if not is_valid:
            raise ToolValidationError(error)
        
        # Execute handler
        try:
            if self.is_async:
                # This should be called from an async context
                raise ToolExecutionError("Async tool cannot be executed synchronously")
            
            return self.handler_func(arguments)
        except Exception as e:
            raise ToolExecutionError(f"Error executing tool: {str(e)}")


class UnifiedToolRegistry:
    """
    Unified registry for tools with enhanced features.
    
    This class combines the features of both the standard and enhanced
    tool registries, providing a single, powerful interface for tool
    registration and execution.
    """
    
    def __init__(self, mediator: Optional[Mediator] = None):
        """
        Initialize unified tool registry.
        
        Args:
            mediator: Optional mediator for component communication
        """
        self.tools: Dict[str, Tool] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        self.category_registry = ToolCategoryRegistry()
        self.lifecycle_manager = ToolLifecycleManager()
        self.mediator = mediator
        self.tool_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
    
    def register_tool(
        self, 
        name: str, 
        description: str, 
        input_schema: Dict[str, Any], 
        handler_func: AnyToolHandlerType,
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        initializer: Optional[Callable] = None,
        cleanup: Optional[Callable] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a new tool.
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: Tool input schema (JSON Schema format)
            handler_func: Function to handle tool execution
            tags: Optional tags for categorization
            category: Optional category name (defaults to "utility")
            initializer: Optional initializer function
            cleanup: Optional cleanup function
            version: Tool version
            metadata: Additional metadata
            
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
            is_async=is_async,
            version=version,
            metadata=metadata
        )
        
        # Add to category
        category = category or "utility"
        if not self.category_registry.get_category(category):
            self.category_registry.add_category(category, f"Tools for {category}")
        
        self.category_registry.add_tool_to_category(category, name, self.tools[name].to_dict())
        
        # Register lifecycle handlers
        if initializer:
            self.lifecycle_manager.register_initializer(name, initializer)
        
        if cleanup:
            self.lifecycle_manager.register_cleanup(name, cleanup)
        
        logger.debug(f"Registered tool: {name} in category: {category}")
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("tool_registered", {
                "name": name,
                "category": category,
                "tags": tags,
                "version": version
            })
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.
        
        Args:
            name: Tool name
            
        Returns:
            True if the tool was unregistered, False if it wasn't registered
        """
        if name not in self.tools:
            return False
        
        # Get tool for notification
        tool = self.tools[name]
        
        # Remove from tools
        del self.tools[name]
        
        # Remove from categories
        for category in self.category_registry.get_categories():
            self.category_registry.remove_tool_from_category(category.name, name)
        
        # Remove lifecycle handlers
        self.lifecycle_manager.unregister_initializer(name)
        self.lifecycle_manager.unregister_cleanup(name)
        
        logger.debug(f"Unregistered tool: {name}")
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("tool_unregistered", {
                "name": name,
                "tags": tool.tags,
                "version": tool.version
            })
        
        return True
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(name)
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all registered tools.
        
        Returns:
            List of tool definitions
        """
        return [tool.to_dict() for tool in self.tools.values()]
    
    def get_tools_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get tools by category.
        
        Args:
            category: Category name
            
        Returns:
            List of matching tool definitions
        """
        category_obj = self.category_registry.get_category(category)
        if not category_obj:
            return []
        
        return [self.tools[name].to_dict() for name in category_obj.list_tools() if name in self.tools]
    
    def get_tools_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Get tools by tag.
        
        Args:
            tag: Tag to filter by
            
        Returns:
            List of matching tool definitions
        """
        return [tool.to_dict() for tool in self.tools.values() if tag in tool.tags]
    
    def get_categories(self) -> List[Dict[str, Any]]:
        """
        Get all categories.
        
        Returns:
            List of category definitions
        """
        return [
            {
                "name": category.name,
                "description": category.description,
                "tool_count": len(category.tools)
            }
            for category in self.category_registry.get_categories()
        ]
    
    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with proper error handling.
        
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
            return await self._handle_mcp_tool(name, arguments)
        
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
        
        # Initialize tool if needed
        if name in self.lifecycle_manager.initializers:
            try:
                await self._initialize_tool(name)
            except Exception as e:
                logger.error(f"Error initializing tool {name}: {str(e)}")
                return {"error": f"Error initializing tool {name}: {str(e)}"}
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("tool_execution_started", {
                "name": name,
                "arguments": arguments
            })
        
        # Add to history
        history_entry = {
            "name": name,
            "arguments": arguments,
            "timestamp": asyncio.get_event_loop().time()
        }
        self.tool_history.append(history_entry)
        
        # Trim history if needed
        if len(self.tool_history) > self.max_history_size:
            self.tool_history = self.tool_history[-self.max_history_size:]
        
        # Execute tool
        try:
            if tool.is_async:
                # For async handlers, call directly
                result = await tool.handler_func(arguments)
            else:
                # For sync handlers, run in thread pool
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    lambda: tool.handler_func(arguments)
                )
            
            # Update history entry with result
            history_entry["result"] = result
            history_entry["success"] = True
            
            # Notify mediator if available
            if self.mediator:
                self.mediator.notify("tool_execution_completed", {
                    "name": name,
                    "arguments": arguments,
                    "result": result
                })
            
            return result
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error executing tool {name}: {str(e)}")
            logger.debug(error_details)
            
            # Update history entry with error
            history_entry["error"] = str(e)
            history_entry["success"] = False
            
            # Notify mediator if available
            if self.mediator:
                self.mediator.notify("tool_execution_failed", {
                    "name": name,
                    "arguments": arguments,
                    "error": str(e),
                    "details": error_details
                })
            
            return {
                "error": f"Error executing tool {name}: {str(e)}",
                "details": error_details
            }
        finally:
            # Only clean up if the tool has a cleanup handler
            # and it's not meant to persist (like servers)
            if name in self.lifecycle_manager.cleanups and not self._is_persistent_tool(name):
                await self._cleanup_tool(name)
    
    async def _initialize_tool(self, name: str) -> None:
        """
        Initialize a tool.
        
        Args:
            name: Tool name
            
        Raises:
            ToolRegistryError: If the tool is not found or initialization fails
        """
        initializer = self.lifecycle_manager.initializers.get(name)
        if not initializer:
            return
        
        try:
            if asyncio.iscoroutinefunction(initializer):
                await initializer()
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, initializer)
        except Exception as e:
            raise ToolRegistryError(f"Error initializing tool {name}: {str(e)}")
    
    async def _cleanup_tool(self, name: str) -> None:
        """
        Clean up a tool.
        
        Args:
            name: Tool name
            
        Raises:
            ToolRegistryError: If the tool is not found or cleanup fails
        """
        cleanup = self.lifecycle_manager.cleanups.get(name)
        if not cleanup:
            return
        
        try:
            if asyncio.iscoroutinefunction(cleanup):
                await cleanup()
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, cleanup)
        except Exception as e:
            logger.error(f"Error cleaning up tool {name}: {str(e)}")
    
    def _is_persistent_tool(self, name: str) -> bool:
        """
        Check if a tool is persistent.
        
        Args:
            name: Tool name
            
        Returns:
            True if the tool is persistent, False otherwise
        """
        # Check if tool has persistent metadata
        tool = self.tools.get(name)
        if tool and tool.metadata.get("persistent", False):
            return True
        
        # Check if tool name indicates persistence
        persistent_prefixes = ["server_", "daemon_", "service_"]
        return any(name.startswith(prefix) for prefix in persistent_prefixes)
    
    async def _handle_mcp_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tool execution.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        try:
            # Import here to avoid circular imports
            from fei.core.mcp import MCPManager
            
            # Get MCP manager from mediator
            mcp_manager = None
            if self.mediator:
                mcp_manager = self.mediator.get_component("mcp_manager")
            
            if not mcp_manager:
                return {"error": "MCP manager not available"}
            
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
                
                # Call Brave Search service
                return await mcp_manager.call_service(
                    "brave_search",
                    "search",
                    {"query": query, "count": count, "offset": offset}
                )
            
            # Handle other MCP tools
            elif name.startswith("mcp_"):
                # Extract service and method from tool name
                parts = name.split("_", 2)
                if len(parts) < 3:
                    return {"error": f"Invalid MCP tool name: {name}"}
                
                service = parts[1]
                method = parts[2]
                
                # Call service method
                return await mcp_manager.call_service(service, method, arguments)
            
            return {"error": f"Unknown MCP tool: {name}"}
            
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
        Get a list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def list_tags(self) -> Set[str]:
        """
        Get a list of all unique tags across tools.
        
        Returns:
            Set of unique tags
        """
        tags = set()
        for tool in self.tools.values():
            tags.update(tool.tags)
        return tags
    
    def get_tool_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent tool execution history.
        
        Args:
            limit: Maximum number of entries to return
            
        Returns:
            List of history entries
        """
        return self.tool_history[-limit:]
    
    def register_class_methods(
        self, 
        cls: Any, 
        prefix: str = "", 
        tags: Optional[List[str]] = None,
        category: Optional[str] = None,
        version: str = "1.0.0",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register all methods of a class as tools.
        
        Args:
            cls: Class instance
            prefix: Prefix for tool names
            tags: Tags to apply to all registered tools
            category: Category to place tools in
            version: Tool version
            metadata: Additional metadata
            
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
            registry.register_class_methods(math_tools, prefix="math_", tags=["math"], category="math")
            ```
            
            This will register tools named "math_add" and "math_subtract" in the "math" category.
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
                
                # Determine parameter type
                param_type = param.annotation
                if param_type is inspect.Parameter.empty:
                    schema_type = "string"  # Default to string
                elif param_type is int:
                    schema_type = "integer"
                elif param_type is float:
                    schema_type = "number"
                elif param_type is bool:
                    schema_type = "boolean"
                elif param_type is list or param_type is List:
                    schema_type = "array"
                elif param_type is dict or param_type is Dict:
                    schema_type = "object"
                else:
                    schema_type = "string"  # Default to string for other types
                
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
                tags=tags,
                category=category,
                version=version,
                metadata=metadata
            )
