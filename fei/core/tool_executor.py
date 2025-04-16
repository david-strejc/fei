"""
Tool executor for Fei code assistant.

This module contains the ToolExecutor class, which handles
tool registration, formatting, and execution.
"""

import asyncio
import json
import traceback
from typing import Dict, List, Optional, Any, Union

from fei.utils.logging import get_logger
from fei.tools.registry import ToolRegistry
from fei.core.mcp import MCPManager
from fei.core.mediator import Mediator

logger = get_logger(__name__)


class ToolExecutor:
    """
    Handles tool registration, formatting, and execution.
    
    This class is responsible for:
    1. Managing tool registration
    2. Formatting tools for different providers
    3. Executing tools safely
    """
    
    def __init__(
        self,
        provider: str,
        mcp_manager: Optional[MCPManager] = None,
        mediator: Optional[Mediator] = None,
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        Initialize the tool executor.
        
        Args:
            provider: Provider name for proper formatting
            mcp_manager: MCP manager instance
            mediator: Mediator for component communication
            tool_registry: Tool registry instance
        """
        self.provider = provider
        self.mcp_manager = mcp_manager
        self.mediator = mediator
        self.tool_registry = tool_registry
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools, using provider-specific format.
        
        Returns:
            List of tool definitions in provider-specific format
        """
        if not self.tool_registry:
            return []
        
        # For Anthropic, use the format that works with LiteLLM
        if self.provider == "anthropic":
            from fei.tools.definitions import ANTHROPIC_TOOL_DEFINITIONS
            tools = []
            
            # Include tools from registry in format that works with LiteLLM
            for tool in ANTHROPIC_TOOL_DEFINITIONS:
                # Format according to LiteLLM's expected format for Anthropic
                tool_copy = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                tools.append(tool_copy)
                
            return tools
        else:
            # Standard format for other providers
            tools = self.tool_registry.get_tools()
            
            # Ensure all tools have a type for OpenAI compatibility
            for tool in tools:
                if "type" not in tool:
                    tool["type"] = "function"
                    
            return tools
    
    async def execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool safely.
        
        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            
        Returns:
            Tool result or error information
        """
        if not self.tool_registry:
            return {"error": "No tool registry available"}
        
        # Notify mediator about tool execution (if available)
        if self.mediator:
            self.mediator.notify("tool_execution_started", {
                "tool_name": tool_name,
                "tool_args": tool_args
            })
        
        try:
            # Execute the tool
            result = await self._execute_tool_internal(tool_name, tool_args)
            
            # Notify mediator about tool execution completion (if available)
            if self.mediator:
                self.mediator.notify("tool_execution_completed", {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result
                })
            
            return result
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            logger.debug(error_details)
            
            # Notify mediator about tool execution failure (if available)
            if self.mediator:
                self.mediator.notify("tool_execution_failed", {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "error": str(e),
                    "details": error_details
                })
            
            return {
                "error": f"Error executing tool {tool_name}: {str(e)}",
                "details": error_details
            }
    
    async def _execute_tool_internal(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool internally.
        
        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            
        Returns:
            Tool result
        """
        # Special handling for MCP tools
        if tool_name == "brave_web_search" or tool_name.startswith("mcp_"):
            return await self._handle_mcp_tool(tool_name, tool_args)
        
        # Execute tool using registry
        return await self._execute_registry_tool(tool_name, tool_args)
    
    async def _execute_registry_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool using the registry.
        
        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            
        Returns:
            Tool result
        """
        # Use the tool registry to execute the tool
        if asyncio.iscoroutinefunction(self.tool_registry.execute_tool):
            return await self.tool_registry.execute_tool(tool_name, tool_args)
        else:
            # Run synchronous execution in a thread pool
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.tool_registry.execute_tool(tool_name, tool_args)
            )
    
    async def _handle_mcp_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP tool execution.
        
        Args:
            tool_name: Tool name
            tool_args: Tool arguments
            
        Returns:
            Tool result
        """
        if not self.mcp_manager:
            return {"error": "MCP manager not available"}
        
        try:
            # Handle specific MCP tools
            if tool_name == "brave_web_search":
                # Handle Brave Search
                query = tool_args.get("query", "")
                count = tool_args.get("count", 10)
                offset = tool_args.get("offset", 0)
                
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
                return await self.mcp_manager.call_service(
                    "brave_search",
                    "search",
                    {"query": query, "count": count, "offset": offset}
                )
            
            # Handle other MCP tools
            elif tool_name.startswith("mcp_"):
                # Extract service and method from tool name
                parts = tool_name.split("_", 2)
                if len(parts) < 3:
                    return {"error": f"Invalid MCP tool name: {tool_name}"}
                
                service = parts[1]
                method = parts[2]
                
                # Call service method
                return await self.mcp_manager.call_service(service, method, tool_args)
            
            return {"error": f"Unknown MCP tool: {tool_name}"}
            
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error handling MCP tool {tool_name}: {str(e)}")
            logger.debug(error_details)
            return {
                "error": f"Error handling MCP tool {tool_name}: {str(e)}",
                "details": error_details
            }
