#!/usr/bin/env python3
"""
Memory management tools for FEI

This module provides tools for interacting with Memdir and Memorychain
to search, create, and manage memories.
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Union

import os # Import os for environment variable access
from fei.tools.registry import ToolRegistry
from fei.tools.memdir_connector import MemdirConnector
from fei.tools.memorychain_connector import MemorychainConnector
from fei.utils.logging import get_logger
from fei.utils.config import get_config # Import get_config

logger = get_logger(__name__)

# Input schemas for memory tools
MEMORY_TOOL_SCHEMAS = {
    "memdir_server_start": {
        "type": "object",
        "properties": {}
    },
    "memdir_server_stop": {
        "type": "object",
        "properties": {}
    },
    "memdir_server_status": {
        "type": "object",
        "properties": {}
    },
    "memory_search": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "Search query string with field operators and shortcuts"
            },
            "folder": {
                "type": "string", 
                "description": "Folder to search in (default: all folders)"
            },
            "status": {
                "type": "string", 
                "description": "Status folder to search in (default: all statuses)"
            },
            "limit": {
                "type": "integer", 
                "description": "Maximum number of results"
            },
            "with_content": {
                "type": "boolean", 
                "description": "Whether to include memory content in results"
            }
        },
        "required": ["query"]
    },
    "memory_create": {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string", 
                "description": "Memory subject/title"
            },
            "content": {
                "type": "string", 
                "description": "Memory content"
            },
            "tags": {
                "type": "string", 
                "description": "Comma-separated tags"
            },
            "priority": {
                "type": "string", 
                "description": "Priority (high, medium, low)"
            },
            "folder": {
                "type": "string", 
                "description": "Target folder"
            }
        },
        "required": ["subject", "content"]
    },
    "memory_view": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string", 
                "description": "Memory ID or filename"
            },
            "folder": {
                "type": "string", 
                "description": "Folder to search in (default: all folders)"
            }
        },
        "required": ["memory_id"]
    },
    "memory_list": {
        "type": "object",
        "properties": {
            "folder": {
                "type": "string", 
                "description": "Folder name (default: root folder)"
            },
            "status": {
                "type": "string", 
                "description": "Status folder (cur, new, tmp)"
            },
            "limit": {
                "type": "integer", 
                "description": "Maximum number of results to show"
            }
        }
    },
    "memory_delete": {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string", 
                "description": "Memory ID or filename"
            },
            "folder": {
                "type": "string", 
                "description": "Memory folder"
            }
        },
        "required": ["memory_id"]
    },
    "memory_search_by_tag": {
        "type": "object",
        "properties": {
            "tag": {
                "type": "string", 
                "description": "Tag to search for"
            }
        },
        "required": ["tag"]
    }
}

# Handlers for memory tools
def memory_search_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for memories in Memdir
    
    Args:
        args: Tool arguments
        
    Returns:
        Search results
    """
    try:
        # Initialize connector - let its __init__ handle API key resolution
        connector = MemdirConnector(auto_start=True)

        # First check if the server is available
        if not connector.start_server_command():
            return {
                "error": "Cannot connect to Memdir server. Server start failed.",
                "count": 0,
                "results": []
            }
            
        # Extract arguments
        query = args.get("query", "")
        folder = args.get("folder")
        status = args.get("status")
        limit = args.get("limit")
        with_content = args.get("with_content", False)
        
        # Perform search
        results = connector.search(
            query=query,
            folder=folder,
            status=status,
            limit=limit,
            with_content=with_content
        )
        
        # Format results for display
        formatted_results = []
        for memory in results.get("results", []):
            headers = memory.get("headers", {})
            metadata = memory.get("metadata", {})
            
            memory_info = {
                "id": metadata.get("unique_id", ""),
                "subject": headers.get("Subject", "No subject"),
                "date": metadata.get("date", ""),
                "tags": headers.get("Tags", ""),
                "priority": headers.get("Priority", ""),
                "status": headers.get("Status", ""),
                "flags": "".join(metadata.get("flags", []))
            }
            
            if with_content:
                memory_info["content"] = memory.get("content", "")
                
            formatted_results.append(memory_info)
        
        return {
            "count": results.get("count", 0),
            "results": formatted_results
        }
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return {
            "error": f"Error searching memories: {e}",
            "count": 0,
            "results": []
        }

def memory_create_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new memory
    
    Args:
        args: Tool arguments
        
    Returns:
        Result of memory creation
    """
    try:
        # Read API key ONLY from environment variable
        api_key = os.environ.get("MEMDIR_API_KEY")
        connector = MemdirConnector(api_key=api_key, auto_start=True)

        # First check if the server is available and start it if needed
        start_result = connector.start_server_command()
        if start_result["status"] not in ["started", "already_running"]:
            return {
                "error": "Cannot connect to Memdir server. Server start failed.",
                "success": False
            }
            
        # Extract arguments
        subject = args.get("subject", "")
        content = args.get("content", "")
        tags = args.get("tags", "")
        priority = args.get("priority", "medium")
        folder = args.get("folder", "")
        
        # Create headers
        headers = {
            "Subject": subject,
            "Tags": tags,
            "Priority": priority,
            "Source": "FEI Assistant",
            "Date": get_current_date_iso()
        }
        
        # Create memory
        result = connector.create_memory(content, headers, folder)
        
        return {
            "success": True,
            "message": "Memory created successfully",
            "memory_id": result.get("id", "")
        }
        
    except Exception as e:
        logger.error(f"Error creating memory: {e}")
        return {
            "error": f"Error creating memory: {e}",
            "success": False
        }

def memory_view_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    View a specific memory
    
    Args:
        args: Tool arguments
        
    Returns:
        Memory details
    """
    try:
        # Read API key ONLY from environment variable
        api_key = os.environ.get("MEMDIR_API_KEY")
        connector = MemdirConnector(api_key=api_key, auto_start=True)

        # First check if the server is available and start it if needed
        start_result = connector.start_server_command()
        if start_result["status"] not in ["started", "already_running"]:
            return {
                "error": "Cannot connect to Memdir server. Server start failed."
            }
        
        # Extract arguments
        memory_id = args.get("memory_id", "")
        folder = args.get("folder", "")
        
        # Get memory
        memory = connector.get_memory(memory_id, folder)
        
        # Format for display
        headers = memory.get("headers", {})
        metadata = memory.get("metadata", {})
        content = memory.get("content", "")
        
        return {
            "id": metadata.get("unique_id", ""),
            "subject": headers.get("Subject", "No subject"),
            "date": metadata.get("date", ""),
            "tags": headers.get("Tags", ""),
            "priority": headers.get("Priority", ""),
            "status": headers.get("Status", ""),
            "flags": "".join(metadata.get("flags", [])),
            "content": content
        }
        
    except Exception as e:
        logger.error(f"Error viewing memory: {e}")
        return {"error": f"Error viewing memory: {e}"}

def memory_list_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    List memories in a folder
    
    Args:
        args: Tool arguments
        
    Returns:
        List of memories
    """
    try:
        # Read API key ONLY from environment variable
        api_key = os.environ.get("MEMDIR_API_KEY")
        connector = MemdirConnector(api_key=api_key, auto_start=True)

        # First check if the server is available and start it if needed
        start_result = connector.start_server_command()
        if start_result["status"] not in ["started", "already_running"]:
            return {
                "error": "Cannot connect to Memdir server. Server start failed.",
                "count": 0,
                "memories": []
            }
        
        # Extract arguments
        folder = args.get("folder", "")
        status = args.get("status", "cur")
        limit = args.get("limit")
        
        # Get memories
        memories = connector.list_memories(folder, status)
        
        # Apply limit if specified
        if limit is not None:
            memories = memories[:int(limit)]
        
        # Format for display
        formatted_memories = []
        for memory in memories:
            headers = memory.get("headers", {})
            metadata = memory.get("metadata", {})
            
            formatted_memories.append({
                "id": metadata.get("unique_id", ""),
                "subject": headers.get("Subject", "No subject"),
                "date": metadata.get("date", ""),
                "tags": headers.get("Tags", ""),
                "flags": "".join(metadata.get("flags", []))
            })
        
        return {
            "count": len(formatted_memories),
            "memories": formatted_memories
        }
        
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        return {
            "error": f"Error listing memories: {e}",
            "count": 0,
            "memories": []
        }

def memory_delete_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Delete a memory
    
    Args:
        args: Tool arguments
        
    Returns:
        Result of deletion
    """
    try:
        # Read API key ONLY from environment variable
        api_key = os.environ.get("MEMDIR_API_KEY")
        connector = MemdirConnector(api_key=api_key, auto_start=True)

        # First check if the server is available and start it if needed
        start_result = connector.start_server_command()
        if start_result["status"] not in ["started", "already_running"]:
            return {
                "error": "Cannot connect to Memdir server. Server start failed.",
                "success": False
            }
            
        # Extract arguments
        memory_id = args.get("memory_id", "")
        folder = args.get("folder", "")
        
        # Delete memory
        result = connector.delete_memory(memory_id, folder)
        
        return {
            "success": True,
            "message": "Memory deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return {
            "error": f"Error deleting memory: {e}",
            "success": False
        }

def memory_search_by_tag_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search for memories with a specific tag
    
    Args:
        args: Tool arguments
        
    Returns:
        Search results
    """
    try:
        # Read API key ONLY from environment variable
        api_key = os.environ.get("MEMDIR_API_KEY")
        connector = MemdirConnector(api_key=api_key, auto_start=True)

        # First check if the server is available and start it if needed
        start_result = connector.start_server_command()
        if start_result["status"] not in ["started", "already_running"]:
            return {
                "error": "Cannot connect to Memdir server. Server start failed.",
                "count": 0,
                "results": []
            }
            
        # Transform tag search into regular search
        tag = args.get("tag", "")
        if tag.startswith("#"):
            tag = tag[1:]
            
        # Create search query
        search_args = {
            "query": f"#tag:{tag}",
            "with_content": args.get("with_content", False)
        }
        
        # Call the regular search handler
        return memory_search_handler(search_args)
        
    except Exception as e:
        logger.error(f"Error searching by tag: {e}")
        return {
            "error": f"Error searching by tag: {e}",
            "count": 0,
            "results": []
        }

# Helper function for formatted date
def get_current_date_iso() -> str:
    """Get current date in ISO format"""
    from datetime import datetime
    return datetime.now().isoformat()

# Handlers for server management
def memdir_server_start_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Start the Memdir server
    
    Args:
        args: Tool arguments (unused)
        
    Returns:
        Status dictionary
    """
    try:
        # Read API key ONLY from environment variable for consistency
        api_key = os.environ.get("MEMDIR_API_KEY")
        # Pass the resolved API key to the connector instance
        connector = MemdirConnector(api_key=api_key, auto_start=True)
        return connector.start_server_command()
    except Exception as e:
        logger.error(f"Error starting Memdir server: {e}")
        return {"status": "error", "message": f"Error starting Memdir server: {e}"}

def memdir_server_stop_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stop the Memdir server
    
    Args:
        args: Tool arguments (unused)
        
    Returns:
        Status dictionary
    """
    try:
        connector = MemdirConnector()
        return connector.stop_server_command()
    except Exception as e:
        logger.error(f"Error stopping Memdir server: {e}")
        return {"status": "error", "message": f"Error stopping Memdir server: {e}"}

def memdir_server_status_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get the Memdir server status
    
    Args:
        args: Tool arguments (unused)
        
    Returns:
        Status dictionary
    """
    try:
        return MemdirConnector.get_server_status()
    except Exception as e:
        logger.error(f"Error getting Memdir server status: {e}")
        return {"status": "error", "message": f"Error getting Memdir server status: {e}"}

def create_memory_tools(registry: ToolRegistry) -> None:
    """
    Create and register memory management tools
    
    Args:
        registry: Tool registry to register with
    """
    # Register server management tools
    registry.register_tool(
        name="memdir_server_start",
        description="Start the Memdir server",
        input_schema=MEMORY_TOOL_SCHEMAS["memdir_server_start"],
        handler_func=memdir_server_start_handler,
        tags=["memory", "server"]
    )
    
    registry.register_tool(
        name="memdir_server_stop",
        description="Stop the Memdir server",
        input_schema=MEMORY_TOOL_SCHEMAS["memdir_server_stop"],
        handler_func=memdir_server_stop_handler,
        tags=["memory", "server"]
    )
    
    registry.register_tool(
        name="memdir_server_status",
        description="Get the status of the Memdir server",
        input_schema=MEMORY_TOOL_SCHEMAS["memdir_server_status"],
        handler_func=memdir_server_status_handler,
        tags=["memory", "server"]
    )
    
    # Register memory search tool
    registry.register_tool(
        name="memory_search",
        description="Search for memories using advanced query syntax",
        input_schema=MEMORY_TOOL_SCHEMAS["memory_search"],
        handler_func=memory_search_handler,
        tags=["memory"]
    )
    
    # Register memory create tool
    registry.register_tool(
        name="memory_create",
        description="Create a new memory",
        input_schema=MEMORY_TOOL_SCHEMAS["memory_create"],
        handler_func=memory_create_handler,
        tags=["memory"]
    )
    
    # Register memory view tool
    registry.register_tool(
        name="memory_view",
        description="View a specific memory by ID",
        input_schema=MEMORY_TOOL_SCHEMAS["memory_view"],
        handler_func=memory_view_handler,
        tags=["memory"]
    )
    
    # Register memory list tool
    registry.register_tool(
        name="memory_list",
        description="List memories in a folder",
        input_schema=MEMORY_TOOL_SCHEMAS["memory_list"],
        handler_func=memory_list_handler,
        tags=["memory"]
    )
    
    # Register memory delete tool
    registry.register_tool(
        name="memory_delete",
        description="Delete a memory",
        input_schema=MEMORY_TOOL_SCHEMAS["memory_delete"],
        handler_func=memory_delete_handler,
        tags=["memory"]
    )
    
    # Register memory search by tag tool
    registry.register_tool(
        name="memory_search_by_tag",
        description="Search for memories with a specific tag",
        input_schema=MEMORY_TOOL_SCHEMAS["memory_search_by_tag"],
        handler_func=memory_search_by_tag_handler,
        tags=["memory"]
    )


class MemoryManager:
    """
    High-level memory management system for FEI
    
    This class provides methods for working with both Memdir and Memorychain,
    offering a unified interface for all memory-related operations.
    """
    
    def __init__(self, use_memdir: bool = True, use_memorychain: bool = True):
        """
        Initialize memory manager
        
        Args:
            use_memdir: Whether to use Memdir
            use_memorychain: Whether to use Memorychain
        """
        self.use_memdir = use_memdir
        self.use_memorychain = use_memorychain
        
        # Initialize connectors
        self.memdir = MemdirConnector() if use_memdir else None
        self.memorychain = MemorychainConnector() if use_memorychain else None
        
        # Check connections
        self._check_connections()
    
    def _check_connections(self) -> Dict[str, bool]:
        """
        Check connections to memory systems
        
        Returns:
            Dictionary with connection statuses
        """
        status = {}
        
        if self.use_memdir:
            try:
                memdir_ok = self.memdir.check_connection()
                status["memdir"] = memdir_ok
            except Exception:
                status["memdir"] = False
                
        if self.use_memorychain:
            try:
                memorychain_ok = self.memorychain.check_connection()
                status["memorychain"] = memorychain_ok
            except Exception:
                status["memorychain"] = False
                
        return status
    
    def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Search for memories
        
        Args:
            query: Search query
            **kwargs: Additional search parameters
            
        Returns:
            Search results
        """
        all_results = {}
        
        if self.use_memdir:
            try:
                memdir_results = self.memdir.search(query, **kwargs)
                all_results["memdir"] = memdir_results
            except Exception as e:
                logger.error(f"Memdir search error: {e}")
                all_results["memdir"] = {"error": str(e)}
                
        if self.use_memorychain:
            try:
                memorychain_results = self.memorychain.search_memories(query)
                all_results["memorychain"] = {"results": memorychain_results}
            except Exception as e:
                logger.error(f"Memorychain search error: {e}")
                all_results["memorychain"] = {"error": str(e)}
                
        return all_results
    
    def create_memory(self, subject: str, content: str, tags: str = "", 
                     priority: str = "medium", folder: str = "") -> Dict[str, Any]:
        """
        Create a new memory
        
        Args:
            subject: Memory subject
            content: Memory content
            tags: Tags for the memory
            priority: Memory priority
            folder: Target folder
            
        Returns:
            Result dictionary
        """
        results = {}
        
        if self.use_memdir:
            try:
                headers = {
                    "Subject": subject,
                    "Tags": tags,
                    "Priority": priority,
                    "Source": "FEI Assistant",
                    "Date": get_current_date_iso()
                }
                
                memdir_result = self.memdir.create_memory(content, headers, folder)
                results["memdir"] = {
                    "success": True,
                    "id": memdir_result.get("id", "")
                }
            except Exception as e:
                logger.error(f"Memdir create error: {e}")
                results["memdir"] = {"error": str(e)}
                
        if self.use_memorychain:
            try:
                memorychain_result = self.memorychain.add_memory(
                    subject=subject,
                    content=content,
                    tags=tags,
                    priority=priority
                )
                results["memorychain"] = {
                    "success": True,
                    "block": memorychain_result.get("block", {})
                }
            except Exception as e:
                logger.error(f"Memorychain create error: {e}")
                results["memorychain"] = {"error": str(e)}
                
        return results
    
    def save_conversation(self, conversation: List[Dict[str, Any]], subject: str, 
                         tags: str = "conversation,fei") -> Dict[str, Any]:
        """
        Save a conversation as a memory
        
        Args:
            conversation: List of conversation messages
            subject: Memory subject
            tags: Tags for the memory
            
        Returns:
            Result dictionary
        """
        # Format the conversation
        formatted_content = ""
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "user":
                formatted_content += f"**User:** {content}\n\n"
            elif role == "assistant":
                formatted_content += f"**Assistant:** {content}\n\n"
            else:
                formatted_content += f"**{role}:** {content}\n\n"
                
        # Create the memory
        return self.create_memory(
            subject=subject,
            content=formatted_content,
            tags=tags,
            priority="medium"
        )
    
    def get_memory_by_id(self, memory_id: str, folder: str = "") -> Dict[str, Any]:
        """
        Get memory by ID
        
        Args:
            memory_id: Memory ID
            folder: Folder to search in
            
        Returns:
            Memory data
        """
        results = {}
        
        if self.use_memdir:
            try:
                memory = self.memdir.get_memory(memory_id, folder)
                results["memdir"] = memory
            except Exception as e:
                logger.error(f"Memdir get error: {e}")
                results["memdir"] = {"error": str(e)}
                
        if self.use_memorychain:
            try:
                memory = self.memorychain.get_memory_by_id(memory_id)
                results["memorychain"] = memory
            except Exception as e:
                logger.error(f"Memorychain get error: {e}")
                results["memorychain"] = {"error": str(e)}
                
        return results
