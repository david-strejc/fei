#!/usr/bin/env python3
"""
Memorychain Connector for FEI

This module provides an interface for FEI (Flying Dragon of Adaptability) 
to interact with the Memorychain distributed memory system.

Key features:
- Connect to local or remote Memorychain nodes
- Propose new memories from FEI conversations
- Query the chain for relevant memories
- Extract memories that the current node is responsible for
- Validate and manage memory chain operations
"""

import os
import json
import time
import uuid
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memorychain-connector')

# Default configuration
DEFAULT_NODE = "localhost:6789"

class MemorychainConnector:
    """
    Connector class for FEI to interact with Memorychain
    
    This class provides methods to:
    - Connect to a Memorychain node
    - Add memories to the chain
    - Query memories from the chain
    - View chain statistics and status
    - Access memories by tag, content, and ID
    """
    
    def __init__(self, node_address: str = None):
        """
        Initialize the connector
        
        Args:
            node_address: Address of the Memorychain node (ip:port)
        """
        # First try environment variable
        if not node_address:
            node_address = os.environ.get("MEMORYCHAIN_NODE", DEFAULT_NODE)
            
        self.node_address = node_address
        self.node_url = f"http://{node_address}/memorychain"
        
        # Test connection
        try:
            self.check_connection()
            logger.info(f"Connected to Memorychain node at {node_address}")
        except Exception as e:
            logger.warning(f"Could not connect to Memorychain node at {node_address}: {e}")
            logger.warning("Operations will fail until a node is available")
    
    def check_connection(self) -> Dict[str, Any]:
        """
        Check if the node is available
        
        Returns:
            Node status information
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            response = requests.get(f"{self.node_url}/health", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Memorychain node: {e}")
            
    def get_node_status(self) -> Dict[str, Any]:
        """
        Get detailed status information for this node
        
        Returns:
            Node status information dictionary
            
        Raises:
            RequestException: If the request fails
        """
        response = requests.get(f"{self.node_url}/node_status", timeout=5)
        response.raise_for_status()
        return response.json()
        
    def update_status(self, 
                     status: Optional[str] = None, 
                     ai_model: Optional[str] = None,
                     current_task_id: Optional[str] = None,
                     load: Optional[float] = None) -> Dict[str, Any]:
        """
        Update the node's status information
        
        Args:
            status: Current node status (e.g., 'idle', 'busy', 'working_on_task')
            ai_model: AI model being used (e.g., 'claude-3-opus', 'gpt-4')
            current_task_id: ID of the task currently being worked on
            load: Node load factor from 0.0 to 1.0
            
        Returns:
            Response data containing update status
            
        Raises:
            RequestException: If the request fails
        """
        update_data = {}
        
        if status is not None:
            update_data["status"] = status
            
        if ai_model is not None:
            update_data["ai_model"] = ai_model
            
        if current_task_id is not None:
            update_data["current_task_id"] = current_task_id
            
        if load is not None:
            update_data["load"] = float(load)
            
        # Only send if we have data to update
        if update_data:
            response = requests.post(
                f"{self.node_url}/update_status",
                json=update_data,
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        else:
            return {"success": False, "message": "No update data provided"}
            
    def get_network_status(self) -> Dict[str, Any]:
        """
        Get status information for all nodes in the network
        
        Returns:
            Dictionary with status information for all connected nodes
            
        Raises:
            RequestException: If the request fails
        """
        response = requests.get(f"{self.node_url}/network_status", timeout=10)
        response.raise_for_status()
        return response.json()
    
    def add_memory(self, 
                 subject: str, 
                 content: str, 
                 tags: Optional[str] = None, 
                 priority: Optional[str] = None,
                 status: Optional[str] = None,
                 flags: Optional[str] = "") -> Dict[str, Any]:
        """
        Add a memory to the chain
        
        Args:
            subject: Memory subject/title
            content: Memory content
            tags: Optional comma-separated tags
            priority: Optional priority (high, medium, low)
            status: Optional status
            flags: Optional flags (e.g., "FP" for Flagged+Priority)
            
        Returns:
            Response data containing success status and block information
            
        Raises:
            RequestException: If the request fails
        """
        # Create memory data structure
        headers = {
            "Subject": subject
        }
        
        if tags:
            headers["Tags"] = tags
        if priority:
            headers["Priority"] = priority
        if status:
            headers["Status"] = status
        
        metadata = {
            "unique_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "date": datetime.now().isoformat(),
            "flags": list(flags) if flags else []
        }
        
        memory_data = {
            "headers": headers,
            "metadata": metadata,
            "content": content
        }
        
        # Submit to the chain
        response = requests.post(
            f"{self.node_url}/propose",
            json={"memory": memory_data},
            timeout=10
        )
        response.raise_for_status()
        
        result = response.json()
        if not result.get("success", False):
            raise ValueError(f"Memory proposal rejected: {result.get('message', 'unknown error')}")
            
        return result
    
    def get_chain(self) -> List[Dict[str, Any]]:
        """
        Get the entire chain
        
        Returns:
            List of blocks in the chain
            
        Raises:
            RequestException: If the request fails
        """
        response = requests.get(f"{self.node_url}/chain", timeout=10)
        response.raise_for_status()
        
        return response.json().get("chain", [])
    
    def get_responsible_memories(self) -> List[Dict[str, Any]]:
        """
        Get memories that this node is responsible for
        
        Returns:
            List of memory data dictionaries
            
        Raises:
            RequestException: If the request fails
        """
        response = requests.get(f"{self.node_url}/responsible_memories", timeout=10)
        response.raise_for_status()
        
        return response.json().get("memories", [])
    
    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a specific memory by ID
        
        Args:
            memory_id: Unique ID of the memory
            
        Returns:
            Memory data or None if not found
            
        Raises:
            RequestException: If the request fails
        """
        chain = self.get_chain()
        
        for block in chain:
            memory = block["memory_data"]
            if memory.get("metadata", {}).get("unique_id", "") == memory_id:
                return memory
                
        return None
    
    def search_memories(self, 
                       query: str, 
                       search_content: bool = True,
                       search_subject: bool = True,
                       search_tags: bool = True) -> List[Dict[str, Any]]:
        """
        Search for memories in the chain
        
        Args:
            query: Search query string
            search_content: Whether to search in content
            search_subject: Whether to search in subject
            search_tags: Whether to search in tags
            
        Returns:
            List of matching memories
            
        Raises:
            RequestException: If the request fails
        """
        chain = self.get_chain()
        results = []
        
        query = query.lower()
        
        for block in chain:
            memory = block["memory_data"]
            headers = memory.get("headers", {})
            content = memory.get("content", "")
            
            # Skip genesis block
            if memory.get("metadata", {}).get("unique_id", "") == "genesis":
                continue
            
            match = False
            
            # Search in subject
            if search_subject and headers.get("Subject", "").lower().find(query) >= 0:
                match = True
                
            # Search in tags
            if search_tags and headers.get("Tags", "").lower().find(query) >= 0:
                match = True
                
            # Search in content
            if search_content and content.lower().find(query) >= 0:
                match = True
                
            if match:
                results.append(memory)
                
        return results
    
    def search_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Search for memories with a specific tag
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of matching memories
            
        Raises:
            RequestException: If the request fails
        """
        chain = self.get_chain()
        results = []
        
        # Remove # prefix if present
        if tag.startswith("#"):
            tag = tag[1:]
            
        tag = tag.lower()
        
        for block in chain:
            memory = block["memory_data"]
            headers = memory.get("headers", {})
            tags_str = headers.get("Tags", "").lower()
            
            # Skip genesis block
            if memory.get("metadata", {}).get("unique_id", "") == "genesis":
                continue
                
            # Check if tag is in the tags list
            tags = [t.strip() for t in tags_str.split(",")]
            if tag in tags:
                results.append(memory)
                
        return results
    
    def get_memories_with_status(self, status: str) -> List[Dict[str, Any]]:
        """
        Get memories with a specific status
        
        Args:
            status: Status to filter by
            
        Returns:
            List of matching memories
            
        Raises:
            RequestException: If the request fails
        """
        chain = self.get_chain()
        results = []
        
        status = status.lower()
        
        for block in chain:
            memory = block["memory_data"]
            headers = memory.get("headers", {})
            memory_status = headers.get("Status", "").lower()
            
            # Skip genesis block
            if memory.get("metadata", {}).get("unique_id", "") == "genesis":
                continue
                
            if memory_status == status:
                results.append(memory)
                
        return results
    
    def get_chain_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the chain
        
        Returns:
            Dictionary of statistics
            
        Raises:
            RequestException: If the request fails
        """
        chain = self.get_chain()
        
        # Skip genesis block for stats
        memories = [block["memory_data"] for block in chain[1:]]
        
        # Count unique tags
        all_tags = set()
        for memory in memories:
            tags = memory.get("headers", {}).get("Tags", "")
            if tags:
                for tag in tags.split(","):
                    all_tags.add(tag.strip().lower())
        
        # Count unique statuses
        statuses = {}
        for memory in memories:
            status = memory.get("headers", {}).get("Status", "")
            if status:
                statuses[status] = statuses.get(status, 0) + 1
                
        # Count by priority
        priorities = {}
        for memory in memories:
            priority = memory.get("headers", {}).get("Priority", "")
            if priority:
                priorities[priority] = priorities.get(priority, 0) + 1
                
        # Get responsible nodes
        responsible_nodes = {}
        for block in chain[1:]:
            node = block["responsible_node"]
            responsible_nodes[node] = responsible_nodes.get(node, 0) + 1
            
        return {
            "total_blocks": len(chain),
            "total_memories": len(memories),
            "tags": list(all_tags),
            "tag_count": len(all_tags),
            "statuses": statuses,
            "priorities": priorities,
            "responsible_nodes": responsible_nodes
        }
    
    def format_memory(self, memory: Dict[str, Any], include_content: bool = True) -> str:
        """
        Format a memory for display
        
        Args:
            memory: Memory data
            include_content: Whether to include the content
            
        Returns:
            Formatted string representation
        """
        headers = memory.get("headers", {})
        metadata = memory.get("metadata", {})
        content = memory.get("content", "")
        
        memory_id = metadata.get("unique_id", "unknown")
        subject = headers.get("Subject", "No subject")
        tags = headers.get("Tags", "")
        status = headers.get("Status", "")
        priority = headers.get("Priority", "")
        flags = "".join(metadata.get("flags", []))
        
        # Format the date
        timestamp = metadata.get("timestamp", 0)
        date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # Build the output
        output = f"Memory: {memory_id}\n"
        output += f"Subject: {subject}\n"
        output += f"Date: {date_str}\n"
        
        if tags:
            output += f"Tags: {tags}\n"
        if status:
            output += f"Status: {status}\n"
        if priority:
            output += f"Priority: {priority}\n"
        if flags:
            output += f"Flags: {flags}\n"
            
        if include_content and content:
            output += "\nContent:\n"
            output += content
            
        return output
    
    def extract_memory_references(self, text: str) -> List[str]:
        """
        Extract memory references from text
        
        Looks for patterns like #mem:id or {mem:id}
        
        Args:
            text: Text to search for references
            
        Returns:
            List of memory IDs
        """
        import re
        pattern = r'(?:#mem:|{mem:)([a-z0-9-]+)(?:})?'
        matches = re.findall(pattern, text)
        return matches
    
    def resolve_memory_references(self, text: str) -> str:
        """
        Replace memory references with actual memory content
        
        Args:
            text: Text containing memory references
            
        Returns:
            Text with references replaced by memory summaries
        """
        import re
        
        # Find all memory references
        pattern = r'((?:#mem:|{mem:)([a-z0-9-]+)(?:})?)'
        
        def replace_reference(match):
            full_match = match.group(1)
            memory_id = match.group(2)
            
            # Try to find the memory
            memory = self.get_memory_by_id(memory_id)
            if not memory:
                return f"{full_match} (not found)"
                
            # Create a short summary
            subject = memory.get("headers", {}).get("Subject", "No subject")
            return f"{full_match} ({subject})"
            
        # Replace all references
        return re.sub(pattern, replace_reference, text)

    def validate_chain(self) -> bool:
        """
        Validate the integrity of the memory chain
        
        Returns:
            True if valid, False otherwise
            
        Raises:
            RequestException: If the request fails
        """
        try:
            response = requests.get(f"{self.node_url}/validate", timeout=10)
            response.raise_for_status()
            return response.json().get("valid", False)
        except:
            # If validation endpoint doesn't exist, use the connector's validation method
            try:
                # Import here to avoid circular imports
                from memdir_tools.memorychain import MemoryChain, MemoryBlock
                
                # Get the chain
                chain_data = self.get_chain()
                
                # Create a temporary chain for validation
                temp_chain = MemoryChain("validator")
                
                # Replace the chain with the data we got
                temp_chain.chain = [MemoryBlock.from_dict(block) for block in chain_data]
                
                # Validate
                return temp_chain.validate_chain()
            except Exception as e:
                logger.error(f"Error validating chain: {e}")
                return False

# Helper functions

def get_connector(node_address: Optional[str] = None) -> MemorychainConnector:
    """
    Get a connector instance, optionally specifying a node address
    
    Args:
        node_address: Optional node address
        
    Returns:
        MemorychainConnector instance
    """
    return MemorychainConnector(node_address)

def add_memory_from_conversation(connector: MemorychainConnector, 
                               conversation: List[Dict[str, Any]],
                               subject: Optional[str] = None,
                               tags: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract a memory from a conversation
    
    Args:
        connector: MemorychainConnector instance
        conversation: List of conversation messages
        subject: Optional subject (default: auto-generate)
        tags: Optional tags
        
    Returns:
        Response from add_memory
    """
    # Format the conversation
    formatted_content = ""
    for msg in conversation:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            formatted_content += f"User: {content}\n\n"
        elif role == "assistant":
            formatted_content += f"Assistant: {content}\n\n"
        else:
            formatted_content += f"{role}: {content}\n\n"
    
    # Generate a subject if not provided
    if not subject:
        # Use the first few words of the first user message
        for msg in conversation:
            if msg.get("role") == "user":
                text = msg.get("content", "")
                words = text.split()
                subject = " ".join(words[:5])
                if len(words) > 5:
                    subject += "..."
                break
                
        # Fallback if no user messages
        if not subject:
            subject = f"Conversation from {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    # Add conversation as a memory
    return connector.add_memory(
        subject=subject,
        content=formatted_content,
        tags=tags or "conversation",
        flags="F"  # Flag by default
    )

# Module test code
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python memorychain_connector.py <command> [args...]")
        print("Commands: check, stats, search, responsible, tag, status")
        sys.exit(1)
        
    command = sys.argv[1]
    connector = get_connector()
    
    try:
        if command == "check":
            status = connector.check_connection()
            print(f"Connected to node: {connector.node_address}")
            print(f"Node ID: {status.get('node_id', 'unknown')}")
            print(f"Chain length: {status.get('chain_length', 0)} blocks")
            print(f"Connected nodes: {status.get('connected_nodes', 0)}")
            
        elif command == "stats":
            stats = connector.get_chain_stats()
            print(f"Memory Chain Statistics:")
            print(f"Total blocks: {stats['total_blocks']}")
            print(f"Total memories: {stats['total_memories']}")
            print(f"Unique tags: {stats['tag_count']}")
            print(f"Tags: {', '.join(stats['tags'])}")
            print(f"Statuses: {stats['statuses']}")
            print(f"Priorities: {stats['priorities']}")
            print(f"Responsible nodes: {stats['responsible_nodes']}")
            
        elif command == "search" and len(sys.argv) > 2:
            query = sys.argv[2]
            memories = connector.search_memories(query)
            print(f"Found {len(memories)} memories matching '{query}':")
            for memory in memories:
                headers = memory.get("headers", {})
                metadata = memory.get("metadata", {})
                print(f"- {metadata.get('unique_id', '')}: {headers.get('Subject', 'No subject')}")
                
        elif command == "responsible":
            memories = connector.get_responsible_memories()
            print(f"This node is responsible for {len(memories)} memories:")
            for memory in memories:
                headers = memory.get("headers", {})
                metadata = memory.get("metadata", {})
                print(f"- {metadata.get('unique_id', '')}: {headers.get('Subject', 'No subject')}")
                
        elif command == "tag" and len(sys.argv) > 2:
            tag = sys.argv[2]
            memories = connector.search_by_tag(tag)
            print(f"Found {len(memories)} memories with tag '{tag}':")
            for memory in memories:
                headers = memory.get("headers", {})
                metadata = memory.get("metadata", {})
                print(f"- {metadata.get('unique_id', '')}: {headers.get('Subject', 'No subject')}")
                
        elif command == "status" and len(sys.argv) > 2:
            status = sys.argv[2]
            memories = connector.get_memories_with_status(status)
            print(f"Found {len(memories)} memories with status '{status}':")
            for memory in memories:
                headers = memory.get("headers", {})
                metadata = memory.get("metadata", {})
                print(f"- {metadata.get('unique_id', '')}: {headers.get('Subject', 'No subject')}")
                
        else:
            print(f"Unknown command: {command}")
            print("Commands: check, stats, search, responsible, tag, status")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)