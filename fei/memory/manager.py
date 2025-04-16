"""
Memory manager for Fei.
"""

from typing import Dict, List, Optional, Any, Union
import logging

from fei.memory.base import MemoryInterface, MemoryFactory
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manager for memory systems."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.memory_systems: Dict[str, MemoryInterface] = {}
        self.default_system = None
        
        # Initialize default memory system
        self._initialize_default_system()
    
    def _initialize_default_system(self) -> None:
        """Initialize the default memory system."""
        default_system = self.config.get("memory_system", "memdir")
        try:
            self.set_default_system(default_system)
        except Exception as e:
            logger.error(f"Failed to initialize default memory system '{default_system}': {e}")
    
    def get_memory_system(self, system_name: Optional[str] = None) -> MemoryInterface:
        """
        Get a memory system instance.
        
        Args:
            system_name: Optional memory system name, defaults to default system
            
        Returns:
            Memory system instance
            
        Raises:
            ValueError: If memory system is not available
        """
        system_name = system_name or self.default_system
        
        if not system_name:
            raise ValueError("No default memory system configured")
        
        if system_name not in self.memory_systems:
            try:
                memory_system = MemoryFactory.create_memory_system(system_name)
                memory_system.setup(self.config.get(f"{system_name}_config", {}))
                self.memory_systems[system_name] = memory_system
            except Exception as e:
                raise ValueError(f"Failed to create memory system '{system_name}': {e}")
        
        return self.memory_systems[system_name]
    
    def set_default_system(self, system_name: str) -> None:
        """
        Set the default memory system.
        
        Args:
            system_name: Memory system name
            
        Raises:
            ValueError: If memory system is not available
        """
        # Try to create the memory system to validate it
        memory_system = self.get_memory_system(system_name)
        
        # If successful, set as default
        self.default_system = system_name
    
    def list_available_systems(self) -> List[str]:
        """
        List all available memory systems.
        
        Returns:
            List of memory system names
        """
        return MemoryFactory.list_memory_systems()
    
    def search(self, query: str, limit: int = 10, system_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for memories matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            system_name: Optional memory system name
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memories
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.search(query, limit, **kwargs)
    
    def create(self, content: str, metadata: Optional[Dict[str, Any]] = None, system_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new memory.
        
        Args:
            content: Memory content
            metadata: Optional metadata
            system_name: Optional memory system name
            
        Returns:
            Created memory
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.create(content, metadata)
    
    def get(self, memory_id: str, system_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            system_name: Optional memory system name
            
        Returns:
            Memory or None if not found
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.get(memory_id)
    
    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None, system_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID
            content: Optional new content
            metadata: Optional new metadata
            system_name: Optional memory system name
            
        Returns:
            Updated memory or None if not found
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.update(memory_id, content, metadata)
    
    def delete(self, memory_id: str, system_name: Optional[str] = None) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
            system_name: Optional memory system name
            
        Returns:
            True if deleted, False if not found
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.delete(memory_id)
    
    def list(self, limit: int = 100, offset: int = 0, system_name: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        List memories.
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            system_name: Optional memory system name
            **kwargs: Additional filter parameters
            
        Returns:
            List of memories
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.list(limit, offset, **kwargs)
    
    def search_by_tag(self, tag: str, limit: int = 10, system_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for memories by tag.
        
        Args:
            tag: Tag to search for
            limit: Maximum number of results to return
            system_name: Optional memory system name
            
        Returns:
            List of matching memories
        """
        memory_system = self.get_memory_system(system_name)
        return memory_system.search_by_tag(tag, limit)
