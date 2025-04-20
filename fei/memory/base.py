"""
Base memory interface for Fei.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union


class MemoryInterface(ABC):
    """Abstract interface for memory systems."""
    
    @abstractmethod
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the memory system.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for memories matching a query.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            List of matching memories
        """
        pass
    
    @abstractmethod
    def create(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new memory.
        
        Args:
            content: Memory content
            metadata: Optional metadata
            
        Returns:
            Created memory
        """
        pass
    
    @abstractmethod
    def get(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a memory by ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Memory or None if not found
        """
        pass
    
    @abstractmethod
    def update(self, memory_id: str, content: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Update a memory.
        
        Args:
            memory_id: Memory ID
            content: Optional new content
            metadata: Optional new metadata
            
        Returns:
            Updated memory or None if not found
        """
        pass
    
    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0, **kwargs) -> List[Dict[str, Any]]:
        """
        List memories.
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            **kwargs: Additional filter parameters
            
        Returns:
            List of memories
        """
        pass
    
    @abstractmethod
    def search_by_tag(self, tag: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for memories by tag.
        
        Args:
            tag: Tag to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching memories
        """
        pass


class MemoryFactory:
    """Factory for creating memory system instances."""
    
    _memory_systems: Dict[str, type] = {}
    
    @classmethod
    def register_memory_system(cls, system_name: str, system_class: type) -> None:
        """
        Register a memory system class.
        
        Args:
            system_name: Name of the memory system
            system_class: Memory system class
        """
        cls._memory_systems[system_name] = system_class
    
    @classmethod
    def create_memory_system(cls, system_name: str, **kwargs) -> MemoryInterface:
        """
        Create a memory system instance.
        
        Args:
            system_name: Name of the memory system
            **kwargs: Additional arguments for the memory system
            
        Returns:
            Memory system instance
            
        Raises:
            ValueError: If memory system is not registered
        """
        if system_name not in cls._memory_systems:
            raise ValueError(f"Memory system '{system_name}' not registered")
        
        return cls._memory_systems[system_name](**kwargs)
    
    @classmethod
    def list_memory_systems(cls) -> List[str]:
        """
        List all registered memory systems.
        
        Returns:
            List of memory system names
        """
        return list(cls._memory_systems.keys())
