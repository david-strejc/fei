"""
Tests for the memory manager.
"""

import pytest
from unittest.mock import MagicMock, patch
import os

from fei.memory.base import MemoryInterface, MemoryFactory
from fei.memory.manager import MemoryManager
from fei.utils.config import Config


class TestMemoryManager:
    """Test the memory manager."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing memory systems
        MemoryFactory._memory_systems = {}
        
        # Register a mock memory system
        self.mock_memory_class = MagicMock()
        self.mock_memory = MagicMock(spec=MemoryInterface)
        self.mock_memory_class.return_value = self.mock_memory
        MemoryFactory.register_memory_system("mock", self.mock_memory_class)
        
        # Create a config with mock memory system
        self.config = Config()
        self.config.set("memory_system", "mock")
    
    def test_init(self):
        """Test initialization."""
        # Create manager
        manager = MemoryManager(self.config)
        
        # Check if memory system is created
        assert manager.memory_system is self.mock_memory
        self.mock_memory_class.assert_called_once()
        self.mock_memory.setup.assert_called_once()
    
    def test_search(self):
        """Test search."""
        # Set up mock response
        self.mock_memory.search.return_value = [{"id": "mock-id", "content": "Mock content"}]
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # Search
        results = manager.search("test query")
        
        # Check if search was called with correct arguments
        self.mock_memory.search.assert_called_once_with("test query", limit=10)
        
        # Check if results are correct
        assert results == [{"id": "mock-id", "content": "Mock content"}]
    
    def test_create(self):
        """Test create."""
        # Set up mock response
        self.mock_memory.create.return_value = {"id": "mock-id", "content": "Test content"}
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # Create memory
        memory = manager.create("Test content", {"tag": "test"})
        
        # Check if create was called with correct arguments
        self.mock_memory.create.assert_called_once_with("Test content", {"tag": "test"})
        
        # Check if memory is correct
        assert memory == {"id": "mock-id", "content": "Test content"}
    
    def test_get(self):
        """Test get."""
        # Set up mock response
        self.mock_memory.get.return_value = {"id": "mock-id", "content": "Mock content"}
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # Get memory
        memory = manager.get("mock-id")
        
        # Check if get was called with correct arguments
        self.mock_memory.get.assert_called_once_with("mock-id")
        
        # Check if memory is correct
        assert memory == {"id": "mock-id", "content": "Mock content"}
    
    def test_update(self):
        """Test update."""
        # Set up mock response
        self.mock_memory.update.return_value = {"id": "mock-id", "content": "Updated content"}
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # Update memory
        memory = manager.update("mock-id", content="Updated content")
        
        # Check if update was called with correct arguments
        self.mock_memory.update.assert_called_once_with("mock-id", content="Updated content", metadata=None)
        
        # Check if memory is correct
        assert memory == {"id": "mock-id", "content": "Updated content"}
    
    def test_delete(self):
        """Test delete."""
        # Set up mock response
        self.mock_memory.delete.return_value = True
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # Delete memory
        result = manager.delete("mock-id")
        
        # Check if delete was called with correct arguments
        self.mock_memory.delete.assert_called_once_with("mock-id")
        
        # Check if result is correct
        assert result is True
    
    def test_list(self):
        """Test list."""
        # Set up mock response
        self.mock_memory.list.return_value = [{"id": "mock-id", "content": "Mock content"}]
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # List memories
        memories = manager.list()
        
        # Check if list was called with correct arguments
        self.mock_memory.list.assert_called_once_with(limit=100, offset=0)
        
        # Check if memories are correct
        assert memories == [{"id": "mock-id", "content": "Mock content"}]
    
    def test_search_by_tag(self):
        """Test search by tag."""
        # Set up mock response
        self.mock_memory.search_by_tag.return_value = [{"id": "mock-id", "content": "Mock content", "tags": ["test-tag"]}]
        
        # Create manager
        manager = MemoryManager(self.config)
        
        # Search by tag
        results = manager.search_by_tag("test-tag")
        
        # Check if search_by_tag was called with correct arguments
        self.mock_memory.search_by_tag.assert_called_once_with("test-tag", limit=10)
        
        # Check if results are correct
        assert results == [{"id": "mock-id", "content": "Mock content", "tags": ["test-tag"]}]
    
    def test_get_memory_system(self):
        """Test get_memory_system."""
        # Create manager
        manager = MemoryManager(self.config)
        
        # Get memory system
        memory_system = manager.get_memory_system()
        
        # Check if memory system is correct
        assert memory_system is self.mock_memory
