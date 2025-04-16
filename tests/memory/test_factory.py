"""
Tests for the memory factory.
"""

import pytest
from unittest.mock import MagicMock

from fei.memory.base import MemoryInterface, MemoryFactory


class TestMemoryFactory:
    """Test the memory factory."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing memory systems
        MemoryFactory._memory_systems = {}
    
    def test_register_memory_system(self):
        """Test registering a memory system."""
        # Create a mock memory system class
        mock_memory_class = MagicMock(spec=MemoryInterface)
        
        # Register the memory system
        MemoryFactory.register_memory_system("mock", mock_memory_class)
        
        # Check if memory system is registered
        assert "mock" in MemoryFactory.list_memory_systems()
    
    def test_create_memory_system(self):
        """Test creating a memory system."""
        # Create a mock memory system class
        mock_memory_class = MagicMock(return_value=MagicMock(spec=MemoryInterface))
        
        # Register the memory system
        MemoryFactory.register_memory_system("mock", mock_memory_class)
        
        # Create a memory system instance
        memory = MemoryFactory.create_memory_system("mock")
        
        # Check if memory system was created
        mock_memory_class.assert_called_once()
        assert memory is mock_memory_class.return_value
    
    def test_create_nonexistent_memory_system(self):
        """Test creating a non-existent memory system."""
        # Try to create a non-existent memory system
        with pytest.raises(ValueError):
            MemoryFactory.create_memory_system("nonexistent")
    
    def test_list_memory_systems(self):
        """Test listing memory systems."""
        # Register some memory systems
        MemoryFactory.register_memory_system("mock1", MagicMock())
        MemoryFactory.register_memory_system("mock2", MagicMock())
        
        # List memory systems
        memory_systems = MemoryFactory.list_memory_systems()
        
        # Check if all memory systems are listed
        assert "mock1" in memory_systems
        assert "mock2" in memory_systems
        assert len(memory_systems) == 2
