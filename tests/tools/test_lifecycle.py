"""
Tests for the tool lifecycle manager.
"""

import pytest
from unittest.mock import MagicMock

from fei.tools.lifecycle import ToolLifecycleManager


class TestToolLifecycleManager:
    """Test the ToolLifecycleManager class."""
    
    def setup_method(self):
        """Set up the test."""
        self.manager = ToolLifecycleManager()
    
    def test_init(self):
        """Test initialization."""
        # Check if manager is initialized correctly
        assert self.manager.initializers == {}
        assert self.manager.cleanups == {}
        assert self.manager.states == {}
    
    def test_register_initializer(self):
        """Test register_initializer."""
        # Register an initializer
        initializer = MagicMock(return_value={"initialized": True})
        self.manager.register_initializer("test_tool", initializer)
        
        # Check if initializer is registered
        assert "test_tool" in self.manager.initializers
        assert self.manager.initializers["test_tool"] is initializer
    
    def test_register_cleanup(self):
        """Test register_cleanup."""
        # Register a cleanup
        cleanup = MagicMock()
        self.manager.register_cleanup("test_tool", cleanup)
        
        # Check if cleanup is registered
        assert "test_tool" in self.manager.cleanups
        assert self.manager.cleanups["test_tool"] is cleanup
    
    def test_initialize_tool(self):
        """Test initialize_tool."""
        # Register an initializer
        initializer = MagicMock(return_value={"initialized": True})
        self.manager.register_initializer("test_tool", initializer)
        
        # Initialize tool
        result = self.manager.initialize_tool("test_tool")
        
        # Check if initializer was called
        initializer.assert_called_once()
        
        # Check if result is correct
        assert result is True
        
        # Check if state is stored
        assert "test_tool" in self.manager.states
        assert self.manager.states["test_tool"] == {"initialized": True}
    
    def test_initialize_tool_with_args(self):
        """Test initialize_tool with arguments."""
        # Register an initializer
        initializer = MagicMock(return_value={"initialized": True})
        self.manager.register_initializer("test_tool", initializer)
        
        # Initialize tool with arguments
        result = self.manager.initialize_tool("test_tool", arg1="value1", arg2="value2")
        
        # Check if initializer was called with correct arguments
        initializer.assert_called_once_with(arg1="value1", arg2="value2")
        
        # Check if result is correct
        assert result is True
    
    def test_initialize_nonexistent_tool(self):
        """Test initialize_tool with non-existent tool."""
        # Initialize non-existent tool
        result = self.manager.initialize_tool("nonexistent")
        
        # Check if result is False
        assert result is False
    
    def test_initialize_tool_exception(self):
        """Test initialize_tool with exception."""
        # Register an initializer that raises an exception
        initializer = MagicMock(side_effect=ValueError("Test error"))
        self.manager.register_initializer("test_tool", initializer)
        
        # Initialize tool
        result = self.manager.initialize_tool("test_tool")
        
        # Check if result is False
        assert result is False
        
        # Check if state is not stored
        assert "test_tool" not in self.manager.states
    
    def test_cleanup_tool(self):
        """Test cleanup_tool."""
        # Register a cleanup
        cleanup = MagicMock()
        self.manager.register_cleanup("test_tool", cleanup)
        
        # Store a state
        self.manager.states["test_tool"] = {"initialized": True}
        
        # Clean up tool
        result = self.manager.cleanup_tool("test_tool")
        
        # Check if cleanup was called with correct arguments
        cleanup.assert_called_once_with({"initialized": True})
        
        # Check if result is correct
        assert result is True
        
        # Check if state is removed
        assert "test_tool" not in self.manager.states
    
    def test_cleanup_nonexistent_tool(self):
        """Test cleanup_tool with non-existent tool."""
        # Clean up non-existent tool
        result = self.manager.cleanup_tool("nonexistent")
        
        # Check if result is False
        assert result is False
    
    def test_cleanup_tool_exception(self):
        """Test cleanup_tool with exception."""
        # Register a cleanup that raises an exception
        cleanup = MagicMock(side_effect=ValueError("Test error"))
        self.manager.register_cleanup("test_tool", cleanup)
        
        # Store a state
        self.manager.states["test_tool"] = {"initialized": True}
        
        # Clean up tool
        result = self.manager.cleanup_tool("test_tool")
        
        # Check if result is False
        assert result is False
        
        # Check if state is still stored
        assert "test_tool" in self.manager.states
    
    def test_get_tool_state(self):
        """Test get_tool_state."""
        # Store a state
        self.manager.states["test_tool"] = {"initialized": True}
        
        # Get tool state
        state = self.manager.get_tool_state("test_tool")
        
        # Check if state is correct
        assert state == {"initialized": True}
    
    def test_get_nonexistent_tool_state(self):
        """Test get_tool_state with non-existent tool."""
        # Get state for non-existent tool
        state = self.manager.get_tool_state("nonexistent")
        
        # Check if state is None
        assert state is None
    
    def test_cleanup_all(self):
        """Test cleanup_all."""
        # Register cleanups
        cleanup1 = MagicMock()
        cleanup2 = MagicMock()
        self.manager.register_cleanup("test_tool1", cleanup1)
        self.manager.register_cleanup("test_tool2", cleanup2)
        
        # Store states
        self.manager.states["test_tool1"] = {"initialized": True}
        self.manager.states["test_tool2"] = {"initialized": True}
        
        # Clean up all tools
        results = self.manager.cleanup_all()
        
        # Check if cleanups were called with correct arguments
        cleanup1.assert_called_once_with({"initialized": True})
        cleanup2.assert_called_once_with({"initialized": True})
        
        # Check if results are correct
        assert results == {"test_tool1": True, "test_tool2": True}
        
        # Check if states are removed
        assert "test_tool1" not in self.manager.states
        assert "test_tool2" not in self.manager.states
    
    def test_cleanup_all_with_exception(self):
        """Test cleanup_all with exception."""
        # Register cleanups
        cleanup1 = MagicMock(side_effect=ValueError("Test error"))
        cleanup2 = MagicMock()
        self.manager.register_cleanup("test_tool1", cleanup1)
        self.manager.register_cleanup("test_tool2", cleanup2)
        
        # Store states
        self.manager.states["test_tool1"] = {"initialized": True}
        self.manager.states["test_tool2"] = {"initialized": True}
        
        # Clean up all tools
        results = self.manager.cleanup_all()
        
        # Check if cleanups were called
        cleanup1.assert_called_once_with({"initialized": True})
        cleanup2.assert_called_once_with({"initialized": True})
        
        # Check if results are correct
        assert results == {"test_tool1": False, "test_tool2": True}
        
        # Check if states are removed or still stored
        assert "test_tool1" in self.manager.states
        assert "test_tool2" not in self.manager.states
