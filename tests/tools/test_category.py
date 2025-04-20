"""
Tests for the tool category system.
"""

import pytest
from unittest.mock import MagicMock

from fei.tools.category import ToolCategory, ToolCategoryRegistry


class TestToolCategory:
    """Test the ToolCategory class."""
    
    def setup_method(self):
        """Set up the test."""
        self.category = ToolCategory("test", "Test category")
    
    def test_init(self):
        """Test initialization."""
        # Check if category is initialized correctly
        assert self.category.name == "test"
        assert self.category.description == "Test category"
        assert self.category.tools == {}
    
    def test_add_tool(self):
        """Test add_tool."""
        # Add a tool
        tool_data = {"name": "test_tool", "description": "Test tool"}
        self.category.add_tool("test_tool", tool_data)
        
        # Check if tool is added
        assert "test_tool" in self.category.tools
        assert self.category.tools["test_tool"] == tool_data
    
    def test_get_tool(self):
        """Test get_tool."""
        # Add a tool
        tool_data = {"name": "test_tool", "description": "Test tool"}
        self.category.add_tool("test_tool", tool_data)
        
        # Get tool
        tool = self.category.get_tool("test_tool")
        
        # Check if tool is correct
        assert tool == tool_data
    
    def test_get_nonexistent_tool(self):
        """Test get_tool with non-existent tool."""
        # Get non-existent tool
        tool = self.category.get_tool("nonexistent")
        
        # Check if tool is None
        assert tool is None
    
    def test_remove_tool(self):
        """Test remove_tool."""
        # Add a tool
        tool_data = {"name": "test_tool", "description": "Test tool"}
        self.category.add_tool("test_tool", tool_data)
        
        # Remove tool
        result = self.category.remove_tool("test_tool")
        
        # Check if tool is removed
        assert result is True
        assert "test_tool" not in self.category.tools
    
    def test_remove_nonexistent_tool(self):
        """Test remove_tool with non-existent tool."""
        # Remove non-existent tool
        result = self.category.remove_tool("nonexistent")
        
        # Check if result is False
        assert result is False
    
    def test_list_tools(self):
        """Test list_tools."""
        # Add some tools
        self.category.add_tool("test_tool1", {"name": "test_tool1", "description": "Test tool 1"})
        self.category.add_tool("test_tool2", {"name": "test_tool2", "description": "Test tool 2"})
        
        # List tools
        tools = self.category.list_tools()
        
        # Check if tools are correct
        assert set(tools) == {"test_tool1", "test_tool2"}
    
    def test_get_tools(self):
        """Test get_tools."""
        # Add some tools
        tool1 = {"name": "test_tool1", "description": "Test tool 1"}
        tool2 = {"name": "test_tool2", "description": "Test tool 2"}
        self.category.add_tool("test_tool1", tool1)
        self.category.add_tool("test_tool2", tool2)
        
        # Get tools
        tools = self.category.get_tools()
        
        # Check if tools are correct
        assert set(tools) == {tool1, tool2}


class TestToolCategoryRegistry:
    """Test the ToolCategoryRegistry class."""
    
    def setup_method(self):
        """Set up the test."""
        self.registry = ToolCategoryRegistry()
    
    def test_init(self):
        """Test initialization."""
        # Check if registry is initialized with default categories
        assert "code" in self.registry.categories
        assert "memory" in self.registry.categories
        assert "mcp" in self.registry.categories
        assert "system" in self.registry.categories
        assert "utility" in self.registry.categories
    
    def test_add_category(self):
        """Test add_category."""
        # Add a category
        category = self.registry.add_category("test", "Test category")
        
        # Check if category is added
        assert "test" in self.registry.categories
        assert self.registry.categories["test"] is category
        assert category.name == "test"
        assert category.description == "Test category"
    
    def test_add_existing_category(self):
        """Test add_category with existing category."""
        # Add a category
        self.registry.add_category("test", "Test category")
        
        # Add the same category again
        category = self.registry.add_category("test", "New description")
        
        # Check if category is updated
        assert category.description == "New description"
    
    def test_get_category(self):
        """Test get_category."""
        # Add a category
        self.registry.add_category("test", "Test category")
        
        # Get category
        category = self.registry.get_category("test")
        
        # Check if category is correct
        assert category.name == "test"
        assert category.description == "Test category"
    
    def test_get_nonexistent_category(self):
        """Test get_category with non-existent category."""
        # Get non-existent category
        with pytest.raises(KeyError):
            self.registry.get_category("nonexistent")
    
    def test_list_categories(self):
        """Test list_categories."""
        # Add a category
        self.registry.add_category("test", "Test category")
        
        # List categories
        categories = self.registry.list_categories()
        
        # Check if categories include default and added ones
        assert "code" in categories
        assert "memory" in categories
        assert "mcp" in categories
        assert "system" in categories
        assert "utility" in categories
        assert "test" in categories
    
    def test_add_tool_to_category(self):
        """Test add_tool_to_category."""
        # Add a tool to a category
        tool_data = {"name": "test_tool", "description": "Test tool"}
        result = self.registry.add_tool_to_category("code", "test_tool", tool_data)
        
        # Check if tool is added
        assert result is True
        assert "test_tool" in self.registry.get_category("code").list_tools()
    
    def test_add_tool_to_nonexistent_category(self):
        """Test add_tool_to_category with non-existent category."""
        # Add a tool to a non-existent category
        tool_data = {"name": "test_tool", "description": "Test tool"}
        with pytest.raises(KeyError):
            self.registry.add_tool_to_category("nonexistent", "test_tool", tool_data)
    
    def test_get_tool_categories(self):
        """Test get_tool_categories."""
        # Add a tool to multiple categories
        tool_data = {"name": "test_tool", "description": "Test tool"}
        self.registry.add_tool_to_category("code", "test_tool", tool_data)
        self.registry.add_tool_to_category("utility", "test_tool", tool_data)
        
        # Get tool categories
        categories = self.registry.get_tool_categories("test_tool")
        
        # Check if categories are correct
        assert set(categories) == {"code", "utility"}
    
    def test_get_tool_categories_nonexistent_tool(self):
        """Test get_tool_categories with non-existent tool."""
        # Get categories for non-existent tool
        categories = self.registry.get_tool_categories("nonexistent")
        
        # Check if categories are empty
        assert categories == []
    
    def test_get_all_tools(self):
        """Test get_all_tools."""
        # Add tools to different categories
        tool1 = {"name": "test_tool1", "description": "Test tool 1"}
        tool2 = {"name": "test_tool2", "description": "Test tool 2"}
        self.registry.add_tool_to_category("code", "test_tool1", tool1)
        self.registry.add_tool_to_category("utility", "test_tool2", tool2)
        
        # Get all tools
        tools = self.registry.get_all_tools()
        
        # Check if tools are correct
        assert "test_tool1" in tools
        assert "test_tool2" in tools
        assert tools["test_tool1"] == tool1
        assert tools["test_tool2"] == tool2
