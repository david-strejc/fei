"""
Tests for the enhanced tool registry.
"""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from fei.tools.enhanced_registry import EnhancedToolRegistry


class TestEnhancedToolRegistry:
    """Test the EnhancedToolRegistry class."""
    
    def setup_method(self):
        """Set up the test."""
        self.registry = EnhancedToolRegistry()
    
    def test_init(self):
        """Test initialization."""
        # Check if registry is initialized correctly
        assert hasattr(self.registry, "tools")
        assert hasattr(self.registry, "category_registry")
        assert hasattr(self.registry, "lifecycle_manager")
    
    def test_register_tool(self):
        """Test register_tool."""
        # Register a tool
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        # Check if tool is registered
        tools = self.registry.get_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["description"] == "Test tool"
        assert tools[0]["tags"] == ["test"]
        
        # Check if tool is added to category
        assert "test_tool" in self.registry.category_registry.get_category("test").list_tools()
    
    def test_register_tool_with_initializer(self):
        """Test register_tool with initializer."""
        # Register a tool with initializer
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        def test_tool_initializer():
            return {"initialized": True}
        
        self.registry.register_tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            initializer_func=test_tool_initializer,
            tags=["test"],
            category="test"
        )
        
        # Check if initializer is registered
        assert "test_tool" in self.registry.lifecycle_manager.initializers
    
    def test_register_tool_with_cleanup(self):
        """Test register_tool with cleanup."""
        # Register a tool with cleanup
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        def test_tool_cleanup(state):
            pass
        
        self.registry.register_tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            cleanup_func=test_tool_cleanup,
            tags=["test"],
            category="test"
        )
        
        # Check if cleanup is registered
        assert "test_tool" in self.registry.lifecycle_manager.cleanups
    
    def test_get_tools(self):
        """Test get_tools."""
        # Register some tools
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test1"],
            category="test"
        )
        
        self.registry.register_tool(
            name="test_tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test2"],
            category="test"
        )
        
        # Get tools
        tools = self.registry.get_tools()
        
        # Check if tools are correct
        assert len(tools) == 2
        assert tools[0]["name"] == "test_tool1"
        assert tools[1]["name"] == "test_tool2"
    
    def test_get_tools_by_category(self):
        """Test get_tools_by_category."""
        # Register tools in different categories
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="category1"
        )
        
        self.registry.register_tool(
            name="test_tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="category2"
        )
        
        # Get tools by category
        tools = self.registry.get_tools_by_category("category1")
        
        # Check if tools are correct
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool1"
    
    def test_get_tools_by_tag(self):
        """Test get_tools_by_tag."""
        # Register tools with different tags
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["tag1"],
            category="test"
        )
        
        self.registry.register_tool(
            name="test_tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["tag2"],
            category="test"
        )
        
        # Get tools by tag
        tools = self.registry.get_tools_by_tag("tag1")
        
        # Check if tools are correct
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool1"
    
    def test_execute_tool(self):
        """Test execute_tool."""
        # Register a tool
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        # Execute tool
        result = self.registry.execute_tool("test_tool", {"input": "test"})
        
        # Check if result is correct
        assert result == {"result": "test"}
    
    def test_execute_nonexistent_tool(self):
        """Test execute_tool with non-existent tool."""
        # Execute non-existent tool
        with pytest.raises(ValueError):
            self.registry.execute_tool("nonexistent", {})
    
    def test_execute_async_tool(self):
        """Test execute_tool with async handler."""
        # Register an async tool
        async def async_test_tool_handler(args):
            await asyncio.sleep(0.1)
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="async_test_tool",
            description="Async test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=async_test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        # Execute tool
        result = self.registry.execute_tool("async_test_tool", {"input": "test"})
        
        # Check if result is correct
        assert result == {"result": "test"}
    
    def test_list_tool_names(self):
        """Test list_tool_names."""
        # Register some tools
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        self.registry.register_tool(
            name="test_tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        # List tool names
        names = self.registry.list_tool_names()
        
        # Check if names are correct
        assert set(names) == {"test_tool1", "test_tool2"}
    
    def test_list_categories(self):
        """Test list_categories."""
        # Register tools in different categories
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="category1"
        )
        
        self.registry.register_tool(
            name="test_tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="category2"
        )
        
        # List categories
        categories = self.registry.list_categories()
        
        # Check if categories are correct
        assert "category1" in categories
        assert "category2" in categories
    
    def test_list_tags(self):
        """Test list_tags."""
        # Register tools with different tags
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        self.registry.register_tool(
            name="test_tool1",
            description="Test tool 1",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["tag1"],
            category="test"
        )
        
        self.registry.register_tool(
            name="test_tool2",
            description="Test tool 2",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["tag2"],
            category="test"
        )
        
        # List tags
        tags = self.registry.list_tags()
        
        # Check if tags are correct
        assert "tag1" in tags
        assert "tag2" in tags
    
    def test_cleanup(self):
        """Test cleanup."""
        # Register a tool with cleanup
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        cleanup = MagicMock()
        
        self.registry.register_tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            cleanup_func=cleanup,
            tags=["test"],
            category="test"
        )
        
        # Initialize tool
        self.registry.lifecycle_manager.states["test_tool"] = {"initialized": True}
        
        # Clean up
        self.registry.cleanup()
        
        # Check if cleanup was called
        cleanup.assert_called_once_with({"initialized": True})
    
    def test_register_class_methods(self):
        """Test register_class_methods."""
        # Create a class with methods
        class TestTools:
            def add(self, a: int, b: int) -> int:
                """Add two numbers"""
                return a + b
            
            def subtract(self, a: int, b: int) -> int:
                """Subtract b from a"""
                return a - b
        
        # Register class methods
        test_tools = TestTools()
        self.registry.register_class_methods(test_tools, prefix="math_", tags=["math"], category="math")
        
        # Check if tools are registered
        tools = self.registry.get_tools_by_category("math")
        assert len(tools) == 2
        
        # Check if tools have correct names
        tool_names = [tool["name"] for tool in tools]
        assert "math_add" in tool_names
        assert "math_subtract" in tool_names
        
        # Execute tools
        add_result = self.registry.execute_tool("math_add", {"a": 2, "b": 3})
        subtract_result = self.registry.execute_tool("math_subtract", {"a": 5, "b": 2})
        
        # Check if results are correct
        assert add_result == 5
        assert subtract_result == 3
