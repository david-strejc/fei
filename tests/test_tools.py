"""
Tests for the tool system enhancements
"""

import unittest
from unittest.mock import MagicMock, patch
import asyncio

from fei.tools.category import ToolCategory, ToolCategoryRegistry
from fei.tools.lifecycle import ToolLifecycleManager
from fei.tools.enhanced_registry import EnhancedToolRegistry


class TestToolSystem(unittest.TestCase):
    """Test the tool system enhancements"""
    
    def test_tool_category(self):
        """Test tool category"""
        # Create a category
        category = ToolCategory("test", "Test category")
        
        # Add a tool
        tool_data = {"name": "test_tool", "description": "Test tool"}
        category.add_tool("test_tool", tool_data)
        
        # Check if tool is in category
        self.assertIn("test_tool", category.list_tools())
        
        # Get tool
        tool = category.get_tool("test_tool")
        self.assertEqual(tool, tool_data)
        
        # Remove tool
        result = category.remove_tool("test_tool")
        self.assertTrue(result)
        
        # Check if tool is removed
        self.assertNotIn("test_tool", category.list_tools())
        
        # Try to remove non-existent tool
        result = category.remove_tool("non_existent")
        self.assertFalse(result)
    
    def test_tool_category_registry(self):
        """Test tool category registry"""
        # Create registry
        registry = ToolCategoryRegistry()
        
        # Check default categories
        default_categories = registry.list_categories()
        self.assertIn("code", default_categories)
        self.assertIn("memory", default_categories)
        self.assertIn("mcp", default_categories)
        self.assertIn("system", default_categories)
        self.assertIn("utility", default_categories)
        
        # Add a new category
        category = registry.add_category("test", "Test category")
        self.assertIsInstance(category, ToolCategory)
        
        # Check if category is in registry
        self.assertIn("test", registry.list_categories())
        
        # Get category
        category = registry.get_category("test")
        self.assertIsInstance(category, ToolCategory)
        self.assertEqual(category.name, "test")
        self.assertEqual(category.description, "Test category")
        
        # Add a tool to category
        tool_data = {"name": "test_tool", "description": "Test tool"}
        result = registry.add_tool_to_category("test", "test_tool", tool_data)
        self.assertTrue(result)
        
        # Get tool categories
        categories = registry.get_tool_categories("test_tool")
        self.assertIn("test", categories)
        
        # Get all tools
        tools = registry.get_all_tools()
        self.assertIn("test_tool", tools)
    
    def test_tool_lifecycle_manager(self):
        """Test tool lifecycle manager"""
        # Create manager
        manager = ToolLifecycleManager()
        
        # Register initializer
        initializer_mock = MagicMock(return_value={"initialized": True})
        manager.register_initializer("test_tool", initializer_mock)
        
        # Register cleanup
        cleanup_mock = MagicMock()
        manager.register_cleanup("test_tool", cleanup_mock)
        
        # Initialize tool
        result = manager.initialize_tool("test_tool")
        self.assertTrue(result)
        initializer_mock.assert_called_once()
        
        # Check tool state
        state = manager.get_tool_state("test_tool")
        self.assertEqual(state, {"initialized": True})
        
        # Clean up tool
        result = manager.cleanup_tool("test_tool")
        self.assertTrue(result)
        cleanup_mock.assert_called_once_with({"initialized": True})
        
        # Check if state is removed
        state = manager.get_tool_state("test_tool")
        self.assertIsNone(state)
    
    def test_enhanced_tool_registry(self):
        """Test enhanced tool registry"""
        # Create registry
        registry = EnhancedToolRegistry()
        
        # Register a tool
        def test_tool_handler(args):
            return {"result": args.get("input", "")}
        
        registry.register_tool(
            name="test_tool",
            description="Test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        # Check if tool is registered
        tools = registry.get_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_tool")
        
        # Get tools by category
        tools = registry.get_tools_by_category("test")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_tool")
        
        # Get tools by tag
        tools = registry.get_tools_by_tag("test")
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["name"], "test_tool")
        
        # Execute tool
        result = registry.execute_tool("test_tool", {"input": "test"})
        self.assertEqual(result, {"result": "test"})
        
        # List tool names
        names = registry.list_tool_names()
        self.assertIn("test_tool", names)
        
        # List categories
        categories = registry.list_categories()
        self.assertIn("test", categories)
        
        # List tags
        tags = registry.list_tags()
        self.assertIn("test", tags)
    
    def test_async_tool_execution(self):
        """Test async tool execution"""
        # Create registry
        registry = EnhancedToolRegistry()
        
        # Register an async tool
        async def async_test_tool_handler(args):
            await asyncio.sleep(0.1)
            return {"result": args.get("input", "")}
        
        registry.register_tool(
            name="async_test_tool",
            description="Async test tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            handler_func=async_test_tool_handler,
            tags=["test"],
            category="test"
        )
        
        # Execute tool
        result = registry.execute_tool("async_test_tool", {"input": "test"})
        self.assertEqual(result, {"result": "test"})
    
    def test_register_class_methods(self):
        """Test registering class methods as tools"""
        # Create registry
        registry = EnhancedToolRegistry()
        
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
        registry.register_class_methods(test_tools, prefix="math_", tags=["math"], category="math")
        
        # Check if tools are registered
        tools = registry.get_tools_by_category("math")
        self.assertEqual(len(tools), 2)
        
        # Execute tools
        result = registry.execute_tool("math_add", {"a": 2, "b": 3})
        self.assertEqual(result, 5)
        
        result = registry.execute_tool("math_subtract", {"a": 5, "b": 2})
        self.assertEqual(result, 3)


if __name__ == "__main__":
    unittest.main()
