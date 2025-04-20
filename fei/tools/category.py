"""
Tool category system for Fei.
"""

from typing import Dict, List, Optional, Any, Set


class ToolCategory:
    """
    Represents a category of tools.
    
    Categories help organize tools by functionality and provide additional metadata.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a tool category.
        
        Args:
            name: Category name
            description: Category description
        """
        self.name = name
        self.description = description
        self.tools = {}
    
    def add_tool(self, tool_name: str, tool_data: Dict[str, Any]) -> None:
        """
        Add a tool to this category.
        
        Args:
            tool_name: Name of the tool
            tool_data: Tool data dictionary
        """
        self.tools[tool_name] = tool_data
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from this category.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            True if tool was removed, False if not found
        """
        if tool_name in self.tools:
            del self.tools[tool_name]
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool data dictionary or None if not found
        """
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        List all tools in this category.
        
        Returns:
            List of tool names
        """
        return list(self.tools.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert category to dictionary.
        
        Returns:
            Dictionary representation of the category
        """
        return {
            "name": self.name,
            "description": self.description,
            "tools": list(self.tools.keys())
        }


class ToolCategoryRegistry:
    """Registry for tool categories."""
    
    def __init__(self):
        """Initialize the tool category registry."""
        self.categories: Dict[str, ToolCategory] = {}
        
        # Create default categories
        self._create_default_categories()
    
    def _create_default_categories(self) -> None:
        """Create default tool categories."""
        self.add_category("code", "Tools for working with code and files")
        self.add_category("memory", "Tools for working with memory")
        self.add_category("mcp", "Model-Controlled Process tools")
        self.add_category("system", "System tools")
        self.add_category("utility", "Utility tools")
    
    def add_category(self, name: str, description: str) -> ToolCategory:
        """
        Add a new category.
        
        Args:
            name: Category name
            description: Category description
            
        Returns:
            The created category
        """
        category = ToolCategory(name, description)
        self.categories[name] = category
        return category
    
    def get_category(self, name: str) -> Optional[ToolCategory]:
        """
        Get a category by name.
        
        Args:
            name: Category name
            
        Returns:
            Category or None if not found
        """
        return self.categories.get(name)
    
    def list_categories(self) -> List[str]:
        """
        List all category names.
        
        Returns:
            List of category names
        """
        return list(self.categories.keys())
    
    def add_tool_to_category(self, category_name: str, tool_name: str, tool_data: Dict[str, Any]) -> bool:
        """
        Add a tool to a category.
        
        Args:
            category_name: Category name
            tool_name: Tool name
            tool_data: Tool data dictionary
            
        Returns:
            True if tool was added, False if category not found
        """
        category = self.get_category(category_name)
        if category:
            category.add_tool(tool_name, tool_data)
            return True
        return False
    
    def get_tool_categories(self, tool_name: str) -> List[str]:
        """
        Get all categories that contain a tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            List of category names
        """
        result = []
        for category_name, category in self.categories.items():
            if category.get_tool(tool_name):
                result.append(category_name)
        return result
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all tools from all categories.
        
        Returns:
            Dictionary of tool names and data
        """
        result = {}
        for category in self.categories.values():
            for tool_name, tool_data in category.tools.items():
                result[tool_name] = tool_data
        return result
