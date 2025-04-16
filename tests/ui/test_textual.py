"""
Tests for the Textual UI.
"""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from fei.ui.enhanced_textual import TextualChatUI, TextualConsentUI, TextualCommandUI, EnhancedTextualApp


class TestTextualChatUI:
    """Test the TextualChatUI."""
    
    def setup_method(self):
        """Set up the test."""
        self.app = MagicMock()
        self.chat_ui = TextualChatUI(self.app)
    
    def test_display_message(self):
        """Test display_message."""
        # Display message
        self.chat_ui.display_message("Test message", "assistant")
        
        # Check if app.add_message was called with correct arguments
        self.app.add_message.assert_called_once()
        args, kwargs = self.app.add_message.call_args
        assert args[0] == "Test message"
        assert args[1] == "assistant"
    
    def test_get_input(self):
        """Test get_input."""
        # Get input
        result = self.chat_ui.get_input()
        
        # Check if result is empty string (not applicable in Textual UI)
        assert result == ""
    
    def test_display_error(self):
        """Test display_error."""
        # Display error
        self.chat_ui.display_error("Test error")
        
        # Check if app.add_message was called with correct arguments
        self.app.add_message.assert_called_once()
        args, kwargs = self.app.add_message.call_args
        assert args[0] == "Error: Test error"
        assert args[1] == "system"
    
    def test_display_thinking(self):
        """Test display_thinking."""
        # Display thinking
        self.chat_ui.display_thinking()
        
        # Check if app.is_loading was set to True
        assert self.app.is_loading is True
    
    def test_clear_thinking(self):
        """Test clear_thinking."""
        # Clear thinking
        self.chat_ui.clear_thinking()
        
        # Check if app.is_loading was set to False
        assert self.app.is_loading is False


class TestTextualConsentUI:
    """Test the TextualConsentUI."""
    
    def setup_method(self):
        """Set up the test."""
        self.app = MagicMock()
        self.consent_ui = TextualConsentUI(self.app)
    
    @patch("asyncio.Future")
    @patch("asyncio.get_event_loop")
    def test_ask_consent(self, mock_get_event_loop, mock_future):
        """Test ask_consent."""
        # Set up mock future
        future_instance = MagicMock()
        mock_future.return_value = future_instance
        
        # Set up mock event loop
        loop = MagicMock()
        loop.run_until_complete.return_value = True
        mock_get_event_loop.return_value = loop
        
        # Ask consent
        result = self.consent_ui.ask_consent("Do you want to proceed?", {"action": "test"})
        
        # Check if app.show_consent_dialog was called with correct arguments
        self.app.show_consent_dialog.assert_called_once()
        args, kwargs = self.app.show_consent_dialog.call_args
        assert args[0] == "Do you want to proceed?"
        assert args[1] == {"action": "test"}
        assert args[2] is future_instance
        
        # Check if loop.run_until_complete was called with future
        loop.run_until_complete.assert_called_once_with(future_instance)
        
        # Check if result is correct
        assert result is True
    
    @patch("asyncio.Future")
    @patch("asyncio.get_event_loop")
    def test_ask_consent_exception(self, mock_get_event_loop, mock_future):
        """Test ask_consent with exception."""
        # Set up mock future
        future_instance = MagicMock()
        mock_future.return_value = future_instance
        
        # Set up mock event loop to raise exception
        loop = MagicMock()
        loop.run_until_complete.side_effect = ValueError("Test error")
        mock_get_event_loop.return_value = loop
        
        # Ask consent
        result = self.consent_ui.ask_consent("Do you want to proceed?")
        
        # Check if result is False
        assert result is False


class TestTextualCommandUI:
    """Test the TextualCommandUI."""
    
    def setup_method(self):
        """Set up the test."""
        self.app = MagicMock()
        self.command_ui = TextualCommandUI(self.app)
    
    def test_register_command(self):
        """Test register_command."""
        # Register a command
        handler = MagicMock()
        self.command_ui.register_command("test", handler, "Test command")
        
        # Check if command is registered
        assert "test" in self.command_ui.commands
        assert self.command_ui.commands["test"]["handler"] is handler
        assert self.command_ui.commands["test"]["help"] == "Test command"
    
    def test_execute_command(self):
        """Test execute_command."""
        # Register a command
        handler = MagicMock(return_value="Test result")
        self.command_ui.register_command("test", handler, "Test command")
        
        # Execute the command
        result = self.command_ui.execute_command("test", ["arg1", "arg2"])
        
        # Check if handler was called with correct arguments
        handler.assert_called_once_with("arg1", "arg2")
        
        # Check if result is correct
        assert result == "Test result"
    
    def test_execute_nonexistent_command(self):
        """Test execute_command with non-existent command."""
        # Execute a non-existent command
        result = self.command_ui.execute_command("nonexistent", [])
        
        # Check if result is None
        assert result is None
        
        # Check if app.add_message was called with correct arguments
        self.app.add_message.assert_called_once()
        args, kwargs = self.app.add_message.call_args
        assert args[0] == "Unknown command: nonexistent"
        assert args[1] == "system"
    
    def test_execute_command_exception(self):
        """Test execute_command with exception."""
        # Register a command that raises an exception
        handler = MagicMock(side_effect=ValueError("Test error"))
        self.command_ui.register_command("test", handler, "Test command")
        
        # Execute the command
        result = self.command_ui.execute_command("test", [])
        
        # Check if result is None
        assert result is None
        
        # Check if app.add_message was called with correct arguments
        self.app.add_message.assert_called_once()
        args, kwargs = self.app.add_message.call_args
        assert args[0] == "Error executing command: Test error"
        assert args[1] == "system"
    
    def test_display_help(self):
        """Test display_help."""
        # Register some commands
        self.command_ui.register_command("test1", MagicMock(), "Test command 1")
        self.command_ui.register_command("test2", MagicMock(), "Test command 2")
        
        # Display help
        self.command_ui.display_help()
        
        # Check if app.add_message was called with correct arguments
        self.app.add_message.assert_called_once()
        args, kwargs = self.app.add_message.call_args
        assert "Available Commands" in args[0]
        assert "test1" in args[0]
        assert "Test command 1" in args[0]
        assert "test2" in args[0]
        assert "Test command 2" in args[0]
        assert args[1] == "system"


@pytest.mark.skip(reason="Requires Textual environment")
class TestEnhancedTextualApp:
    """Test the EnhancedTextualApp."""
    
    @patch("fei.ui.enhanced_textual.EnhancedTextualApp._setup_tools")
    def test_init(self, mock_setup_tools):
        """Test initialization."""
        # Create app
        app = EnhancedTextualApp()
        
        # Check if app is initialized correctly
        assert isinstance(app.config, object)
        assert app.tool_registry is None
        assert app.assistant is None
        
        # Check if UI interfaces are created
        assert isinstance(app._chat_ui, TextualChatUI)
        assert isinstance(app._consent_ui, TextualConsentUI)
        assert isinstance(app._command_ui, TextualCommandUI)
    
    @patch("fei.ui.enhanced_textual.EnhancedTextualApp._setup_tools")
    def test_setup(self, mock_setup_tools):
        """Test setup."""
        # Set up mock tool registry
        mock_tool_registry = MagicMock()
        mock_setup_tools.return_value = mock_tool_registry
        
        # Create app
        app = EnhancedTextualApp()
        
        # Set up app
        app.setup({"test": "value"})
        
        # Check if app is set up correctly
        assert app.config.get("test") == "value"
        assert app.tool_registry is mock_tool_registry
        
        # Check if commands are registered
        assert "clear" in app._command_ui.commands
        assert "help" in app._command_ui.commands
        assert "search" in app._command_ui.commands
    
    @patch("fei.ui.enhanced_textual.EnhancedTextualApp._setup_tools")
    def test_chat_property(self, mock_setup_tools):
        """Test chat property."""
        # Create app
        app = EnhancedTextualApp()
        
        # Check if chat property returns correct interface
        assert app.chat is app._chat_ui
    
    @patch("fei.ui.enhanced_textual.EnhancedTextualApp._setup_tools")
    def test_consent_property(self, mock_setup_tools):
        """Test consent property."""
        # Create app
        app = EnhancedTextualApp()
        
        # Check if consent property returns correct interface
        assert app.consent is app._consent_ui
    
    @patch("fei.ui.enhanced_textual.EnhancedTextualApp._setup_tools")
    def test_command_property(self, mock_setup_tools):
        """Test command property."""
        # Create app
        app = EnhancedTextualApp()
        
        # Check if command property returns correct interface
        assert app.command is app._command_ui
