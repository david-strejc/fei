"""
Tests for the CLI UI.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import io

from fei.ui.enhanced_cli import CLIChatUI, CLIConsentUI, CLICommandUI, EnhancedCLI


class TestCLIChatUI:
    """Test the CLIChatUI."""
    
    def setup_method(self):
        """Set up the test."""
        self.chat_ui = CLIChatUI()
    
    def test_display_message(self):
        """Test display_message."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Display messages
            self.chat_ui.display_message("Test message", "assistant")
            self.chat_ui.display_message("User message", "user")
            self.chat_ui.display_message("System message", "system")
            self.chat_ui.display_message("Custom message", "custom")
            
            # Check output
            output = captured_output.getvalue()
            assert "Fei:" in output
            assert "Test message" in output
            assert "You:" in output
            assert "User message" in output
            assert "System:" in output
            assert "System message" in output
            assert "Custom:" in output
            assert "Custom message" in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    @patch("builtins.input")
    def test_get_input(self, mock_input):
        """Test get_input."""
        # Set up mock input
        mock_input.return_value = "Test input"
        
        # Get input
        result = self.chat_ui.get_input()
        
        # Check if input was returned
        assert result == "Test input"
    
    def test_display_error(self):
        """Test display_error."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Display error
            self.chat_ui.display_error("Test error")
            
            # Check output
            output = captured_output.getvalue()
            assert "Error:" in output
            assert "Test error" in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_display_thinking(self):
        """Test display_thinking."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Display thinking
            self.chat_ui.display_thinking()
            
            # Check output
            output = captured_output.getvalue()
            assert "Fei is thinking..." in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_clear_thinking(self):
        """Test clear_thinking."""
        # This method doesn't do anything in CLI
        self.chat_ui.clear_thinking()


class TestCLIConsentUI:
    """Test the CLIConsentUI."""
    
    def setup_method(self):
        """Set up the test."""
        self.consent_ui = CLIConsentUI()
    
    @patch("builtins.input")
    def test_ask_consent(self, mock_input):
        """Test ask_consent."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Test consent granted
            mock_input.return_value = "yes"
            result = self.consent_ui.ask_consent("Do you want to proceed?")
            assert result is True
            
            # Test consent denied
            mock_input.return_value = "no"
            result = self.consent_ui.ask_consent("Do you want to proceed?")
            assert result is False
            
            # Test invalid input followed by valid input
            mock_input.side_effect = ["invalid", "yes"]
            result = self.consent_ui.ask_consent("Do you want to proceed?")
            assert result is True
            
            # Test with details
            mock_input.return_value = "yes"
            result = self.consent_ui.ask_consent("Do you want to proceed?", {"action": "test", "risk": "low"})
            assert result is True
            
            # Check output
            output = captured_output.getvalue()
            assert "MCP Consent Required" in output
            assert "Do you want to proceed?" in output
            assert "Details:" in output
            assert "action: test" in output
            assert "risk: low" in output
            assert "Consent granted." in output
            assert "Consent denied." in output
            assert "Please answer 'yes' or 'no'." in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    @patch("builtins.input")
    def test_ask_consent_eof(self, mock_input):
        """Test ask_consent with EOF."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Test EOF
            mock_input.side_effect = EOFError()
            result = self.consent_ui.ask_consent("Do you want to proceed?")
            assert result is False
            
            # Check output
            output = captured_output.getvalue()
            assert "MCP Consent Required" in output
            assert "Do you want to proceed?" in output
            assert "Input stream closed. Denying consent." in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    @patch("builtins.input")
    def test_ask_consent_exception(self, mock_input):
        """Test ask_consent with exception."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Test exception
            mock_input.side_effect = ValueError("Test error")
            result = self.consent_ui.ask_consent("Do you want to proceed?")
            assert result is False
            
            # Check output
            output = captured_output.getvalue()
            assert "MCP Consent Required" in output
            assert "Do you want to proceed?" in output
            assert "An error occurred: Test error. Denying consent." in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__


class TestCLICommandUI:
    """Test the CLICommandUI."""
    
    def setup_method(self):
        """Set up the test."""
        self.command_ui = CLICommandUI()
    
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
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Execute a non-existent command
            result = self.command_ui.execute_command("nonexistent", [])
            
            # Check if result is None
            assert result is None
            
            # Check output
            output = captured_output.getvalue()
            assert "Unknown command: nonexistent" in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_execute_command_exception(self):
        """Test execute_command with exception."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Register a command that raises an exception
            handler = MagicMock(side_effect=ValueError("Test error"))
            self.command_ui.register_command("test", handler, "Test command")
            
            # Execute the command
            result = self.command_ui.execute_command("test", [])
            
            # Check if result is None
            assert result is None
            
            # Check output
            output = captured_output.getvalue()
            assert "Error executing command: Test error" in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_display_help(self):
        """Test display_help."""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Register some commands
            self.command_ui.register_command("test1", MagicMock(), "Test command 1")
            self.command_ui.register_command("test2", MagicMock(), "Test command 2")
            
            # Display help
            self.command_ui.display_help()
            
            # Check output
            output = captured_output.getvalue()
            assert "Available commands:" in output
            assert "test1:" in output
            assert "Test command 1" in output
            assert "test2:" in output
            assert "Test command 2" in output
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__


class TestEnhancedCLI:
    """Test the EnhancedCLI."""
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    def test_init(self, mock_setup_tools):
        """Test initialization."""
        # Create CLI
        cli = EnhancedCLI()
        
        # Check if CLI is initialized correctly
        assert isinstance(cli.config, object)
        assert cli.tool_registry is None
        assert cli.assistant is None
        assert hasattr(cli, "history")
        assert cli.history_file is None
        assert hasattr(cli, "loop")
        
        # Check if UI interfaces are created
        assert isinstance(cli._chat_ui, CLIChatUI)
        assert isinstance(cli._consent_ui, CLIConsentUI)
        assert isinstance(cli._command_ui, CLICommandUI)
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    def test_setup(self, mock_setup_tools):
        """Test setup."""
        # Set up mock tool registry
        mock_tool_registry = MagicMock()
        mock_setup_tools.return_value = mock_tool_registry
        
        # Create CLI
        cli = EnhancedCLI()
        
        # Set up CLI
        cli.setup({"test": "value"})
        
        # Check if CLI is set up correctly
        assert cli.config.get("test") == "value"
        assert cli.tool_registry is mock_tool_registry
        assert cli.history_file is not None
        
        # Check if commands are registered
        assert "exit" in cli._command_ui.commands
        assert "quit" in cli._command_ui.commands
        assert "clear" in cli._command_ui.commands
        assert "history" in cli._command_ui.commands
        assert "help" in cli._command_ui.commands
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_assistant")
    @patch("fei.ui.enhanced_cli.EnhancedCLI._parse_args")
    @patch("fei.ui.enhanced_cli.load_stage")
    def test_run(self, mock_load_stage, mock_parse_args, mock_setup_assistant, mock_setup_tools):
        """Test run."""
        # Set up mocks
        mock_tool_registry = MagicMock()
        mock_setup_tools.return_value = mock_tool_registry
        
        mock_assistant = MagicMock()
        mock_setup_assistant.return_value = mock_assistant
        
        mock_args = MagicMock()
        mock_args.debug = False
        mock_args.message = None
        mock_args.task = None
        mock_args.api_key = None
        mock_args.model = None
        mock_args.provider = None
        mock_parse_args.return_value = mock_args
        
        mock_load_stage.return_value = "stage_0"
        
        # Create CLI
        cli = EnhancedCLI()
        
        # Set up CLI
        cli.setup()
        
        # Mock _chat_loop
        cli._chat_loop = MagicMock()
        
        # Run CLI
        cli.loop.run_until_complete(cli.run())
        
        # Check if assistant was set up
        mock_setup_assistant.assert_called_once_with(None, None, None)
        
        # Check if evolution stage was loaded
        mock_load_stage.assert_called_once_with(mock_assistant, mock_tool_registry)
        
        # Check if chat loop was called
        cli._chat_loop.assert_called_once()
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    def test_chat_property(self, mock_setup_tools):
        """Test chat property."""
        # Create CLI
        cli = EnhancedCLI()
        
        # Check if chat property returns correct interface
        assert cli.chat is cli._chat_ui
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    def test_consent_property(self, mock_setup_tools):
        """Test consent property."""
        # Create CLI
        cli = EnhancedCLI()
        
        # Check if consent property returns correct interface
        assert cli.consent is cli._consent_ui
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    def test_command_property(self, mock_setup_tools):
        """Test command property."""
        # Create CLI
        cli = EnhancedCLI()
        
        # Check if command property returns correct interface
        assert cli.command is cli._command_ui
