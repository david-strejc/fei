"""
Tests for the UI system
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import io

from fei.ui.base import UIInterface, ChatUIInterface, ConsentUIInterface, CommandUIInterface
from fei.ui.enhanced_cli import CLIChatUI, CLIConsentUI, CLICommandUI, EnhancedCLI


class TestUISystem(unittest.TestCase):
    """Test the UI system"""
    
    def test_cli_chat_ui(self):
        """Test CLI chat UI"""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Create chat UI
            chat_ui = CLIChatUI()
            
            # Display messages
            chat_ui.display_message("Test message", "assistant")
            chat_ui.display_message("User message", "user")
            chat_ui.display_message("System message", "system")
            chat_ui.display_error("Test error")
            chat_ui.display_thinking()
            chat_ui.clear_thinking()
            
            # Check output
            output = captured_output.getvalue()
            self.assertIn("Fei:", output)
            self.assertIn("Test message", output)
            self.assertIn("You:", output)
            self.assertIn("User message", output)
            self.assertIn("System:", output)
            self.assertIn("System message", output)
            self.assertIn("Error:", output)
            self.assertIn("Test error", output)
            self.assertIn("Fei is thinking...", output)
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    @patch("builtins.input")
    def test_cli_consent_ui(self, mock_input):
        """Test CLI consent UI"""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Create consent UI
            consent_ui = CLIConsentUI()
            
            # Test consent granted
            mock_input.return_value = "yes"
            result = consent_ui.ask_consent("Do you want to proceed?")
            self.assertTrue(result)
            
            # Test consent denied
            mock_input.return_value = "no"
            result = consent_ui.ask_consent("Do you want to proceed?")
            self.assertFalse(result)
            
            # Test invalid input followed by valid input
            mock_input.side_effect = ["invalid", "yes"]
            result = consent_ui.ask_consent("Do you want to proceed?")
            self.assertTrue(result)
            
            # Test with details
            mock_input.return_value = "yes"
            result = consent_ui.ask_consent("Do you want to proceed?", {"action": "test", "risk": "low"})
            self.assertTrue(result)
            
            # Check output
            output = captured_output.getvalue()
            self.assertIn("MCP Consent Required", output)
            self.assertIn("Do you want to proceed?", output)
            self.assertIn("Details:", output)
            self.assertIn("action: test", output)
            self.assertIn("risk: low", output)
            self.assertIn("Consent granted.", output)
            self.assertIn("Consent denied.", output)
            self.assertIn("Please answer 'yes' or 'no'.", output)
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_cli_command_ui(self):
        """Test CLI command UI"""
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Create command UI
            command_ui = CLICommandUI()
            
            # Register commands
            test_handler = MagicMock(return_value="Test result")
            error_handler = MagicMock(side_effect=ValueError("Test error"))
            
            command_ui.register_command("test", test_handler, "Test command")
            command_ui.register_command("error", error_handler, "Error command")
            
            # Execute commands
            result = command_ui.execute_command("test", ["arg1", "arg2"])
            self.assertEqual(result, "Test result")
            test_handler.assert_called_once_with("arg1", "arg2")
            
            # Execute command with error
            result = command_ui.execute_command("error", [])
            self.assertIsNone(result)
            error_handler.assert_called_once_with()
            
            # Execute unknown command
            result = command_ui.execute_command("unknown", [])
            self.assertIsNone(result)
            
            # Display help
            command_ui.display_help()
            
            # Check output
            output = captured_output.getvalue()
            self.assertIn("Unknown command: unknown", output)
            self.assertIn("Error executing command:", output)
            self.assertIn("Test error", output)
            self.assertIn("Available commands:", output)
            self.assertIn("test:", output)
            self.assertIn("Test command", output)
            self.assertIn("error:", output)
            self.assertIn("Error command", output)
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_tools")
    @patch("fei.ui.enhanced_cli.EnhancedCLI._setup_assistant")
    @patch("fei.ui.enhanced_cli.EnhancedCLI._parse_args")
    def test_enhanced_cli(self, mock_parse_args, mock_setup_assistant, mock_setup_tools):
        """Test enhanced CLI"""
        # Mock dependencies
        mock_tool_registry = MagicMock()
        mock_setup_tools.return_value = mock_tool_registry
        
        mock_assistant = MagicMock()
        mock_setup_assistant.return_value = mock_assistant
        
        mock_args = MagicMock()
        mock_args.debug = False
        mock_args.message = None
        mock_args.task = None
        mock_parse_args.return_value = mock_args
        
        # Create enhanced CLI
        cli = EnhancedCLI()
        
        # Set up CLI
        cli.setup()
        
        # Check interfaces
        self.assertIsInstance(cli.chat, ChatUIInterface)
        self.assertIsInstance(cli.consent, ConsentUIInterface)
        self.assertIsInstance(cli.command, CommandUIInterface)
        
        # Test command registration
        self.assertIn("exit", cli._command_ui.commands)
        self.assertIn("quit", cli._command_ui.commands)
        self.assertIn("clear", cli._command_ui.commands)
        self.assertIn("history", cli._command_ui.commands)
        self.assertIn("help", cli._command_ui.commands)


if __name__ == "__main__":
    unittest.main()
