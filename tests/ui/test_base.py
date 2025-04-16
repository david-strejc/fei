"""
Tests for the UI base interfaces.
"""

import pytest
from unittest.mock import MagicMock

from fei.ui.base import ChatUIInterface, ConsentUIInterface, CommandUIInterface, UIInterface


class TestChatUIInterface:
    """Test the ChatUIInterface."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Check if abstract methods are defined
        assert hasattr(ChatUIInterface, "display_message")
        assert hasattr(ChatUIInterface, "get_input")
        assert hasattr(ChatUIInterface, "display_error")
        assert hasattr(ChatUIInterface, "display_thinking")
        assert hasattr(ChatUIInterface, "clear_thinking")
        
        # Check if methods are abstract
        with pytest.raises(TypeError):
            ChatUIInterface()


class TestConsentUIInterface:
    """Test the ConsentUIInterface."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Check if abstract methods are defined
        assert hasattr(ConsentUIInterface, "ask_consent")
        
        # Check if methods are abstract
        with pytest.raises(TypeError):
            ConsentUIInterface()


class TestCommandUIInterface:
    """Test the CommandUIInterface."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Check if abstract methods are defined
        assert hasattr(CommandUIInterface, "register_command")
        assert hasattr(CommandUIInterface, "execute_command")
        assert hasattr(CommandUIInterface, "display_help")
        
        # Check if methods are abstract
        with pytest.raises(TypeError):
            CommandUIInterface()


class TestUIInterface:
    """Test the UIInterface."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Check if abstract methods are defined
        assert hasattr(UIInterface, "setup")
        assert hasattr(UIInterface, "run")
        assert hasattr(UIInterface, "shutdown")
        
        # Check if properties are defined
        assert hasattr(UIInterface, "chat")
        assert hasattr(UIInterface, "consent")
        assert hasattr(UIInterface, "command")
        
        # Check if methods are abstract
        with pytest.raises(TypeError):
            UIInterface()


class MockChatUI(ChatUIInterface):
    """Mock implementation of ChatUIInterface."""
    
    def display_message(self, message, role="assistant"):
        pass
    
    def get_input(self, prompt=None):
        return "Mock input"
    
    def display_error(self, error):
        pass
    
    def display_thinking(self):
        pass
    
    def clear_thinking(self):
        pass


class MockConsentUI(ConsentUIInterface):
    """Mock implementation of ConsentUIInterface."""
    
    def ask_consent(self, message, details=None):
        return True


class MockCommandUI(CommandUIInterface):
    """Mock implementation of CommandUIInterface."""
    
    def register_command(self, name, handler, help_text):
        pass
    
    def execute_command(self, command, args):
        return None
    
    def display_help(self):
        pass


class MockUI(UIInterface):
    """Mock implementation of UIInterface."""
    
    def __init__(self):
        self._chat = MockChatUI()
        self._consent = MockConsentUI()
        self._command = MockCommandUI()
    
    def setup(self, config=None):
        pass
    
    def run(self):
        pass
    
    def shutdown(self):
        pass
    
    @property
    def chat(self):
        return self._chat
    
    @property
    def consent(self):
        return self._consent
    
    @property
    def command(self):
        return self._command


class TestMockImplementations:
    """Test the mock implementations."""
    
    def test_mock_chat_ui(self):
        """Test MockChatUI."""
        # Create mock chat UI
        chat_ui = MockChatUI()
        
        # Check if methods can be called
        chat_ui.display_message("Test message")
        assert chat_ui.get_input() == "Mock input"
        chat_ui.display_error("Test error")
        chat_ui.display_thinking()
        chat_ui.clear_thinking()
    
    def test_mock_consent_ui(self):
        """Test MockConsentUI."""
        # Create mock consent UI
        consent_ui = MockConsentUI()
        
        # Check if methods can be called
        assert consent_ui.ask_consent("Test message") is True
    
    def test_mock_command_ui(self):
        """Test MockCommandUI."""
        # Create mock command UI
        command_ui = MockCommandUI()
        
        # Check if methods can be called
        command_ui.register_command("test", lambda: None, "Test command")
        assert command_ui.execute_command("test", []) is None
        command_ui.display_help()
    
    def test_mock_ui(self):
        """Test MockUI."""
        # Create mock UI
        ui = MockUI()
        
        # Check if methods can be called
        ui.setup()
        ui.run()
        ui.shutdown()
        
        # Check if properties return correct interfaces
        assert isinstance(ui.chat, ChatUIInterface)
        assert isinstance(ui.consent, ConsentUIInterface)
        assert isinstance(ui.command, CommandUIInterface)
