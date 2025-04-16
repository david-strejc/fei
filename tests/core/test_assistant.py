"""
Tests for the assistant core.
"""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from fei.core.assistant import Assistant
from fei.utils.config import Config


class TestAssistant:
    """Test the Assistant class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create mock tool registry
        self.tool_registry = MagicMock()
        self.tool_registry.get_tools.return_value = []
        
        # Create mock provider
        self.provider = MagicMock()
        self.provider.complete.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "role": "assistant"
                    }
                }
            ]
        }
        self.provider.extract_response_content.return_value = "Test response"
        self.provider.extract_tool_calls.return_value = []
        
        # Create mock provider manager
        self.provider_manager = MagicMock()
        self.provider_manager.get_provider.return_value = self.provider
        
        # Create mock consent UI
        self.consent_ui = MagicMock(return_value=True)
        
        # Create assistant
        with patch("fei.core.assistant.ProviderManager", return_value=self.provider_manager):
            self.assistant = Assistant(
                config=Config(),
                api_key="test-key",
                model="test-model",
                provider="test-provider",
                tool_registry=self.tool_registry,
                consent_ui_provider=self.consent_ui
            )
    
    def test_init(self):
        """Test initialization."""
        # Check if assistant is initialized correctly
        assert self.assistant.api_key == "test-key"
        assert self.assistant.model == "test-model"
        assert self.assistant.provider == "test-provider"
        assert self.assistant.tool_registry is self.tool_registry
        assert self.assistant.consent_ui_provider is self.consent_ui
        assert self.assistant.conversation == []
    
    @pytest.mark.asyncio
    async def test_chat(self):
        """Test chat."""
        # Chat with assistant
        response = await self.assistant.chat("Hello")
        
        # Check if provider was called with correct arguments
        self.provider.complete.assert_called_once()
        args, kwargs = self.provider.complete.call_args
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        
        # Check if response is correct
        assert response == "Test response"
        
        # Check if conversation is updated
        assert len(self.assistant.conversation) == 2
        assert self.assistant.conversation[0]["role"] == "user"
        assert self.assistant.conversation[0]["content"] == "Hello"
        assert self.assistant.conversation[1]["role"] == "assistant"
        assert self.assistant.conversation[1]["content"] == "Test response"
    
    @pytest.mark.asyncio
    async def test_chat_with_tool_calls(self):
        """Test chat with tool calls."""
        # Set up mock provider to return tool calls
        self.provider.extract_tool_calls.return_value = [
            {
                "name": "test_tool",
                "arguments": {"input": "test"}
            }
        ]
        
        # Set up mock tool registry to execute tool
        self.tool_registry.execute_tool.return_value = {"result": "test"}
        
        # Chat with assistant
        response = await self.assistant.chat("Use the test tool")
        
        # Check if provider was called with correct arguments
        assert self.provider.complete.call_count == 2
        
        # First call should be to get tool calls
        args1, kwargs1 = self.provider.complete.call_args_list[0]
        assert kwargs1["messages"] == [{"role": "user", "content": "Use the test tool"}]
        
        # Second call should be to get final response with tool results
        args2, kwargs2 = self.provider.complete.call_args_list[1]
        assert len(kwargs2["messages"]) == 3
        assert kwargs2["messages"][0]["role"] == "user"
        assert kwargs2["messages"][0]["content"] == "Use the test tool"
        assert kwargs2["messages"][1]["role"] == "assistant"
        assert "tool_calls" in kwargs2["messages"][1]
        assert kwargs2["messages"][2]["role"] == "tool"
        assert kwargs2["messages"][2]["tool_call_id"] is not None
        assert kwargs2["messages"][2]["content"] == '{"result": "test"}'
        
        # Check if tool was executed
        self.tool_registry.execute_tool.assert_called_once_with("test_tool", {"input": "test"})
        
        # Check if response is correct
        assert response == "Test response"
    
    @pytest.mark.asyncio
    async def test_chat_with_tool_calls_denied_consent(self):
        """Test chat with tool calls but consent denied."""
        # Set up mock provider to return tool calls
        self.provider.extract_tool_calls.return_value = [
            {
                "name": "test_tool",
                "arguments": {"input": "test"}
            }
        ]
        
        # Set up mock consent UI to deny consent
        self.assistant.consent_ui_provider = MagicMock(return_value=False)
        
        # Chat with assistant
        response = await self.assistant.chat("Use the test tool")
        
        # Check if provider was called with correct arguments
        assert self.provider.complete.call_count == 2
        
        # Check if tool was not executed
        self.tool_registry.execute_tool.assert_not_called()
        
        # Check if response is correct
        assert response == "Test response"
    
    def test_reset_conversation(self):
        """Test reset_conversation."""
        # Add some messages to conversation
        self.assistant.conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Reset conversation
        self.assistant.reset_conversation()
        
        # Check if conversation is empty
        assert self.assistant.conversation == []
    
    def test_get_conversation(self):
        """Test get_conversation."""
        # Add some messages to conversation
        self.assistant.conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
        
        # Get conversation
        conversation = self.assistant.get_conversation()
        
        # Check if conversation is correct
        assert conversation == [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]
    
    def test_set_conversation(self):
        """Test set_conversation."""
        # Set conversation
        new_conversation = [
            {"role": "user", "content": "New message"},
            {"role": "assistant", "content": "New response"}
        ]
        self.assistant.set_conversation(new_conversation)
        
        # Check if conversation is updated
        assert self.assistant.conversation == new_conversation
