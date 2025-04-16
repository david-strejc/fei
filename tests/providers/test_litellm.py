"""
Tests for the LiteLLM provider.
"""

import pytest
from unittest.mock import MagicMock, patch
import os

from fei.providers.litellm_provider import LiteLLMProvider


class TestLiteLLMProvider:
    """Test the LiteLLM provider."""
    
    def test_init(self):
        """Test initialization."""
        # Create provider
        provider = LiteLLMProvider()
        
        # Check if provider is initialized correctly
        assert provider.api_key is None
        assert provider.model is None
    
    @patch("litellm.completion")
    def test_setup(self, mock_completion):
        """Test setup."""
        # Create provider
        provider = LiteLLMProvider()
        
        # Set up provider
        provider.setup(api_key="test-key", model="claude-3-opus-20240229")
        
        # Check if provider is set up correctly
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-opus-20240229"
    
    @patch("litellm.completion")
    def test_setup_with_env(self, mock_completion):
        """Test setup with environment variables."""
        # Create provider
        provider = LiteLLMProvider()
        
        # Mock environment variables
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            # Set up provider without API key
            provider.setup(model="claude-3-opus-20240229")
            
            # Check if provider is set up correctly
            assert provider.api_key == "env-key"
            assert provider.model == "claude-3-opus-20240229"
    
    @patch("litellm.completion")
    def test_get_default_model(self, mock_completion):
        """Test getting default model."""
        # Create provider
        provider = LiteLLMProvider()
        
        # Get default model
        model = provider.get_default_model()
        
        # Check if default model is correct
        assert model == "claude-3-opus-20240229"
    
    @patch("litellm.completion")
    def test_complete(self, mock_completion):
        """Test completion."""
        # Set up mock response
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "role": "assistant"
                    }
                }
            ]
        }
        
        # Create provider
        provider = LiteLLMProvider()
        provider.setup(api_key="test-key", model="claude-3-opus-20240229")
        
        # Call complete
        messages = [{"role": "user", "content": "Hello"}]
        response = provider.complete(messages)
        
        # Check if litellm.completion was called with correct arguments
        mock_completion.assert_called_once()
        args, kwargs = mock_completion.call_args
        assert kwargs["model"] == "anthropic/claude-3-opus-20240229"
        assert kwargs["messages"] == messages
        
        # Check if response is correct
        assert response == mock_completion.return_value
    
    @patch("litellm.completion")
    def test_complete_with_tools(self, mock_completion):
        """Test completion with tools."""
        # Set up mock response
        mock_completion.return_value = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "test_tool",
                                    "arguments": '{"arg1": "value1"}'
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        # Create provider
        provider = LiteLLMProvider()
        provider.setup(api_key="test-key", model="claude-3-opus-20240229")
        
        # Create tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "Test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arg1": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
        # Call complete with tools
        messages = [{"role": "user", "content": "Use the test tool"}]
        response = provider.complete(messages, tools=tools)
        
        # Check if litellm.completion was called with correct arguments
        mock_completion.assert_called_once()
        args, kwargs = mock_completion.call_args
        assert kwargs["model"] == "anthropic/claude-3-opus-20240229"
        assert kwargs["messages"] == messages
        assert kwargs["tools"] == tools
        
        # Check if response is correct
        assert response == mock_completion.return_value
    
    @patch("litellm.completion")
    def test_extract_response_content(self, mock_completion):
        """Test extracting response content."""
        # Create provider
        provider = LiteLLMProvider()
        
        # Create response
        response = {
            "choices": [
                {
                    "message": {
                        "content": "Test response",
                        "role": "assistant"
                    }
                }
            ]
        }
        
        # Extract content
        content = provider.extract_response_content(response)
        
        # Check if content is correct
        assert content == "Test response"
    
    @patch("litellm.completion")
    def test_extract_tool_calls(self, mock_completion):
        """Test extracting tool calls."""
        # Create provider
        provider = LiteLLMProvider()
        
        # Create response with tool calls
        response = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "test_tool",
                                    "arguments": '{"arg1": "value1"}'
                                }
                            }
                        ]
                    }
                }
            ]
        }
        
        # Extract tool calls
        tool_calls = provider.extract_tool_calls(response)
        
        # Check if tool calls are correct
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "test_tool"
        assert tool_calls[0]["arguments"] == {"arg1": "value1"}
