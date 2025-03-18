#!/usr/bin/env python3
"""
Tests for LiteLLM integration in Fei
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from fei.core.assistant import Assistant
from fei.utils.config import Config

@pytest.fixture
def mock_litellm_response():
    """Create a mock response from LiteLLM"""
    mock_response = MagicMock()
    mock_response.id = "resp_12345"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "This is a test response"
    mock_response.tool_calls = None
    return mock_response

@pytest.fixture
def mock_litellm_response_with_tools():
    """Create a mock response from LiteLLM with tool calls"""
    mock_response = MagicMock()
    mock_response.id = "resp_12345"
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "I'll use a tool to help with this"
    
    tool_call = MagicMock()
    tool_call.id = "call_12345"
    tool_call.function = MagicMock()
    tool_call.function.name = "test_tool"
    tool_call.function.arguments = '{"param1": "value1"}'
    
    mock_response.tool_calls = [tool_call]
    return mock_response

@pytest.fixture
def assistant():
    """Create an assistant instance with a mock config"""
    config = Config()
    
    # Set a mock API key
    os.environ["ANTHROPIC_API_KEY"] = "sk-test-key"
    
    assistant = Assistant(config=config)
    return assistant

@patch("fei.core.assistant.litellm_completion")
async def test_chat_simple(mock_litellm, assistant, mock_litellm_response):
    """Test simple chat without tool calls"""
    # Configure the mock
    mock_litellm.return_value = mock_litellm_response
    
    # Call the chat method
    response = await assistant.chat("Hello, how are you?")
    
    # Verify the response
    assert response == "This is a test response"
    assert assistant.last_message_id == "resp_12345"
    assert len(assistant.conversation) == 2  # User message + assistant response
    
    # Verify litellm was called correctly
    mock_litellm.assert_called_once()
    call_args = mock_litellm.call_args[1]
    assert call_args["model"] == assistant.model
    assert call_args["max_tokens"] == 4000
    assert len(call_args["messages"]) == 1
    assert call_args["messages"][0]["role"] == "user"
    assert call_args["messages"][0]["content"] == "Hello, how are you?"

@patch("fei.core.assistant.litellm_completion")
async def test_chat_with_tools(mock_litellm, assistant, mock_litellm_response_with_tools, mock_litellm_response):
    """Test chat with tool calls"""
    # Configure the mock to return a response with tools first, then a normal response
    mock_litellm.side_effect = [mock_litellm_response_with_tools, mock_litellm_response]
    
    # Mock the tool execution
    async def mock_execute_tool(tool_name, tool_args):
        return {"result": "Tool execution result"}
    
    assistant.execute_tool = mock_execute_tool
    
    # Call the chat method
    response = await assistant.chat("Can you help me with a task?")
    
    # Verify the response (should be from the second call)
    assert response == "This is a test response"
    assert assistant.last_message_id == "resp_12345"
    assert len(assistant.conversation) == 4  # User, assistant, tool, assistant
    
    # Verify litellm was called twice
    assert mock_litellm.call_count == 2
    
    # Check first call arguments
    first_call_args = mock_litellm.call_args_list[0][1]
    assert first_call_args["model"] == assistant.model
    assert first_call_args["messages"][0]["role"] == "user"
    assert first_call_args["messages"][0]["content"] == "Can you help me with a task?"
    
    # Check second call arguments
    second_call_args = mock_litellm.call_args_list[1][1]
    assert second_call_args["model"] == assistant.model
    assert len(second_call_args["messages"]) == 3  # Original message, first response, tool results

@patch("fei.core.assistant.litellm_completion")
async def test_multi_provider_init(mock_litellm, mock_litellm_response):
    """Test initialization with different providers"""
    # Test OpenAI provider
    os.environ["OPENAI_API_KEY"] = "sk-test-openai"
    openai_assistant = Assistant(provider="openai")
    assert openai_assistant.provider == "openai"
    assert openai_assistant.model == "gpt-4o"
    
    # Test Groq provider
    os.environ["GROQ_API_KEY"] = "gsk-test-groq"
    groq_assistant = Assistant(provider="groq")
    assert groq_assistant.provider == "groq"
    assert groq_assistant.model == "llama-3-8b-8192"
    
    # Test custom provider and model
    os.environ["CUSTOM_API_KEY"] = "custom-test-key"
    custom_assistant = Assistant(provider="custom", model="custom-model")
    assert custom_assistant.provider == "custom"
    assert custom_assistant.model == "custom-model"

@patch("fei.core.assistant.litellm_completion")
async def test_chat_with_system_prompt(mock_litellm, assistant, mock_litellm_response):
    """Test chat with system prompt"""
    # Configure the mock
    mock_litellm.return_value = mock_litellm_response
    
    # Call the chat method with a system prompt
    system_prompt = "You are a helpful assistant."
    response = await assistant.chat("Hello", system_prompt=system_prompt)
    
    # Verify the response
    assert response == "This is a test response"
    
    # Verify litellm was called with the system prompt
    mock_litellm.assert_called_once()
    call_args = mock_litellm.call_args[1]
    assert call_args["system"] == system_prompt