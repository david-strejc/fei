"""
Tests for the provider abstraction layer
"""

import unittest
from unittest.mock import MagicMock, patch
import os

from fei.providers.base import LLMProviderInterface, ProviderFactory
from fei.providers.litellm_provider import LiteLLMProvider
from fei.providers.manager import ProviderManager
from fei.utils.config import Config


class TestProviderAbstraction(unittest.TestCase):
    """Test the provider abstraction layer"""
    
    def test_provider_factory_registration(self):
        """Test provider factory registration"""
        # Clear existing providers
        ProviderFactory._providers = {}
        
        # Register a mock provider
        class MockProvider(LLMProviderInterface):
            def setup(self, api_key=None, model=None):
                pass
            
            def get_default_model(self):
                return "mock-model"
            
            def complete(self, messages, tools=None, temperature=0.7, max_tokens=None, stream=False):
                return {"choices": [{"message": {"content": "Mock response"}}]}
            
            def extract_response_content(self, response):
                return "Mock content"
            
            def extract_tool_calls(self, response):
                return []
        
        ProviderFactory.register_provider("mock", MockProvider)
        
        # Check if provider is registered
        self.assertIn("mock", ProviderFactory.list_providers())
        
        # Create provider instance
        provider = ProviderFactory.create_provider("mock")
        self.assertIsInstance(provider, MockProvider)
        
        # Test default model
        self.assertEqual(provider.get_default_model(), "mock-model")
    
    @patch("litellm.completion")
    def test_litellm_provider(self, mock_completion):
        """Test LiteLLM provider"""
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
        
        # Set up provider with mock API key
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            provider.setup(api_key="test-key", model="claude-3-opus-20240229")
        
        # Test complete method
        messages = [{"role": "user", "content": "Hello"}]
        response = provider.complete(messages)
        
        # Check if litellm.completion was called with correct arguments
        mock_completion.assert_called_once()
        args, kwargs = mock_completion.call_args
        self.assertEqual(kwargs["model"], "anthropic/claude-3-opus-20240229")
        self.assertEqual(kwargs["messages"], messages)
        
        # Test extract_response_content
        content = provider.extract_response_content(response)
        self.assertEqual(content, "Test response")
    
    def test_provider_manager(self):
        """Test provider manager"""
        # Clear existing providers
        ProviderFactory._providers = {}
        
        # Register a mock provider
        class MockProvider(LLMProviderInterface):
            def setup(self, api_key=None, model=None):
                self.api_key = api_key
                self.model = model
            
            def get_default_model(self):
                return "mock-model"
            
            def complete(self, messages, tools=None, temperature=0.7, max_tokens=None, stream=False):
                return {"choices": [{"message": {"content": "Mock response"}}]}
            
            def extract_response_content(self, response):
                return "Mock content"
            
            def extract_tool_calls(self, response):
                return []
        
        ProviderFactory.register_provider("mock", MockProvider)
        
        # Create config with mock provider
        config = Config()
        config.set("provider", "mock")
        
        # Create provider manager
        manager = ProviderManager(config)
        
        # Get provider
        provider = manager.get_provider()
        self.assertIsInstance(provider, MockProvider)
        
        # List providers
        providers = manager.list_providers()
        self.assertIn("mock", providers)


if __name__ == "__main__":
    unittest.main()
