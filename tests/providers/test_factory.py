"""
Tests for the provider factory.
"""

import pytest
from unittest.mock import MagicMock

from fei.providers.base import LLMProviderInterface, ProviderFactory


class TestProviderFactory:
    """Test the provider factory."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing providers
        ProviderFactory._providers = {}
    
    def test_register_provider(self):
        """Test registering a provider."""
        # Create a mock provider class
        mock_provider_class = MagicMock(spec=LLMProviderInterface)
        
        # Register the provider
        ProviderFactory.register_provider("mock", mock_provider_class)
        
        # Check if provider is registered
        assert "mock" in ProviderFactory.list_providers()
    
    def test_create_provider(self):
        """Test creating a provider."""
        # Create a mock provider class
        mock_provider_class = MagicMock(return_value=MagicMock(spec=LLMProviderInterface))
        
        # Register the provider
        ProviderFactory.register_provider("mock", mock_provider_class)
        
        # Create a provider instance
        provider = ProviderFactory.create_provider("mock", api_key="test-key")
        
        # Check if provider was created with correct arguments
        mock_provider_class.assert_called_once_with(api_key="test-key")
        assert provider is mock_provider_class.return_value
    
    def test_create_nonexistent_provider(self):
        """Test creating a non-existent provider."""
        # Try to create a non-existent provider
        with pytest.raises(ValueError):
            ProviderFactory.create_provider("nonexistent")
    
    def test_list_providers(self):
        """Test listing providers."""
        # Register some providers
        ProviderFactory.register_provider("mock1", MagicMock())
        ProviderFactory.register_provider("mock2", MagicMock())
        
        # List providers
        providers = ProviderFactory.list_providers()
        
        # Check if all providers are listed
        assert "mock1" in providers
        assert "mock2" in providers
        assert len(providers) == 2
