"""
Tests for the provider manager.
"""

import pytest
from unittest.mock import MagicMock, patch
import os

from fei.providers.base import LLMProviderInterface, ProviderFactory
from fei.providers.manager import ProviderManager
from fei.utils.config import Config


class TestProviderManager:
    """Test the provider manager."""
    
    def setup_method(self):
        """Set up the test."""
        # Clear existing providers
        ProviderFactory._providers = {}
        
        # Register a mock provider
        self.mock_provider_class = MagicMock()
        self.mock_provider = MagicMock(spec=LLMProviderInterface)
        self.mock_provider_class.return_value = self.mock_provider
        ProviderFactory.register_provider("mock", self.mock_provider_class)
        
        # Create a config with mock provider
        self.config = Config()
        self.config.set("provider", "mock")
        self.config.set("model", "mock-model")
    
    def test_init(self):
        """Test initialization."""
        # Create manager
        manager = ProviderManager(self.config)
        
        # Check if default provider is set
        assert manager.default_provider == "mock"
    
    def test_get_provider(self):
        """Test getting a provider."""
        # Create manager
        manager = ProviderManager(self.config)
        
        # Get provider
        provider = manager.get_provider()
        
        # Check if correct provider is returned
        assert provider is self.mock_provider
        
        # Check if provider was created with correct arguments
        self.mock_provider_class.assert_called_once()
    
    def test_get_provider_with_api_key(self):
        """Test getting a provider with API key."""
        # Create manager
        manager = ProviderManager(self.config)
        
        # Get provider with API key
        provider = manager.get_provider(api_key="test-key")
        
        # Check if correct provider is returned
        assert provider is self.mock_provider
        
        # Check if provider was created with correct arguments
        self.mock_provider_class.assert_called_once()
        self.mock_provider.setup.assert_called_once()
    
    def test_get_provider_with_model(self):
        """Test getting a provider with model."""
        # Create manager
        manager = ProviderManager(self.config)
        
        # Get provider with model
        provider = manager.get_provider(model="test-model")
        
        # Check if correct provider is returned
        assert provider is self.mock_provider
        
        # Check if provider was created with correct arguments
        self.mock_provider_class.assert_called_once()
        self.mock_provider.setup.assert_called_once()
    
    def test_get_nonexistent_provider(self):
        """Test getting a non-existent provider."""
        # Create a config with non-existent provider
        config = Config()
        config.set("provider", "nonexistent")
        
        # Create manager
        manager = ProviderManager(config)
        
        # Try to get non-existent provider
        with pytest.raises(ValueError):
            manager.get_provider()
    
    def test_list_providers(self):
        """Test listing providers."""
        # Register another provider
        ProviderFactory.register_provider("mock2", MagicMock())
        
        # Create manager
        manager = ProviderManager(self.config)
        
        # List providers
        providers = manager.list_providers()
        
        # Check if all providers are listed
        assert "mock" in providers
        assert "mock2" in providers
        assert len(providers) == 2
    
    def test_get_provider_from_env(self):
        """Test getting a provider from environment variables."""
        # Create a config without provider
        config = Config()
        
        # Create manager
        manager = ProviderManager(config)
        
        # Mock environment variables
        with patch.dict(os.environ, {"FEI_PROVIDER": "mock"}):
            # Get provider
            provider = manager.get_provider()
            
            # Check if correct provider is returned
            assert provider is self.mock_provider
