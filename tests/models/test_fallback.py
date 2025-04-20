"""
Tests for the fallback strategies.
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from fei.models.fallback import (
    FallbackStrategy, RetryStrategy, ProviderFallbackStrategy, ChainedFallbackStrategy
)
from fei.models.base import ModelRequirement, ModelInfo
from fei.models.selector import BaseModelSelector
from fei.utils.config import Config


class TestFallbackStrategy:
    """Test the FallbackStrategy class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock selector
        self.selector = MagicMock(spec=BaseModelSelector)
        self.selector.get_fallback_model.return_value = ("fallback_provider", "fallback_model")
        
        # Create a strategy
        self.strategy = FallbackStrategy(self.selector)
    
    def test_init(self):
        """Test initialization."""
        # Check if strategy is initialized correctly
        assert self.strategy.selector is self.selector
        assert hasattr(self.strategy, "config")
    
    def test_handle_failure(self):
        """Test handle_failure."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if selector was called with correct arguments
        self.selector.get_fallback_model.assert_called_once_with(("test_provider", "test_model"), requirement, None)
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
    
    def test_handle_failure_with_error(self):
        """Test handle_failure with error."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Create an error
        error = ValueError("Test error")
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement, error=error)
        
        # Check if selector was called with correct arguments
        self.selector.get_fallback_model.assert_called_once_with(("test_provider", "test_model"), requirement, None)
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
    
    def test_handle_failure_with_context(self):
        """Test handle_failure with context."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Create a context
        context = {"test_key": "test_value"}
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement, context=context)
        
        # Check if selector was called with correct arguments
        self.selector.get_fallback_model.assert_called_once_with(("test_provider", "test_model"), requirement, context)
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"


class TestRetryStrategy:
    """Test the RetryStrategy class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock selector
        self.selector = MagicMock(spec=BaseModelSelector)
        self.selector.get_fallback_model.return_value = ("fallback_provider", "fallback_model")
        
        # Create a strategy
        self.strategy = RetryStrategy(self.selector, max_retries=2, retry_delay=0.01)
    
    def test_init(self):
        """Test initialization."""
        # Check if strategy is initialized correctly
        assert self.strategy.selector is self.selector
        assert hasattr(self.strategy, "config")
        assert self.strategy.max_retries == 2
        assert self.strategy.retry_delay == 0.01
        assert self.strategy.retry_counts == {}
    
    def test_handle_failure_with_retry(self):
        """Test handle_failure with retry."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if selector was not called
        self.selector.get_fallback_model.assert_not_called()
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
        
        # Check if retry count is incremented
        assert self.strategy.retry_counts[("test_provider", "test_model")] == 1
    
    def test_handle_failure_with_max_retries_exceeded(self):
        """Test handle_failure with max retries exceeded."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Set retry count to max
        self.strategy.retry_counts[("test_provider", "test_model")] = 2
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if selector was called with correct arguments
        self.selector.get_fallback_model.assert_called_once_with(("test_provider", "test_model"), requirement, None)
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
        
        # Check if retry count is reset
        assert self.strategy.retry_counts[("test_provider", "test_model")] == 0


class TestProviderFallbackStrategy:
    """Test the ProviderFallbackStrategy class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock selector
        self.selector = MagicMock(spec=BaseModelSelector)
        
        # Create a strategy
        self.strategy = ProviderFallbackStrategy(self.selector)
    
    def test_init(self):
        """Test initialization."""
        # Check if strategy is initialized correctly
        assert self.strategy.selector is self.selector
        assert hasattr(self.strategy, "config")
    
    def test_handle_failure_with_different_provider(self):
        """Test handle_failure with a different provider."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "test_provider"
        model1.model_id = "test_model"
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "other_provider"
        model2.model_id = "other_model"
        
        # Set up selector to return models from different providers
        self.selector.registry.find_models.return_value = [model1, model2]
        self.selector._select_best_model.return_value = model2
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if selector was called with correct arguments
        self.selector.registry.find_models.assert_called_once()
        self.selector._select_best_model.assert_called_once()
        
        # Check if result is correct
        assert provider == "other_provider"
        assert model == "other_model"
    
    def test_handle_failure_with_no_different_provider(self):
        """Test handle_failure with no different provider."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "test_provider"
        model1.model_id = "test_model"
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "test_provider"
        model2.model_id = "other_model"
        
        # Set up selector to return models from the same provider
        self.selector.registry.find_models.return_value = [model1, model2]
        self.selector.get_fallback_model.return_value = ("fallback_provider", "fallback_model")
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if selector was called with correct arguments
        self.selector.registry.find_models.assert_called_once()
        self.selector.get_fallback_model.assert_called_once()
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"


class TestChainedFallbackStrategy:
    """Test the ChainedFallbackStrategy class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock selector
        self.selector = MagicMock(spec=BaseModelSelector)
        
        # Create mock strategies
        self.strategy1 = MagicMock(spec=FallbackStrategy)
        self.strategy1.handle_failure.return_value = ("provider1", "model1")
        
        self.strategy2 = MagicMock(spec=FallbackStrategy)
        self.strategy2.handle_failure.return_value = ("provider2", "model2")
        
        self.strategy3 = MagicMock(spec=FallbackStrategy)
        self.strategy3.handle_failure.return_value = ("provider3", "model3")
        
        # Create a chained strategy
        self.strategy = ChainedFallbackStrategy(self.selector, [self.strategy1, self.strategy2, self.strategy3])
    
    def test_init(self):
        """Test initialization."""
        # Check if strategy is initialized correctly
        assert self.strategy.selector is self.selector
        assert hasattr(self.strategy, "config")
        assert self.strategy.strategies == [self.strategy1, self.strategy2, self.strategy3]
    
    def test_handle_failure_with_first_strategy_success(self):
        """Test handle_failure with first strategy success."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if strategies were called with correct arguments
        self.strategy1.handle_failure.assert_called_once_with(("test_provider", "test_model"), requirement, None, None)
        self.strategy2.handle_failure.assert_not_called()
        self.strategy3.handle_failure.assert_not_called()
        
        # Check if result is correct
        assert provider == "provider1"
        assert model == "model1"
    
    def test_handle_failure_with_first_strategy_failure(self):
        """Test handle_failure with first strategy failure."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Set up first strategy to fail
        self.strategy1.handle_failure.side_effect = ValueError("Test error")
        
        # Handle failure
        provider, model = self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if strategies were called with correct arguments
        self.strategy1.handle_failure.assert_called_once_with(("test_provider", "test_model"), requirement, None, None)
        self.strategy2.handle_failure.assert_called_once_with(("test_provider", "test_model"), requirement, None, ValueError("Test error"))
        self.strategy3.handle_failure.assert_not_called()
        
        # Check if result is correct
        assert provider == "provider2"
        assert model == "model2"
    
    def test_handle_failure_with_all_strategies_failure(self):
        """Test handle_failure with all strategies failure."""
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Set up all strategies to fail
        self.strategy1.handle_failure.side_effect = ValueError("Error 1")
        self.strategy2.handle_failure.side_effect = ValueError("Error 2")
        self.strategy3.handle_failure.side_effect = ValueError("Error 3")
        
        # Try to handle failure
        with pytest.raises(ValueError):
            self.strategy.handle_failure(("test_provider", "test_model"), requirement)
        
        # Check if strategies were called with correct arguments
        self.strategy1.handle_failure.assert_called_once_with(("test_provider", "test_model"), requirement, None, None)
        self.strategy2.handle_failure.assert_called_once_with(("test_provider", "test_model"), requirement, None, ValueError("Error 1"))
        self.strategy3.handle_failure.assert_called_once_with(("test_provider", "test_model"), requirement, None, ValueError("Error 2"))
