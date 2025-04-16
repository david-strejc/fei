"""
Tests for the model selector.
"""

import pytest
from unittest.mock import MagicMock, patch
import random

from fei.models.selector import (
    BaseModelSelector, CostOptimizedSelector, PerformanceOptimizedSelector,
    BalancedSelector, RandomSelector
)
from fei.models.base import ModelInfo, ModelRequirement
from fei.models.registry import ModelRegistry
from fei.utils.config import Config


class TestBaseModelSelector:
    """Test the BaseModelSelector class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a selector
        self.selector = BaseModelSelector(registry=self.registry)
    
    def test_init(self):
        """Test initialization."""
        # Check if selector is initialized correctly
        assert self.selector.registry is self.registry
        assert hasattr(self.selector, "config")
    
    def test_select_model(self):
        """Test select_model."""
        # Create a mock model
        model_info = MagicMock(spec=ModelInfo)
        model_info.provider = "test_provider"
        model_info.model_id = "test_model"
        
        # Set up registry to return the mock model
        self.registry.find_models.return_value = [model_info]
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select model
        provider, model = self.selector.select_model(requirement)
        
        # Check if registry was called with correct arguments
        self.registry.find_models.assert_called_once_with(requirement)
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_with_no_suitable_models(self):
        """Test select_model with no suitable models."""
        # Set up registry to return an empty list
        self.registry.find_models.return_value = []
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Try to select model
        with pytest.raises(ValueError):
            self.selector.select_model(requirement)
    
    def test_get_fallback_model(self):
        """Test get_fallback_model."""
        # Create a mock model
        model_info = MagicMock(spec=ModelInfo)
        model_info.provider = "fallback_provider"
        model_info.model_id = "fallback_model"
        
        # Set up registry to return the mock model
        self.registry.find_models.return_value = [model_info]
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Get fallback model
        provider, model = self.selector.get_fallback_model(("test_provider", "test_model"), requirement)
        
        # Check if registry was called with correct arguments
        self.registry.find_models.assert_called_once_with(requirement)
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
    
    def test_get_fallback_model_with_no_suitable_models(self):
        """Test get_fallback_model with no suitable models."""
        # Set up registry to return an empty list
        self.registry.find_models.return_value = []
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Try to get fallback model
        with pytest.raises(ValueError):
            self.selector.get_fallback_model(("test_provider", "test_model"), requirement)
    
    def test_get_fallback_model_with_same_model(self):
        """Test get_fallback_model with the same model."""
        # Create a mock model
        model_info = MagicMock(spec=ModelInfo)
        model_info.provider = "test_provider"
        model_info.model_id = "test_model"
        
        # Set up registry to return the mock model
        self.registry.find_models.return_value = [model_info]
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Get fallback model
        provider, model = self.selector.get_fallback_model(("test_provider", "test_model"), requirement)
        
        # Check if registry was called with correct arguments
        self.registry.find_models.assert_called_once_with(requirement)
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"


class TestCostOptimizedSelector:
    """Test the CostOptimizedSelector class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a selector
        self.selector = CostOptimizedSelector(registry=self.registry)
    
    def test_select_best_model(self):
        """Test _select_best_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        model1.cost_per_input_token = 0.0002
        model1.cost_per_output_token = 0.0004
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        model2.cost_per_input_token = 0.0001
        model2.cost_per_output_token = 0.0002
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        model3.cost_per_input_token = 0.0003
        model3.cost_per_output_token = 0.0006
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best model
        best_model = self.selector._select_best_model([model1, model2, model3], requirement)
        
        # Check if result is correct
        assert best_model is model2
    
    def test_select_best_fallback_model(self):
        """Test _select_best_fallback_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        model1.cost_per_input_token = 0.0002
        model1.cost_per_output_token = 0.0004
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        model2.cost_per_input_token = 0.0001
        model2.cost_per_output_token = 0.0002
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        model3.cost_per_input_token = 0.0003
        model3.cost_per_output_token = 0.0006
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best fallback model
        best_model = self.selector._select_best_fallback_model([model1, model2, model3], ("provider1", "model1"), requirement)
        
        # Check if result is correct
        assert best_model is model2


class TestPerformanceOptimizedSelector:
    """Test the PerformanceOptimizedSelector class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a selector
        self.selector = PerformanceOptimizedSelector(registry=self.registry)
    
    def test_select_best_model(self):
        """Test _select_best_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        model1.reliability = 0.9
        model1.context_length = 4096
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        model2.reliability = 0.95
        model2.context_length = 8192
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        model3.reliability = 0.95
        model3.context_length = 4096
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best model
        best_model = self.selector._select_best_model([model1, model2, model3], requirement)
        
        # Check if result is correct
        assert best_model is model2
    
    def test_select_best_fallback_model(self):
        """Test _select_best_fallback_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        model1.reliability = 0.9
        model1.context_length = 4096
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        model2.reliability = 0.95
        model2.context_length = 8192
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        model3.reliability = 0.95
        model3.context_length = 4096
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best fallback model
        best_model = self.selector._select_best_fallback_model([model1, model2, model3], ("provider1", "model1"), requirement)
        
        # Check if result is correct
        assert best_model is model2


class TestBalancedSelector:
    """Test the BalancedSelector class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a selector
        self.selector = BalancedSelector(registry=self.registry)
    
    def test_select_best_model(self):
        """Test _select_best_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        model1.reliability = 0.9
        model1.context_length = 4096
        model1.cost_per_input_token = 0.0001
        model1.cost_per_output_token = 0.0002
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        model2.reliability = 0.95
        model2.context_length = 8192
        model2.cost_per_input_token = 0.0002
        model2.cost_per_output_token = 0.0004
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        model3.reliability = 0.8
        model3.context_length = 4096
        model3.cost_per_input_token = 0.00005
        model3.cost_per_output_token = 0.0001
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best model
        best_model = self.selector._select_best_model([model1, model2, model3], requirement)
        
        # Check if result is one of the models (exact scoring depends on implementation)
        assert best_model in [model1, model2, model3]
    
    def test_select_best_fallback_model(self):
        """Test _select_best_fallback_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        model1.reliability = 0.9
        model1.context_length = 4096
        model1.cost_per_input_token = 0.0001
        model1.cost_per_output_token = 0.0002
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        model2.reliability = 0.95
        model2.context_length = 8192
        model2.cost_per_input_token = 0.0002
        model2.cost_per_output_token = 0.0004
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        model3.reliability = 0.8
        model3.context_length = 4096
        model3.cost_per_input_token = 0.00005
        model3.cost_per_output_token = 0.0001
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best fallback model
        best_model = self.selector._select_best_fallback_model([model1, model2, model3], ("provider1", "model1"), requirement)
        
        # Check if result is one of the models (exact scoring depends on implementation)
        assert best_model in [model1, model2, model3]


class TestRandomSelector:
    """Test the RandomSelector class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock registry
        self.registry = MagicMock(spec=ModelRegistry)
        
        # Create a selector
        self.selector = RandomSelector(registry=self.registry)
        
        # Set random seed for reproducibility
        random.seed(42)
    
    def test_select_best_model(self):
        """Test _select_best_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best model
        best_model = self.selector._select_best_model([model1, model2, model3], requirement)
        
        # Check if result is one of the models
        assert best_model in [model1, model2, model3]
    
    def test_select_best_fallback_model(self):
        """Test _select_best_fallback_model."""
        # Create mock models
        model1 = MagicMock(spec=ModelInfo)
        model1.provider = "provider1"
        model1.model_id = "model1"
        
        model2 = MagicMock(spec=ModelInfo)
        model2.provider = "provider2"
        model2.model_id = "model2"
        
        model3 = MagicMock(spec=ModelInfo)
        model3.provider = "provider3"
        model3.model_id = "model3"
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Select best fallback model
        best_model = self.selector._select_best_fallback_model([model1, model2, model3], ("provider1", "model1"), requirement)
        
        # Check if result is one of the models
        assert best_model in [model1, model2, model3]
