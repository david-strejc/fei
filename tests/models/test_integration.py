"""
Tests for the model switching integration.
"""

import pytest
from unittest.mock import MagicMock, patch

from fei.models.integration import ModelSwitcher
from fei.models.base import ModelCapability, ModelRequirement, ModelInfo
from fei.models.selector import BaseModelSelector
from fei.models.fallback import FallbackStrategy
from fei.utils.config import Config


class TestModelSwitcher:
    """Test the ModelSwitcher class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create mock selector and fallback strategy
        self.selector = MagicMock(spec=BaseModelSelector)
        self.selector.select_model.return_value = ("test_provider", "test_model")
        
        self.fallback_strategy = MagicMock(spec=FallbackStrategy)
        self.fallback_strategy.handle_failure.return_value = ("fallback_provider", "fallback_model")
        
        # Create a model switcher
        self.switcher = ModelSwitcher(
            config=Config(),
            selector=self.selector,
            fallback_strategy=self.fallback_strategy
        )
    
    def test_init(self):
        """Test initialization."""
        # Check if switcher is initialized correctly
        assert self.switcher.selector is self.selector
        assert self.switcher.fallback_strategy is self.fallback_strategy
        assert hasattr(self.switcher, "config")
        assert self.switcher.current_model is None
        assert self.switcher.failed_models == []
    
    def test_select_model_for_task(self):
        """Test select_model_for_task."""
        # Select model for task
        provider, model = self.switcher.select_model_for_task("chat")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
        
        # Check if current model is updated
        assert self.switcher.current_model == ("test_provider", "test_model")
    
    def test_select_model_for_code_task(self):
        """Test select_model_for_task with code task."""
        # Select model for code task
        provider, model = self.switcher.select_model_for_task("code")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        assert ModelCapability.CODE_GENERATION in requirement.capabilities
        assert ModelCapability.CODE_UNDERSTANDING in requirement.capabilities
        assert ModelCapability.FUNCTION_CALLING in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_for_reasoning_task(self):
        """Test select_model_for_task with reasoning task."""
        # Select model for reasoning task
        provider, model = self.switcher.select_model_for_task("reasoning")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        assert ModelCapability.REASONING in requirement.capabilities
        assert ModelCapability.PLANNING in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_for_tool_use_task(self):
        """Test select_model_for_task with tool_use task."""
        # Select model for tool_use task
        provider, model = self.switcher.select_model_for_task("tool_use")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        assert ModelCapability.FUNCTION_CALLING in requirement.capabilities
        assert ModelCapability.TOOL_USE in requirement.capabilities
        assert ModelCapability.REASONING in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_for_math_task(self):
        """Test select_model_for_task with math task."""
        # Select model for math task
        provider, model = self.switcher.select_model_for_task("math")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        assert ModelCapability.MATH in requirement.capabilities
        assert ModelCapability.REASONING in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_for_creative_task(self):
        """Test select_model_for_task with creative task."""
        # Select model for creative task
        provider, model = self.switcher.select_model_for_task("creative")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        assert ModelCapability.CREATIVE_WRITING in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_for_summarization_task(self):
        """Test select_model_for_task with summarization task."""
        # Select model for summarization task
        provider, model = self.switcher.select_model_for_task("summarization")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert ModelCapability.TEXT_GENERATION in requirement.capabilities
        assert ModelCapability.CHAT in requirement.capabilities
        assert ModelCapability.SUMMARIZATION in requirement.capabilities
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_with_context_length(self):
        """Test select_model_for_task with context length."""
        # Select model for task with context length
        provider, model = self.switcher.select_model_for_task("chat", context_length=8192)
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert requirement.min_context_length == 8192
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_with_max_cost(self):
        """Test select_model_for_task with max cost."""
        # Select model for task with max cost
        provider, model = self.switcher.select_model_for_task("chat", max_cost=0.0001)
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert requirement.max_cost_per_token == 0.0001
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_with_preferred_provider(self):
        """Test select_model_for_task with preferred provider."""
        # Select model for task with preferred provider
        provider, model = self.switcher.select_model_for_task("chat", preferred_provider="preferred_provider")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert requirement.preferred_provider == "preferred_provider"
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_select_model_with_preferred_model(self):
        """Test select_model_for_task with preferred model."""
        # Select model for task with preferred model
        provider, model = self.switcher.select_model_for_task("chat", preferred_model="preferred_model")
        
        # Check if selector was called with correct arguments
        self.selector.select_model.assert_called_once()
        args, kwargs = self.selector.select_model.call_args
        requirement = args[0]
        assert isinstance(requirement, ModelRequirement)
        assert requirement.preferred_model == "preferred_model"
        
        # Check if result is correct
        assert provider == "test_provider"
        assert model == "test_model"
    
    def test_handle_model_failure(self):
        """Test handle_model_failure."""
        # Set current model
        self.switcher.current_model = ("test_provider", "test_model")
        
        # Set up registry to return model info
        model_info = MagicMock(spec=ModelInfo)
        model_info.capabilities = ["text_generation", "chat"]
        model_info.context_length = 4096
        self.selector.registry.get_model_info.return_value = model_info
        
        # Handle model failure
        provider, model = self.switcher.handle_model_failure()
        
        # Check if fallback strategy was called with correct arguments
        self.fallback_strategy.handle_failure.assert_called_once()
        args, kwargs = self.fallback_strategy.handle_failure.call_args
        assert args[0] == ("test_provider", "test_model")
        assert isinstance(args[1], ModelRequirement)
        assert args[1].capabilities == ["text_generation", "chat"]
        assert args[1].min_context_length == 2048  # Half of original
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
        
        # Check if current model is updated
        assert self.switcher.current_model == ("fallback_provider", "fallback_model")
        
        # Check if failed models is updated
        assert self.switcher.failed_models == [("test_provider", "test_model")]
    
    def test_handle_model_failure_with_error(self):
        """Test handle_model_failure with error."""
        # Set current model
        self.switcher.current_model = ("test_provider", "test_model")
        
        # Set up registry to return model info
        model_info = MagicMock(spec=ModelInfo)
        model_info.capabilities = ["text_generation", "chat"]
        model_info.context_length = 4096
        self.selector.registry.get_model_info.return_value = model_info
        
        # Create an error
        error = ValueError("Test error")
        
        # Handle model failure
        provider, model = self.switcher.handle_model_failure(error=error)
        
        # Check if fallback strategy was called with correct arguments
        self.fallback_strategy.handle_failure.assert_called_once()
        args, kwargs = self.fallback_strategy.handle_failure.call_args
        assert args[0] == ("test_provider", "test_model")
        assert isinstance(args[1], ModelRequirement)
        assert kwargs["error"] == error
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
    
    def test_handle_model_failure_with_context(self):
        """Test handle_model_failure with context."""
        # Set current model
        self.switcher.current_model = ("test_provider", "test_model")
        
        # Set up registry to return model info
        model_info = MagicMock(spec=ModelInfo)
        model_info.capabilities = ["text_generation", "chat"]
        model_info.context_length = 4096
        self.selector.registry.get_model_info.return_value = model_info
        
        # Create a context
        context = {"test_key": "test_value"}
        
        # Handle model failure
        provider, model = self.switcher.handle_model_failure(context=context)
        
        # Check if fallback strategy was called with correct arguments
        self.fallback_strategy.handle_failure.assert_called_once()
        args, kwargs = self.fallback_strategy.handle_failure.call_args
        assert args[0] == ("test_provider", "test_model")
        assert isinstance(args[1], ModelRequirement)
        assert kwargs["context"] == context
        
        # Check if result is correct
        assert provider == "fallback_provider"
        assert model == "fallback_model"
    
    def test_handle_model_failure_with_no_current_model(self):
        """Test handle_model_failure with no current model."""
        # Try to handle model failure
        with pytest.raises(ValueError):
            self.switcher.handle_model_failure()
        
        # Check if fallback strategy was not called
        self.fallback_strategy.handle_failure.assert_not_called()
    
    def test_handle_model_failure_with_no_model_info(self):
        """Test handle_model_failure with no model info."""
        # Set current model
        self.switcher.current_model = ("test_provider", "test_model")
        
        # Set up registry to return no model info
        self.selector.registry.get_model_info.return_value = None
        
        # Try to handle model failure
        with pytest.raises(ValueError):
            self.switcher.handle_model_failure()
        
        # Check if fallback strategy was not called
        self.fallback_strategy.handle_failure.assert_not_called()
    
    def test_handle_model_failure_with_all_models_failed(self):
        """Test handle_model_failure with all models failed."""
        # Set current model
        self.switcher.current_model = ("test_provider", "test_model")
        
        # Set up registry to return model info
        model_info = MagicMock(spec=ModelInfo)
        model_info.capabilities = ["text_generation", "chat"]
        model_info.context_length = 4096
        self.selector.registry.get_model_info.return_value = model_info
        
        # Set up registry to return all models
        self.selector.registry.list_models.return_value = [model_info]
        
        # Set up fallback strategy to raise an error
        self.fallback_strategy.handle_failure.side_effect = ValueError("No suitable fallback model")
        
        # Set failed models to include all models
        self.switcher.failed_models = [("test_provider", "test_model")]
        
        # Try to handle model failure
        with pytest.raises(ValueError):
            self.switcher.handle_model_failure()
    
    def test_reset(self):
        """Test reset."""
        # Set current model and failed models
        self.switcher.current_model = ("test_provider", "test_model")
        self.switcher.failed_models = [("test_provider", "test_model")]
        
        # Reset
        self.switcher.reset()
        
        # Check if state is reset
        assert self.switcher.current_model is None
        assert self.switcher.failed_models == []
