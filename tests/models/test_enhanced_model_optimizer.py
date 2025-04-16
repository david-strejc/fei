"""
Tests for the enhanced model optimizer.
"""

import pytest
from unittest.mock import MagicMock, patch
import time

from fei.core.enhanced_model_optimizer import EnhancedModelOptimizer
from fei.models.integration import ModelSwitcher
from fei.utils.config import Config


class TestEnhancedModelOptimizer:
    """Test the EnhancedModelOptimizer class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a mock config
        self.config = MagicMock(spec=Config)
        self.config.get_bool.return_value = True
        self.config.get_string.return_value = "balanced"
        self.config.get_list.return_value = ["test_provider"]
        
        # Create a mock model switcher
        self.model_switcher = MagicMock(spec=ModelSwitcher)
        self.model_switcher.select_model_for_task.return_value = ("test_provider", "test_model")
        self.model_switcher.handle_model_failure.return_value = ("fallback_provider", "fallback_model")
        
        # Patch the ModelSwitcher class
        self.model_switcher_patch = patch("fei.core.enhanced_model_optimizer.ModelSwitcher", return_value=self.model_switcher)
        self.model_switcher_patch.start()
        
        # Create an optimizer
        self.optimizer = EnhancedModelOptimizer(self.config)
    
    def teardown_method(self):
        """Tear down the test."""
        # Stop the patch
        self.model_switcher_patch.stop()
    
    def test_init(self):
        """Test initialization."""
        # Check if optimizer is initialized correctly
        assert self.optimizer.config is self.config
        assert self.optimizer.model_switcher is self.model_switcher
        assert hasattr(self.optimizer, "update_interval")
        assert self.optimizer._last_update_time == 0
        assert self.optimizer._cached_optimal_model is None
    
    def test_get_optimal_model_disabled(self):
        """Test get_optimal_model with optimizer disabled."""
        # Set up config to disable optimizer
        self.config.get_bool.return_value = False
        
        # Get optimal model
        model = self.optimizer.get_optimal_model("test_provider", "test_model")
        
        # Check if result is the current model
        assert model == "test_model"
        
        # Check if model switcher was not called
        self.model_switcher.select_model_for_task.assert_not_called()
    
    def test_get_optimal_model_with_cache(self):
        """Test get_optimal_model with cache."""
        # Set up cache
        self.optimizer._cached_optimal_model = ("cached_provider", "cached_model")
        self.optimizer._last_update_time = time.time()
        
        # Get optimal model
        model = self.optimizer.get_optimal_model("test_provider", "test_model")
        
        # Check if result is the cached model
        assert model == "cached_provider/cached_model"
        
        # Check if model switcher was not called
        self.model_switcher.select_model_for_task.assert_not_called()
    
    def test_get_optimal_model_with_expired_cache(self):
        """Test get_optimal_model with expired cache."""
        # Set up expired cache
        self.optimizer._cached_optimal_model = ("cached_provider", "cached_model")
        self.optimizer._last_update_time = time.time() - self.optimizer.update_interval - 1
        
        # Get optimal model
        model = self.optimizer.get_optimal_model("test_provider", "test_model")
        
        # Check if result is the new model
        assert model == "test_provider/test_model"
        
        # Check if model switcher was called with correct arguments
        self.model_switcher.select_model_for_task.assert_called_once()
        args, kwargs = self.model_switcher.select_model_for_task.call_args
        assert args[0] == "chat"  # Default task type
        assert "context_length" in kwargs
        assert "max_cost" in kwargs
        assert kwargs["preferred_provider"] == "test_provider"
    
    def test_get_optimal_model_with_task_description(self):
        """Test get_optimal_model with task description."""
        # Get optimal model with task description
        model = self.optimizer.get_optimal_model("test_provider", "test_model", "Write some code to solve this problem")
        
        # Check if result is the new model
        assert model == "test_provider/test_model"
        
        # Check if model switcher was called with correct arguments
        self.model_switcher.select_model_for_task.assert_called_once()
        args, kwargs = self.model_switcher.select_model_for_task.call_args
        assert args[0] == "code"  # Task type determined from description
    
    def test_get_optimal_model_with_error(self):
        """Test get_optimal_model with error."""
        # Set up model switcher to raise an error
        self.model_switcher.select_model_for_task.side_effect = ValueError("Test error")
        
        # Get optimal model
        model = self.optimizer.get_optimal_model("test_provider", "test_model")
        
        # Check if result is the current model
        assert model == "test_model"
    
    def test_handle_model_failure(self):
        """Test handle_model_failure."""
        # Handle model failure
        model = self.optimizer.handle_model_failure()
        
        # Check if result is the fallback model
        assert model == "fallback_provider/fallback_model"
        
        # Check if model switcher was called with correct arguments
        self.model_switcher.handle_model_failure.assert_called_once_with(None, None)
        
        # Check if cache is updated
        assert self.optimizer._cached_optimal_model == ("fallback_provider", "fallback_model")
    
    def test_handle_model_failure_with_error(self):
        """Test handle_model_failure with error."""
        # Create an error
        error = ValueError("Test error")
        
        # Handle model failure
        model = self.optimizer.handle_model_failure(error=error)
        
        # Check if result is the fallback model
        assert model == "fallback_provider/fallback_model"
        
        # Check if model switcher was called with correct arguments
        self.model_switcher.handle_model_failure.assert_called_once_with(error, None)
    
    def test_handle_model_failure_with_context(self):
        """Test handle_model_failure with context."""
        # Create a context
        context = {"test_key": "test_value"}
        
        # Handle model failure
        model = self.optimizer.handle_model_failure(context=context)
        
        # Check if result is the fallback model
        assert model == "fallback_provider/fallback_model"
        
        # Check if model switcher was called with correct arguments
        self.model_switcher.handle_model_failure.assert_called_once_with(None, context)
    
    def test_handle_model_failure_with_error(self):
        """Test handle_model_failure with error from model switcher."""
        # Set up model switcher to raise an error
        self.model_switcher.handle_model_failure.side_effect = ValueError("Test error")
        
        # Handle model failure
        model = self.optimizer.handle_model_failure()
        
        # Check if result is None
        assert model is None
    
    def test_reset(self):
        """Test reset."""
        # Set up cache
        self.optimizer._cached_optimal_model = ("cached_provider", "cached_model")
        self.optimizer._last_update_time = time.time()
        
        # Reset
        self.optimizer.reset()
        
        # Check if cache is reset
        assert self.optimizer._cached_optimal_model is None
        assert self.optimizer._last_update_time == 0
        
        # Check if model switcher was reset
        self.model_switcher.reset.assert_called_once()
    
    def test_determine_task_type_code(self):
        """Test _determine_task_type with code task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Write a Python function to calculate Fibonacci numbers")
        
        # Check if result is correct
        assert task_type == "code"
    
    def test_determine_task_type_reasoning(self):
        """Test _determine_task_type with reasoning task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Analyze the logical fallacies in this argument")
        
        # Check if result is correct
        assert task_type == "reasoning"
    
    def test_determine_task_type_tool_use(self):
        """Test _determine_task_type with tool_use task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Execute this command and analyze the output")
        
        # Check if result is correct
        assert task_type == "tool_use"
    
    def test_determine_task_type_math(self):
        """Test _determine_task_type with math task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Solve this equation: 2x + 3 = 7")
        
        # Check if result is correct
        assert task_type == "math"
    
    def test_determine_task_type_creative(self):
        """Test _determine_task_type with creative task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Write a short story about a robot learning to love")
        
        # Check if result is correct
        assert task_type == "creative"
    
    def test_determine_task_type_summarization(self):
        """Test _determine_task_type with summarization task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Summarize this article about climate change")
        
        # Check if result is correct
        assert task_type == "summarization"
    
    def test_determine_task_type_default(self):
        """Test _determine_task_type with default task."""
        # Determine task type
        task_type = self.optimizer._determine_task_type("Hello, how are you today?")
        
        # Check if result is correct
        assert task_type == "chat"
    
    def test_determine_task_type_none(self):
        """Test _determine_task_type with None."""
        # Determine task type
        task_type = self.optimizer._determine_task_type(None)
        
        # Check if result is correct
        assert task_type == "chat"
