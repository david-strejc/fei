"""
Tests for the model registry.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil

from fei.models.registry import ModelRegistry
from fei.models.base import ModelInfo, ModelRequirement
from fei.utils.config import Config


class TestModelRegistry:
    """Test the ModelRegistry class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a temporary config file
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.config_data = {
            "models": {
                "test_provider": {
                    "test_model": {
                        "capabilities": ["text_generation", "chat"],
                        "context_length": 4096,
                        "cost_per_input_token": 0.0001,
                        "cost_per_output_token": 0.0002,
                        "reliability": 0.95,
                        "metadata": {"test_key": "test_value"}
                    }
                }
            }
        }
        
        with open(self.config_path, "w") as f:
            json.dump(self.config_data, f)
        
        # Create a config with the temporary file
        self.config = Config(config_file=self.config_path)
        
        # Create a registry
        self.registry = ModelRegistry(self.config)
    
    def teardown_method(self):
        """Tear down the test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        # Check if registry is initialized correctly
        assert hasattr(self.registry, "models")
        assert hasattr(self.registry, "config")
        
        # Check if models are loaded from config
        key = ("test_provider", "test_model")
        assert key in self.registry.models
        assert self.registry.models[key].model_id == "test_model"
        assert self.registry.models[key].provider == "test_provider"
        assert self.registry.models[key].capabilities == ["text_generation", "chat"]
        assert self.registry.models[key].context_length == 4096
        assert self.registry.models[key].cost_per_input_token == 0.0001
        assert self.registry.models[key].cost_per_output_token == 0.0002
        assert self.registry.models[key].reliability == 0.95
        assert self.registry.models[key].metadata == {"test_key": "test_value"}
    
    def test_register_model(self):
        """Test register_model."""
        # Create a model info
        model_info = ModelInfo(
            model_id="new_model",
            provider="new_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002
        )
        
        # Register the model
        self.registry.register_model(model_info)
        
        # Check if model is registered
        key = ("new_provider", "new_model")
        assert key in self.registry.models
        assert self.registry.models[key] is model_info
    
    def test_unregister_model(self):
        """Test unregister_model."""
        # Unregister a model
        result = self.registry.unregister_model("test_provider", "test_model")
        
        # Check if model is unregistered
        assert result is True
        key = ("test_provider", "test_model")
        assert key not in self.registry.models
    
    def test_unregister_nonexistent_model(self):
        """Test unregister_model with non-existent model."""
        # Unregister a non-existent model
        result = self.registry.unregister_model("nonexistent", "nonexistent")
        
        # Check if result is False
        assert result is False
    
    def test_get_model_info(self):
        """Test get_model_info."""
        # Get model info
        model_info = self.registry.get_model_info("test_provider", "test_model")
        
        # Check if model info is correct
        assert model_info.model_id == "test_model"
        assert model_info.provider == "test_provider"
        assert model_info.capabilities == ["text_generation", "chat"]
        assert model_info.context_length == 4096
        assert model_info.cost_per_input_token == 0.0001
        assert model_info.cost_per_output_token == 0.0002
        assert model_info.reliability == 0.95
        assert model_info.metadata == {"test_key": "test_value"}
    
    def test_get_nonexistent_model_info(self):
        """Test get_model_info with non-existent model."""
        # Get non-existent model info
        model_info = self.registry.get_model_info("nonexistent", "nonexistent")
        
        # Check if model info is None
        assert model_info is None
    
    def test_list_models(self):
        """Test list_models."""
        # Register another model
        model_info = ModelInfo(
            model_id="new_model",
            provider="new_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002
        )
        self.registry.register_model(model_info)
        
        # List models
        models = self.registry.list_models()
        
        # Check if models are correct
        assert len(models) == 2
        assert any(m.model_id == "test_model" and m.provider == "test_provider" for m in models)
        assert any(m.model_id == "new_model" and m.provider == "new_provider" for m in models)
    
    def test_list_models_by_provider(self):
        """Test list_models_by_provider."""
        # Register another model from the same provider
        model_info = ModelInfo(
            model_id="new_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002
        )
        self.registry.register_model(model_info)
        
        # List models by provider
        models = self.registry.list_models_by_provider("test_provider")
        
        # Check if models are correct
        assert len(models) == 2
        assert any(m.model_id == "test_model" for m in models)
        assert any(m.model_id == "new_model" for m in models)
    
    def test_list_models_by_nonexistent_provider(self):
        """Test list_models_by_provider with non-existent provider."""
        # List models by non-existent provider
        models = self.registry.list_models_by_provider("nonexistent")
        
        # Check if models list is empty
        assert len(models) == 0
    
    def test_find_models(self):
        """Test find_models."""
        # Register another model with different capabilities
        model_info = ModelInfo(
            model_id="new_model",
            provider="new_provider",
            capabilities=["text_generation", "chat", "function_calling"],
            context_length=8192,
            cost_per_input_token=0.0002,
            cost_per_output_token=0.0004
        )
        self.registry.register_model(model_info)
        
        # Create a requirement
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat", "function_calling"],
            min_context_length=4096,
            max_cost_per_token=0.0003
        )
        
        # Find models
        models = self.registry.find_models(requirement)
        
        # Check if models are correct
        assert len(models) == 1
        assert models[0].model_id == "new_model"
        assert models[0].provider == "new_provider"
    
    def test_find_models_with_no_matches(self):
        """Test find_models with no matches."""
        # Create a requirement that no model meets
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat", "function_calling", "nonexistent"],
            min_context_length=16384,
            max_cost_per_token=0.00001
        )
        
        # Find models
        models = self.registry.find_models(requirement)
        
        # Check if models list is empty
        assert len(models) == 0
    
    def test_save_to_config(self):
        """Test save_to_config."""
        # Register a new model
        model_info = ModelInfo(
            model_id="new_model",
            provider="new_provider",
            capabilities=["text_generation", "chat", "function_calling"],
            context_length=8192,
            cost_per_input_token=0.0002,
            cost_per_output_token=0.0004,
            reliability=0.9,
            metadata={"new_key": "new_value"}
        )
        self.registry.register_model(model_info)
        
        # Save to config
        self.registry.save_to_config()
        
        # Check if config is updated
        models_config = self.config.get("models", {})
        assert "new_provider" in models_config
        assert "new_model" in models_config["new_provider"]
        assert models_config["new_provider"]["new_model"]["capabilities"] == ["text_generation", "chat", "function_calling"]
        assert models_config["new_provider"]["new_model"]["context_length"] == 8192
        assert models_config["new_provider"]["new_model"]["cost_per_input_token"] == 0.0002
        assert models_config["new_provider"]["new_model"]["cost_per_output_token"] == 0.0004
        assert models_config["new_provider"]["new_model"]["reliability"] == 0.9
        assert models_config["new_provider"]["new_model"]["metadata"] == {"new_key": "new_value"}
