"""
Tests for the model switching base components.
"""

import pytest
from unittest.mock import MagicMock

from fei.models.base import ModelCapability, ModelRequirement, ModelInfo, ModelSelectorInterface


class TestModelCapability:
    """Test the ModelCapability class."""
    
    def test_capabilities_exist(self):
        """Test that capabilities are defined."""
        assert hasattr(ModelCapability, "TEXT_GENERATION")
        assert hasattr(ModelCapability, "CHAT")
        assert hasattr(ModelCapability, "FUNCTION_CALLING")
        assert hasattr(ModelCapability, "CODE_GENERATION")
        assert hasattr(ModelCapability, "CODE_UNDERSTANDING")
        assert hasattr(ModelCapability, "REASONING")
        assert hasattr(ModelCapability, "PLANNING")
        assert hasattr(ModelCapability, "TOOL_USE")
        assert hasattr(ModelCapability, "MATH")
        assert hasattr(ModelCapability, "SCIENCE")
        assert hasattr(ModelCapability, "CREATIVE_WRITING")
        assert hasattr(ModelCapability, "SUMMARIZATION")


class TestModelRequirement:
    """Test the ModelRequirement class."""
    
    def test_init(self):
        """Test initialization."""
        # Create a requirement
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=4096,
            max_cost_per_token=0.0001,
            preferred_provider="test_provider",
            preferred_model="test_model",
            min_reliability=0.9
        )
        
        # Check if requirement is initialized correctly
        assert requirement.capabilities == ["text_generation", "chat"]
        assert requirement.min_context_length == 4096
        assert requirement.max_cost_per_token == 0.0001
        assert requirement.preferred_provider == "test_provider"
        assert requirement.preferred_model == "test_model"
        assert requirement.min_reliability == 0.9
    
    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        # Create a requirement with only required parameters
        requirement = ModelRequirement(capabilities=["text_generation"])
        
        # Check if requirement has default values
        assert requirement.capabilities == ["text_generation"]
        assert requirement.min_context_length is None
        assert requirement.max_cost_per_token is None
        assert requirement.preferred_provider is None
        assert requirement.preferred_model is None
        assert requirement.min_reliability is None


class TestModelInfo:
    """Test the ModelInfo class."""
    
    def test_init(self):
        """Test initialization."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95,
            metadata={"test_key": "test_value"}
        )
        
        # Check if model info is initialized correctly
        assert model_info.model_id == "test_model"
        assert model_info.provider == "test_provider"
        assert model_info.capabilities == ["text_generation", "chat"]
        assert model_info.context_length == 4096
        assert model_info.cost_per_input_token == 0.0001
        assert model_info.cost_per_output_token == 0.0002
        assert model_info.reliability == 0.95
        assert model_info.metadata == {"test_key": "test_value"}
    
    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        # Create a model info with only required parameters
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002
        )
        
        # Check if model info has default values
        assert model_info.reliability == 0.95
        assert model_info.metadata == {}
    
    def test_meets_requirements(self):
        """Test meets_requirements method."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat", "function_calling"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95
        )
        
        # Create a requirement that the model meets
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=2048,
            max_cost_per_token=0.0002,
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is True
    
    def test_meets_requirements_with_missing_capability(self):
        """Test meets_requirements with missing capability."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95
        )
        
        # Create a requirement with a capability the model doesn't have
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat", "function_calling"],
            min_context_length=2048,
            max_cost_per_token=0.0002,
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is False
    
    def test_meets_requirements_with_insufficient_context_length(self):
        """Test meets_requirements with insufficient context length."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95
        )
        
        # Create a requirement with a context length the model doesn't meet
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=8192,
            max_cost_per_token=0.0002,
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is False
    
    def test_meets_requirements_with_excessive_cost(self):
        """Test meets_requirements with excessive cost."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95
        )
        
        # Create a requirement with a cost constraint the model doesn't meet
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=2048,
            max_cost_per_token=0.00005,
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is False
    
    def test_meets_requirements_with_different_provider(self):
        """Test meets_requirements with different provider."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95
        )
        
        # Create a requirement with a different preferred provider
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=2048,
            max_cost_per_token=0.0002,
            preferred_provider="other_provider",
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is False
    
    def test_meets_requirements_with_different_model(self):
        """Test meets_requirements with different model."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.95
        )
        
        # Create a requirement with a different preferred model
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=2048,
            max_cost_per_token=0.0002,
            preferred_model="other_model",
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is False
    
    def test_meets_requirements_with_insufficient_reliability(self):
        """Test meets_requirements with insufficient reliability."""
        # Create a model info
        model_info = ModelInfo(
            model_id="test_model",
            provider="test_provider",
            capabilities=["text_generation", "chat"],
            context_length=4096,
            cost_per_input_token=0.0001,
            cost_per_output_token=0.0002,
            reliability=0.85
        )
        
        # Create a requirement with a reliability constraint the model doesn't meet
        requirement = ModelRequirement(
            capabilities=["text_generation", "chat"],
            min_context_length=2048,
            max_cost_per_token=0.0002,
            min_reliability=0.9
        )
        
        # Check if model meets the requirement
        assert model_info.meets_requirements(requirement) is False


class MockModelSelector(ModelSelectorInterface):
    """Mock implementation of ModelSelectorInterface."""
    
    def select_model(self, requirements, context=None):
        return "test_provider", "test_model"
    
    def get_fallback_model(self, failed_model, requirements, context=None):
        return "fallback_provider", "fallback_model"


class TestModelSelectorInterface:
    """Test the ModelSelectorInterface."""
    
    def test_abstract_methods(self):
        """Test that abstract methods are defined."""
        # Check if abstract methods are defined
        assert hasattr(ModelSelectorInterface, "select_model")
        assert hasattr(ModelSelectorInterface, "get_fallback_model")
        
        # Check if methods are abstract
        with pytest.raises(TypeError):
            ModelSelectorInterface()
    
    def test_mock_implementation(self):
        """Test mock implementation."""
        # Create a mock selector
        selector = MockModelSelector()
        
        # Create a requirement
        requirement = ModelRequirement(capabilities=["text_generation", "chat"])
        
        # Test select_model
        provider, model = selector.select_model(requirement)
        assert provider == "test_provider"
        assert model == "test_model"
        
        # Test get_fallback_model
        provider, model = selector.get_fallback_model(("test_provider", "test_model"), requirement)
        assert provider == "fallback_provider"
        assert model == "fallback_model"
