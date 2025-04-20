"""
Base interfaces and classes for model selection and switching.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple


class ModelCapability:
    """Represents a capability that a model can have."""
    
    # Core capabilities
    TEXT_GENERATION = "text_generation"
    CHAT = "chat"
    FUNCTION_CALLING = "function_calling"
    
    # Advanced capabilities
    CODE_GENERATION = "code_generation"
    CODE_UNDERSTANDING = "code_understanding"
    REASONING = "reasoning"
    PLANNING = "planning"
    TOOL_USE = "tool_use"
    
    # Specialized capabilities
    MATH = "math"
    SCIENCE = "science"
    CREATIVE_WRITING = "creative_writing"
    SUMMARIZATION = "summarization"


class ModelRequirement:
    """Represents a requirement for model selection."""
    
    def __init__(
        self,
        capabilities: List[str],
        min_context_length: Optional[int] = None,
        max_cost_per_token: Optional[float] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None,
        min_reliability: Optional[float] = None,
    ):
        """
        Initialize model requirements.
        
        Args:
            capabilities: List of required capabilities
            min_context_length: Minimum context length required
            max_cost_per_token: Maximum cost per token allowed
            preferred_provider: Preferred provider (if any)
            preferred_model: Preferred model (if any)
            min_reliability: Minimum reliability score required (0-1)
        """
        self.capabilities = capabilities
        self.min_context_length = min_context_length
        self.max_cost_per_token = max_cost_per_token
        self.preferred_provider = preferred_provider
        self.preferred_model = preferred_model
        self.min_reliability = min_reliability


class ModelInfo:
    """Information about a model."""
    
    def __init__(
        self,
        model_id: str,
        provider: str,
        capabilities: List[str],
        context_length: int,
        cost_per_input_token: float,
        cost_per_output_token: float,
        reliability: float = 0.95,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model information.
        
        Args:
            model_id: Model identifier
            provider: Provider name
            capabilities: List of model capabilities
            context_length: Maximum context length
            cost_per_input_token: Cost per input token
            cost_per_output_token: Cost per output token
            reliability: Reliability score (0-1)
            metadata: Additional metadata
        """
        self.model_id = model_id
        self.provider = provider
        self.capabilities = capabilities
        self.context_length = context_length
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.reliability = reliability
        self.metadata = metadata or {}
    
    def meets_requirements(self, requirements: ModelRequirement) -> bool:
        """
        Check if the model meets the given requirements.
        
        Args:
            requirements: Model requirements
            
        Returns:
            True if the model meets all requirements, False otherwise
        """
        # Check capabilities
        for capability in requirements.capabilities:
            if capability not in self.capabilities:
                return False
        
        # Check context length
        if requirements.min_context_length and self.context_length < requirements.min_context_length:
            return False
        
        # Check cost
        if requirements.max_cost_per_token and (
            self.cost_per_input_token > requirements.max_cost_per_token or
            self.cost_per_output_token > requirements.max_cost_per_token
        ):
            return False
        
        # Check provider
        if requirements.preferred_provider and self.provider != requirements.preferred_provider:
            return False
        
        # Check model ID
        if requirements.preferred_model and self.model_id != requirements.preferred_model:
            return False
        
        # Check reliability
        if requirements.min_reliability and self.reliability < requirements.min_reliability:
            return False
        
        return True


class ModelSelectorInterface(ABC):
    """Interface for model selectors."""
    
    @abstractmethod
    def select_model(
        self,
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Select a model based on requirements and context.
        
        Args:
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            Tuple of (provider_name, model_id)
            
        Raises:
            ValueError: If no suitable model is found
        """
        pass
    
    @abstractmethod
    def get_fallback_model(
        self,
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Get a fallback model when the primary model fails.
        
        Args:
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            Tuple of (provider_name, model_id)
            
        Raises:
            ValueError: If no suitable fallback model is found
        """
        pass
