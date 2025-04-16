"""
Model selection strategies for choosing the best model for a task.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import random

from fei.models.base import ModelSelectorInterface, ModelRequirement, ModelInfo
from fei.models.registry import ModelRegistry
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class BaseModelSelector(ModelSelectorInterface):
    """Base class for model selectors."""
    
    def __init__(self, registry: Optional[ModelRegistry] = None, config: Optional[Config] = None):
        """
        Initialize the model selector.
        
        Args:
            registry: Optional model registry
            config: Optional configuration
        """
        self.config = config or Config()
        self.registry = registry or ModelRegistry(self.config)
    
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
        # Find models that meet the requirements
        suitable_models = self.registry.find_models(requirements)
        
        if not suitable_models:
            raise ValueError("No suitable model found for the given requirements")
        
        # Select the best model
        selected_model = self._select_best_model(suitable_models, requirements, context)
        
        return selected_model.provider, selected_model.model_id
    
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
        # Find models that meet the requirements
        suitable_models = self.registry.find_models(requirements)
        
        # Remove the failed model
        suitable_models = [
            model for model in suitable_models
            if (model.provider, model.model_id) != failed_model
        ]
        
        if not suitable_models:
            raise ValueError("No suitable fallback model found")
        
        # Select the best fallback model
        fallback_model = self._select_best_fallback_model(suitable_models, failed_model, requirements, context)
        
        return fallback_model.provider, fallback_model.model_id
    
    def _select_best_model(
        self,
        models: List[ModelInfo],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the best model from a list of suitable models.
        
        Args:
            models: List of suitable models
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The best model
        """
        # Default implementation: return the first model
        return models[0]
    
    def _select_best_fallback_model(
        self,
        models: List[ModelInfo],
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the best fallback model from a list of suitable models.
        
        Args:
            models: List of suitable models
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The best fallback model
        """
        # Default implementation: return the first model
        return models[0]


class CostOptimizedSelector(BaseModelSelector):
    """Model selector that optimizes for cost."""
    
    def _select_best_model(
        self,
        models: List[ModelInfo],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the model with the lowest cost.
        
        Args:
            models: List of suitable models
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The model with the lowest cost
        """
        # Sort models by cost (average of input and output cost)
        return min(
            models,
            key=lambda model: (model.cost_per_input_token + model.cost_per_output_token) / 2
        )
    
    def _select_best_fallback_model(
        self,
        models: List[ModelInfo],
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the fallback model with the lowest cost.
        
        Args:
            models: List of suitable models
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The fallback model with the lowest cost
        """
        # Sort models by cost (average of input and output cost)
        return min(
            models,
            key=lambda model: (model.cost_per_input_token + model.cost_per_output_token) / 2
        )


class PerformanceOptimizedSelector(BaseModelSelector):
    """Model selector that optimizes for performance."""
    
    def _select_best_model(
        self,
        models: List[ModelInfo],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the model with the best performance.
        
        Args:
            models: List of suitable models
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The model with the best performance
        """
        # Sort models by reliability and context length
        return max(
            models,
            key=lambda model: (model.reliability, model.context_length)
        )
    
    def _select_best_fallback_model(
        self,
        models: List[ModelInfo],
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the fallback model with the best performance.
        
        Args:
            models: List of suitable models
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The fallback model with the best performance
        """
        # Sort models by reliability and context length
        return max(
            models,
            key=lambda model: (model.reliability, model.context_length)
        )


class BalancedSelector(BaseModelSelector):
    """Model selector that balances cost and performance."""
    
    def _select_best_model(
        self,
        models: List[ModelInfo],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the model with the best balance of cost and performance.
        
        Args:
            models: List of suitable models
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The model with the best balance of cost and performance
        """
        # Calculate a score for each model
        def score(model):
            cost_factor = (model.cost_per_input_token + model.cost_per_output_token) / 2
            performance_factor = model.reliability * (model.context_length / 100000)
            
            # Lower cost and higher performance is better
            return performance_factor / (cost_factor + 0.0001)
        
        return max(models, key=score)
    
    def _select_best_fallback_model(
        self,
        models: List[ModelInfo],
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Select the fallback model with the best balance of cost and performance.
        
        Args:
            models: List of suitable models
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            The fallback model with the best balance of cost and performance
        """
        # Prefer models from a different provider
        different_provider_models = [
            model for model in models
            if model.provider != failed_model[0]
        ]
        
        if different_provider_models:
            models = different_provider_models
        
        # Calculate a score for each model
        def score(model):
            cost_factor = (model.cost_per_input_token + model.cost_per_output_token) / 2
            performance_factor = model.reliability * (model.context_length / 100000)
            
            # Lower cost and higher performance is better
            return performance_factor / (cost_factor + 0.0001)
        
        return max(models, key=score)


class RandomSelector(BaseModelSelector):
    """Model selector that randomly selects a model."""
    
    def _select_best_model(
        self,
        models: List[ModelInfo],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Randomly select a model.
        
        Args:
            models: List of suitable models
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            A randomly selected model
        """
        return random.choice(models)
    
    def _select_best_fallback_model(
        self,
        models: List[ModelInfo],
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None
    ) -> ModelInfo:
        """
        Randomly select a fallback model.
        
        Args:
            models: List of suitable models
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            
        Returns:
            A randomly selected fallback model
        """
        return random.choice(models)
