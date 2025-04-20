"""
Fallback mechanisms for handling model failures.
"""

from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import logging
import time

from fei.models.base import ModelRequirement
from fei.models.selector import BaseModelSelector
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class FallbackStrategy:
    """Base class for fallback strategies."""
    
    def __init__(self, selector: BaseModelSelector, config: Optional[Config] = None):
        """
        Initialize the fallback strategy.
        
        Args:
            selector: Model selector
            config: Optional configuration
        """
        self.selector = selector
        self.config = config or Config()
    
    def handle_failure(
        self,
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> Tuple[str, str]:
        """
        Handle a model failure.
        
        Args:
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            error: Optional exception that caused the failure
            
        Returns:
            Tuple of (provider_name, model_id) for the fallback model
            
        Raises:
            ValueError: If no suitable fallback model is found
        """
        logger.warning(f"Model {failed_model[1]} from provider {failed_model[0]} failed: {error}")
        
        # Get a fallback model
        fallback_model = self.selector.get_fallback_model(failed_model, requirements, context)
        
        logger.info(f"Using fallback model {fallback_model[1]} from provider {fallback_model[0]}")
        
        return fallback_model


class RetryStrategy(FallbackStrategy):
    """Fallback strategy that retries the same model before falling back."""
    
    def __init__(
        self,
        selector: BaseModelSelector,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        config: Optional[Config] = None
    ):
        """
        Initialize the retry strategy.
        
        Args:
            selector: Model selector
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            config: Optional configuration
        """
        super().__init__(selector, config)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_counts: Dict[Tuple[str, str], int] = {}
    
    def handle_failure(
        self,
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> Tuple[str, str]:
        """
        Handle a model failure by retrying or falling back.
        
        Args:
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            error: Optional exception that caused the failure
            
        Returns:
            Tuple of (provider_name, model_id) for the fallback model
            
        Raises:
            ValueError: If no suitable fallback model is found
        """
        # Increment retry count
        self.retry_counts[failed_model] = self.retry_counts.get(failed_model, 0) + 1
        
        # Check if we should retry
        if self.retry_counts[failed_model] <= self.max_retries:
            logger.info(f"Retrying model {failed_model[1]} from provider {failed_model[0]} (attempt {self.retry_counts[failed_model]} of {self.max_retries})")
            
            # Wait before retrying
            time.sleep(self.retry_delay)
            
            # Return the same model
            return failed_model
        
        # Reset retry count
        self.retry_counts[failed_model] = 0
        
        # Fall back to another model
        return super().handle_failure(failed_model, requirements, context, error)


class ProviderFallbackStrategy(FallbackStrategy):
    """Fallback strategy that tries models from different providers."""
    
    def handle_failure(
        self,
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> Tuple[str, str]:
        """
        Handle a model failure by trying a model from a different provider.
        
        Args:
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            error: Optional exception that caused the failure
            
        Returns:
            Tuple of (provider_name, model_id) for the fallback model
            
        Raises:
            ValueError: If no suitable fallback model is found
        """
        logger.warning(f"Model {failed_model[1]} from provider {failed_model[0]} failed: {error}")
        
        # Create a new requirement that excludes the failed provider
        new_requirements = ModelRequirement(
            capabilities=requirements.capabilities,
            min_context_length=requirements.min_context_length,
            max_cost_per_token=requirements.max_cost_per_token,
            preferred_provider=None,  # Clear preferred provider
            preferred_model=None,  # Clear preferred model
            min_reliability=requirements.min_reliability
        )
        
        # Find models that meet the requirements
        suitable_models = self.selector.registry.find_models(new_requirements)
        
        # Filter out models from the failed provider
        suitable_models = [
            model for model in suitable_models
            if model.provider != failed_model[0]
        ]
        
        if not suitable_models:
            # If no models from different providers, fall back to any model
            return super().handle_failure(failed_model, requirements, context, error)
        
        # Select the best model from a different provider
        fallback_model = self.selector._select_best_model(suitable_models, new_requirements, context)
        
        logger.info(f"Using fallback model {fallback_model.model_id} from provider {fallback_model.provider}")
        
        return fallback_model.provider, fallback_model.model_id


class ChainedFallbackStrategy(FallbackStrategy):
    """Fallback strategy that chains multiple strategies."""
    
    def __init__(
        self,
        selector: BaseModelSelector,
        strategies: List[FallbackStrategy],
        config: Optional[Config] = None
    ):
        """
        Initialize the chained fallback strategy.
        
        Args:
            selector: Model selector
            strategies: List of fallback strategies to try in order
            config: Optional configuration
        """
        super().__init__(selector, config)
        self.strategies = strategies
    
    def handle_failure(
        self,
        failed_model: Tuple[str, str],
        requirements: ModelRequirement,
        context: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> Tuple[str, str]:
        """
        Handle a model failure by trying multiple strategies.
        
        Args:
            failed_model: Tuple of (provider_name, model_id) that failed
            requirements: Model requirements
            context: Optional context information
            error: Optional exception that caused the failure
            
        Returns:
            Tuple of (provider_name, model_id) for the fallback model
            
        Raises:
            ValueError: If no suitable fallback model is found
        """
        current_model = failed_model
        current_error = error
        
        # Try each strategy in order
        for strategy in self.strategies:
            try:
                return strategy.handle_failure(current_model, requirements, context, current_error)
            except ValueError as e:
                # Strategy failed, try the next one
                current_error = e
        
        # All strategies failed
        raise ValueError(f"All fallback strategies failed: {current_error}")
