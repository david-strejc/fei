"""
Model selection and switching for Fei.

This module provides interfaces and implementations for selecting and switching
between different LLM models based on requirements and context.
"""

from typing import Dict, List, Optional, Any, Union, Tuple

from fei.models.base import ModelCapability, ModelRequirement, ModelInfo, ModelSelectorInterface
from fei.models.registry import ModelRegistry
from fei.models.selector import (
    BaseModelSelector, CostOptimizedSelector, PerformanceOptimizedSelector,
    BalancedSelector, RandomSelector
)
from fei.models.fallback import (
    FallbackStrategy, RetryStrategy, ProviderFallbackStrategy, ChainedFallbackStrategy
)
from fei.models.config import ModelConfig

__all__ = [
    'ModelCapability',
    'ModelRequirement',
    'ModelInfo',
    'ModelSelectorInterface',
    'ModelRegistry',
    'BaseModelSelector',
    'CostOptimizedSelector',
    'PerformanceOptimizedSelector',
    'BalancedSelector',
    'RandomSelector',
    'FallbackStrategy',
    'RetryStrategy',
    'ProviderFallbackStrategy',
    'ChainedFallbackStrategy',
    'ModelConfig',
    'create_default_selector',
    'create_default_fallback_strategy',
]


def create_default_selector(config: Optional[Any] = None) -> BaseModelSelector:
    """
    Create a default model selector based on configuration.
    
    Args:
        config: Optional configuration
        
    Returns:
        Model selector
    """
    model_config = ModelConfig(config)
    registry = ModelRegistry(config)
    
    selector_config = model_config.get_selector_config()
    selector_type = selector_config.get("type", "balanced")
    
    if selector_type == "cost":
        return CostOptimizedSelector(registry, config)
    elif selector_type == "performance":
        return PerformanceOptimizedSelector(registry, config)
    elif selector_type == "random":
        return RandomSelector(registry, config)
    else:
        return BalancedSelector(registry, config)


def create_default_fallback_strategy(selector: Optional[BaseModelSelector] = None, config: Optional[Any] = None) -> FallbackStrategy:
    """
    Create a default fallback strategy based on configuration.
    
    Args:
        selector: Optional model selector
        config: Optional configuration
        
    Returns:
        Fallback strategy
    """
    model_config = ModelConfig(config)
    
    if selector is None:
        selector = create_default_selector(config)
    
    fallback_config = model_config.get_fallback_config()
    fallback_type = fallback_config.get("type", "chained")
    
    if fallback_type == "retry":
        max_retries = fallback_config.get("max_retries", 3)
        retry_delay = fallback_config.get("retry_delay", 1.0)
        return RetryStrategy(selector, max_retries, retry_delay, config)
    elif fallback_type == "provider":
        return ProviderFallbackStrategy(selector, config)
    elif fallback_type == "chained":
        # Create a chain of retry -> provider -> basic
        retry_strategy = RetryStrategy(selector, config=config)
        provider_strategy = ProviderFallbackStrategy(selector, config=config)
        basic_strategy = FallbackStrategy(selector, config=config)
        
        return ChainedFallbackStrategy(selector, [retry_strategy, provider_strategy, basic_strategy], config)
    else:
        return FallbackStrategy(selector, config)
