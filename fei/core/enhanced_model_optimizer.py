#!/usr/bin/env python3
"""
Enhanced Model Optimizer for Fei

This module provides an enhanced version of the ModelOptimizer that uses
the new model switching system for more sophisticated model selection.
"""

import time
import os
import asyncio
from typing import Optional, Dict, List, Tuple, Any
import logging

from fei.utils.config import Config
from fei.utils.logging import get_logger
from fei.models.integration import ModelSwitcher
from fei.models.base import ModelRequirement

logger = get_logger(__name__)


class EnhancedModelOptimizer:
    """
    Enhanced model optimizer that uses the model switching system.
    
    This class provides a bridge between the existing ModelOptimizer interface
    and the new model switching system, allowing for more sophisticated model
    selection based on task requirements.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the enhanced model optimizer.
        
        Args:
            config: The Fei configuration object
        """
        self.config = config
        self.model_switcher = ModelSwitcher(config)
        self.update_interval = self.config.get_int("optimizer.update_interval", 3600)
        self._last_update_time = 0
        self._cached_optimal_model: Optional[Tuple[str, str]] = None
    
    def get_optimal_model(
        self,
        current_provider: str,
        current_model: str,
        task_description: Optional[str] = None
    ) -> str:
        """
        Determines the optimal model to use, considering configuration and task.
        
        Args:
            current_provider: The currently configured provider
            current_model: The default model configured in the Assistant
            task_description: Optional description of the task for context-aware optimization
            
        Returns:
            The identifier string of the selected optimal model (e.g., "openai/gpt-4o"),
            or the current_model if optimization is disabled or fails
        """
        if not self.config.get_bool("optimizer.enabled", False):
            logger.debug("Model optimizer is disabled in config.")
            return current_model
        
        # Check if we should use the cached model
        current_time = time.time()
        if self._cached_optimal_model and (current_time - self._last_update_time < self.update_interval):
            provider, model = self._cached_optimal_model
            # Format the model ID in the expected format
            return f"{provider}/{model}" if "/" not in model else model
        
        # Determine task type from description
        task_type = self._determine_task_type(task_description)
        
        try:
            # Get optimization strategy from config
            strategy = self.config.get_string("optimizer.strategy", "balanced")
            
            # Map strategy to parameters
            if strategy == "cost":
                max_cost = 0.00001  # Very low cost per token
                context_length = None  # No specific context length requirement
            elif strategy == "performance":
                max_cost = None  # No cost constraint
                context_length = 100000  # Require large context
            elif strategy == "balanced":
                max_cost = 0.0001  # Moderate cost per token
                context_length = 32000  # Moderate context length
            else:
                max_cost = None
                context_length = None
            
            # Get allowed providers from config
            allowed_providers = self.config.get_list("optimizer.allowed_providers", [])
            preferred_provider = allowed_providers[0] if allowed_providers else None
            
            # Select model for task
            provider, model = self.model_switcher.select_model_for_task(
                task_type=task_type,
                context_length=context_length,
                max_cost=max_cost,
                preferred_provider=preferred_provider
            )
            
            # Cache the result
            self._cached_optimal_model = (provider, model)
            self._last_update_time = current_time
            
            # Format the model ID in the expected format
            return f"{provider}/{model}" if "/" not in model else model
        except Exception as e:
            logger.error(f"Error selecting optimal model: {e}", exc_info=True)
            return current_model
    
    def handle_model_failure(
        self,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Handle a model failure by selecting a fallback model.
        
        Args:
            error: Optional exception that caused the failure
            context: Optional context information
            
        Returns:
            The identifier string of the fallback model, or None if no fallback is available
        """
        try:
            provider, model = self.model_switcher.handle_model_failure(error, context)
            
            # Cache the result
            self._cached_optimal_model = (provider, model)
            self._last_update_time = time.time()
            
            # Format the model ID in the expected format
            return f"{provider}/{model}" if "/" not in model else model
        except Exception as e:
            logger.error(f"Error selecting fallback model: {e}", exc_info=True)
            return None
    
    def reset(self) -> None:
        """Reset the optimizer state."""
        self._cached_optimal_model = None
        self._last_update_time = 0
        self.model_switcher.reset()
    
    def _determine_task_type(self, task_description: Optional[str]) -> str:
        """
        Determine the task type from a description.
        
        Args:
            task_description: Optional description of the task
            
        Returns:
            Task type (chat, code, reasoning, tool_use, math, creative, summarization)
        """
        if not task_description:
            return "chat"  # Default task type
        
        task_description = task_description.lower()
        
        # Simple keyword-based classification
        if any(kw in task_description for kw in ["code", "program", "function", "class", "method"]):
            return "code"
        elif any(kw in task_description for kw in ["reason", "think", "analyze", "logic"]):
            return "reasoning"
        elif any(kw in task_description for kw in ["tool", "execute", "run", "command"]):
            return "tool_use"
        elif any(kw in task_description for kw in ["math", "calculate", "equation", "formula"]):
            return "math"
        elif any(kw in task_description for kw in ["creative", "write", "story", "poem"]):
            return "creative"
        elif any(kw in task_description for kw in ["summarize", "summary", "condense"]):
            return "summarization"
        else:
            return "chat"
