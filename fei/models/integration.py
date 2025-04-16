"""
Integration between the model switching system and the assistant.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging

from fei.models.base import ModelCapability, ModelRequirement, ModelInfo
from fei.models.registry import ModelRegistry
from fei.models.selector import BaseModelSelector
from fei.models.fallback import FallbackStrategy
from fei.models import create_default_selector, create_default_fallback_strategy
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class ModelSwitcher:
    """
    Model switcher for the assistant.
    
    This class integrates the model switching system with the assistant,
    providing methods to select models based on requirements and handle
    fallbacks when models fail.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        selector: Optional[BaseModelSelector] = None,
        fallback_strategy: Optional[FallbackStrategy] = None
    ):
        """
        Initialize the model switcher.
        
        Args:
            config: Optional configuration
            selector: Optional model selector
            fallback_strategy: Optional fallback strategy
        """
        self.config = config or Config()
        self.selector = selector or create_default_selector(self.config)
        self.fallback_strategy = fallback_strategy or create_default_fallback_strategy(self.selector, self.config)
        self.current_model: Optional[Tuple[str, str]] = None
        self.failed_models: List[Tuple[str, str]] = []
    
    def select_model_for_task(
        self,
        task_type: str,
        context_length: Optional[int] = None,
        max_cost: Optional[float] = None,
        preferred_provider: Optional[str] = None,
        preferred_model: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Select a model for a specific task.
        
        Args:
            task_type: Type of task (chat, code, reasoning, etc.)
            context_length: Optional minimum context length required
            max_cost: Optional maximum cost per token
            preferred_provider: Optional preferred provider
            preferred_model: Optional preferred model
            
        Returns:
            Tuple of (provider_name, model_id)
            
        Raises:
            ValueError: If no suitable model is found
        """
        # Map task type to capabilities
        capabilities = self._map_task_to_capabilities(task_type)
        
        # Create requirements
        requirements = ModelRequirement(
            capabilities=capabilities,
            min_context_length=context_length,
            max_cost_per_token=max_cost,
            preferred_provider=preferred_provider,
            preferred_model=preferred_model
        )
        
        # Select model
        provider, model = self.selector.select_model(requirements)
        
        # Store current model
        self.current_model = (provider, model)
        
        logger.info(f"Selected model {model} from provider {provider} for task {task_type}")
        
        return provider, model
    
    def handle_model_failure(
        self,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str]:
        """
        Handle a model failure by selecting a fallback model.
        
        Args:
            error: Optional exception that caused the failure
            context: Optional context information
            
        Returns:
            Tuple of (provider_name, model_id) for the fallback model
            
        Raises:
            ValueError: If no suitable fallback model is found or all models have failed
        """
        if not self.current_model:
            raise ValueError("No current model to fall back from")
        
        # Add current model to failed models
        if self.current_model not in self.failed_models:
            self.failed_models.append(self.current_model)
        
        # Create requirements based on current model
        model_info = self.selector.registry.get_model_info(*self.current_model)
        if not model_info:
            raise ValueError(f"Model info not found for {self.current_model}")
        
        requirements = ModelRequirement(
            capabilities=model_info.capabilities,
            min_context_length=model_info.context_length // 2,  # Relax context length requirement
            max_cost_per_token=None,  # Relax cost requirement
            preferred_provider=None,  # No preferred provider
            preferred_model=None  # No preferred model
        )
        
        # Get fallback model
        try:
            provider, model = self.fallback_strategy.handle_failure(
                self.current_model,
                requirements,
                context,
                error
            )
        except ValueError as e:
            # If all models have failed, raise an error
            if len(self.failed_models) >= len(self.selector.registry.list_models()):
                raise ValueError("All available models have failed") from e
            raise
        
        # Update current model
        self.current_model = (provider, model)
        
        logger.info(f"Switched to fallback model {model} from provider {provider}")
        
        return provider, model
    
    def reset(self) -> None:
        """Reset the model switcher state."""
        self.current_model = None
        self.failed_models = []
    
    def _map_task_to_capabilities(self, task_type: str) -> List[str]:
        """
        Map a task type to required capabilities.
        
        Args:
            task_type: Type of task
            
        Returns:
            List of required capabilities
        """
        # Basic capabilities required for all tasks
        capabilities = [ModelCapability.TEXT_GENERATION, ModelCapability.CHAT]
        
        # Add task-specific capabilities
        if task_type == "chat":
            # Basic chat only requires the basic capabilities
            pass
        elif task_type == "code":
            capabilities.extend([
                ModelCapability.CODE_GENERATION,
                ModelCapability.CODE_UNDERSTANDING,
                ModelCapability.FUNCTION_CALLING
            ])
        elif task_type == "reasoning":
            capabilities.extend([
                ModelCapability.REASONING,
                ModelCapability.PLANNING
            ])
        elif task_type == "tool_use":
            capabilities.extend([
                ModelCapability.FUNCTION_CALLING,
                ModelCapability.TOOL_USE,
                ModelCapability.REASONING
            ])
        elif task_type == "math":
            capabilities.extend([
                ModelCapability.MATH,
                ModelCapability.REASONING
            ])
        elif task_type == "creative":
            capabilities.extend([
                ModelCapability.CREATIVE_WRITING
            ])
        elif task_type == "summarization":
            capabilities.extend([
                ModelCapability.SUMMARIZATION
            ])
        
        return capabilities
