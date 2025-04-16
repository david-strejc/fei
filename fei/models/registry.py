"""
Model registry for storing and retrieving model information.
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import logging

from fei.models.base import ModelInfo, ModelRequirement
from fei.utils.config import Config

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model information."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the model registry.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.models: Dict[Tuple[str, str], ModelInfo] = {}
        self._load_models_from_config()
    
    def _load_models_from_config(self) -> None:
        """Load models from configuration."""
        models_config = self.config.get("models", {})
        
        for provider_name, provider_models in models_config.items():
            for model_id, model_config in provider_models.items():
                try:
                    model_info = ModelInfo(
                        model_id=model_id,
                        provider=provider_name,
                        capabilities=model_config.get("capabilities", []),
                        context_length=model_config.get("context_length", 4096),
                        cost_per_input_token=model_config.get("cost_per_input_token", 0.0),
                        cost_per_output_token=model_config.get("cost_per_output_token", 0.0),
                        reliability=model_config.get("reliability", 0.95),
                        metadata=model_config.get("metadata", {})
                    )
                    self.register_model(model_info)
                except Exception as e:
                    logger.error(f"Error loading model {model_id} from provider {provider_name}: {e}")
    
    def register_model(self, model_info: ModelInfo) -> None:
        """
        Register a model.
        
        Args:
            model_info: Model information
        """
        key = (model_info.provider, model_info.model_id)
        self.models[key] = model_info
        logger.debug(f"Registered model {model_info.model_id} from provider {model_info.provider}")
    
    def unregister_model(self, provider: str, model_id: str) -> bool:
        """
        Unregister a model.
        
        Args:
            provider: Provider name
            model_id: Model ID
            
        Returns:
            True if the model was unregistered, False if it wasn't registered
        """
        key = (provider, model_id)
        if key in self.models:
            del self.models[key]
            logger.debug(f"Unregistered model {model_id} from provider {provider}")
            return True
        return False
    
    def get_model_info(self, provider: str, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a model.
        
        Args:
            provider: Provider name
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        key = (provider, model_id)
        return self.models.get(key)
    
    def list_models(self) -> List[ModelInfo]:
        """
        List all registered models.
        
        Returns:
            List of model information
        """
        return list(self.models.values())
    
    def list_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """
        List all models from a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of model information
        """
        return [model for key, model in self.models.items() if key[0] == provider]
    
    def find_models(self, requirements: ModelRequirement) -> List[ModelInfo]:
        """
        Find models that meet the given requirements.
        
        Args:
            requirements: Model requirements
            
        Returns:
            List of model information for models that meet the requirements
        """
        return [
            model for model in self.models.values()
            if model.meets_requirements(requirements)
        ]
    
    def save_to_config(self) -> None:
        """Save the model registry to configuration."""
        models_config = {}
        
        for (provider, model_id), model_info in self.models.items():
            if provider not in models_config:
                models_config[provider] = {}
            
            models_config[provider][model_id] = {
                "capabilities": model_info.capabilities,
                "context_length": model_info.context_length,
                "cost_per_input_token": model_info.cost_per_input_token,
                "cost_per_output_token": model_info.cost_per_output_token,
                "reliability": model_info.reliability,
                "metadata": model_info.metadata
            }
        
        self.config.set("models", models_config)
        self.config.save()
        logger.debug("Saved model registry to configuration")
