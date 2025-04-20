"""
Enhanced stage loader for Fei evolution system.
"""

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, List
import logging

from fei.utils.logging import get_logger
from fei.evolution.state import EvolutionStateManager
from fei.utils.config import Config

logger = get_logger(__name__)


class StageLoader:
    """Loader for evolution stages."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the stage loader.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.state_manager = EvolutionStateManager(self.config)
        self.manifest_path = self._get_manifest_path()
    
    def _get_manifest_path(self) -> Path:
        """
        Get the path to the manifest file.
        
        Returns:
            Path to the manifest file
        """
        # Get manifest path from config or use default
        manifest_path = self.config.get("evolution_manifest")
        if manifest_path:
            return Path(manifest_path)
        
        # Use default path in package directory
        return Path(__file__).parent / "manifest.json"
    
    def load_manifest(self) -> Dict[str, Any]:
        """
        Load the evolution manifest.
        
        Returns:
            Manifest data
            
        Raises:
            FileNotFoundError: If the manifest file is not found
            json.JSONDecodeError: If the manifest file is invalid JSON
        """
        if not self.manifest_path.exists():
            logger.error(f"Evolution manifest not found at {self.manifest_path}")
            raise FileNotFoundError(f"Evolution manifest not found at {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            return json.load(f)
    
    def get_current_stage_id(self) -> str:
        """
        Get the current stage ID.
        
        Returns:
            Current stage ID
        """
        # Get current state from state manager
        current_state = self.state_manager.get_current_state()
        return current_state.stage_id
    
    def get_stage_info(self, stage_id: str) -> Dict[str, Any]:
        """
        Get information about a stage.
        
        Args:
            stage_id: Stage ID
            
        Returns:
            Stage information
            
        Raises:
            KeyError: If the stage ID is not found in the manifest
        """
        manifest = self.load_manifest()
        stages = manifest.get("stages", {})
        
        if stage_id not in stages:
            logger.error(f"Stage ID {stage_id} not found in manifest.")
            raise KeyError(f"Stage ID {stage_id} not found in manifest.")
        
        return stages[stage_id]
    
    def load_stage_module(self, stage_id: str) -> Any:
        """
        Load a stage module.
        
        Args:
            stage_id: Stage ID
            
        Returns:
            Stage module
            
        Raises:
            KeyError: If the stage ID is not found in the manifest
            ImportError: If the stage module cannot be imported
        """
        stage_info = self.get_stage_info(stage_id)
        
        module_path = stage_info.get("module")
        if not module_path:
            logger.error(f"Module path not defined for stage ID {stage_id}.")
            raise KeyError(f"Module path not defined for stage ID {stage_id}.")
        
        logger.info(f"Importing stage module: {module_path}")
        try:
            return importlib.import_module(module_path)
        except ImportError as e:
            logger.error(f"Failed to import stage module {module_path}: {e}")
            raise
    
    def apply_stage(self, stage_id: str, assistant: Any, tool_registry: Any) -> bool:
        """
        Apply a stage to the assistant and tool registry.
        
        Args:
            stage_id: Stage ID
            assistant: Assistant instance
            tool_registry: Tool registry instance
            
        Returns:
            True if stage was applied successfully, False otherwise
            
        Raises:
            KeyError: If the stage ID is not found in the manifest
            ImportError: If the stage module cannot be imported
            AttributeError: If the stage module lacks apply_stage_modifications
        """
        try:
            stage_module = self.load_stage_module(stage_id)
            
            apply_func = getattr(stage_module, "apply_stage_modifications", None)
            if not callable(apply_func):
                logger.error(f"Function 'apply_stage_modifications' not found or not callable in stage {stage_id}.")
                raise AttributeError(f"Function 'apply_stage_modifications' not found or not callable in stage {stage_id}.")
            
            logger.info(f"Applying modifications for stage {stage_id}...")
            apply_func(assistant, tool_registry)
            logger.info(f"Successfully applied stage {stage_id}.")
            
            return True
        except (KeyError, ImportError, AttributeError) as e:
            logger.error(f"Failed to apply stage {stage_id}: {e}")
            return False
    
    def load_current_stage(self, assistant: Any, tool_registry: Any) -> str:
        """
        Load the current stage.
        
        Args:
            assistant: Assistant instance
            tool_registry: Tool registry instance
            
        Returns:
            Current stage ID
            
        Raises:
            Exception: If the stage cannot be loaded
        """
        logger.info("Loading current evolution stage...")
        
        try:
            # Get current stage ID
            current_stage_id = self.get_current_stage_id()
            logger.info(f"Current stage ID: {current_stage_id}")
            
            # Apply stage
            success = self.apply_stage(current_stage_id, assistant, tool_registry)
            if not success:
                logger.error(f"Failed to apply stage {current_stage_id}")
                raise Exception(f"Failed to apply stage {current_stage_id}")
            
            return current_stage_id
        except Exception as e:
            logger.exception(f"An error occurred during stage loading: {e}")
            
            # Try to fall back to stage_0
            if self.get_current_stage_id() != "stage_0":
                logger.info("Falling back to stage_0...")
                try:
                    success = self.apply_stage("stage_0", assistant, tool_registry)
                    if success:
                        logger.info("Successfully fell back to stage_0")
                        # Update state to reflect fallback
                        self.state_manager.transition_to_stage("stage_0", {"reason": "fallback", "error": str(e)})
                        return "stage_0"
                except Exception as fallback_error:
                    logger.exception(f"Failed to fall back to stage_0: {fallback_error}")
            
            raise
    
    def transition_to_stage(self, stage_id: str, assistant: Any, tool_registry: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transition to a new stage.
        
        Args:
            stage_id: Target stage ID
            assistant: Assistant instance
            tool_registry: Tool registry instance
            metadata: Optional transition metadata
            
        Returns:
            True if transition was successful, False otherwise
        """
        logger.info(f"Transitioning to stage {stage_id}...")
        
        # Check if transition is valid
        current_stage_id = self.get_current_stage_id()
        if not self.state_manager.can_transition_to(current_stage_id, stage_id):
            logger.error(f"Invalid transition from {current_stage_id} to {stage_id}")
            return False
        
        # Apply new stage
        try:
            success = self.apply_stage(stage_id, assistant, tool_registry)
            if not success:
                logger.error(f"Failed to apply stage {stage_id}")
                return False
            
            # Update state
            return self.state_manager.transition_to_stage(stage_id, metadata)
        except Exception as e:
            logger.exception(f"Error during transition to stage {stage_id}: {e}")
            return False
    
    def get_available_transitions(self) -> List[Dict[str, Any]]:
        """
        Get available transitions from the current stage.
        
        Returns:
            List of available transitions with stage information
        """
        current_stage_id = self.get_current_stage_id()
        current_stage_info = self.get_stage_info(current_stage_id)
        allowed_transitions = current_stage_info.get("allowed_transitions", [])
        
        result = []
        for stage_id in allowed_transitions:
            try:
                stage_info = self.get_stage_info(stage_id)
                result.append({
                    "stage_id": stage_id,
                    "name": stage_info.get("name", stage_id),
                    "description": stage_info.get("description", "")
                })
            except KeyError:
                logger.warning(f"Stage ID {stage_id} referenced in transitions but not found in manifest")
        
        return result
