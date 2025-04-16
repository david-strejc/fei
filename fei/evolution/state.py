"""
Evolution state management for Fei.
"""

from typing import Dict, List, Optional, Any, Union
import json
import os
import logging
from datetime import datetime

from fei.utils.config import Config

logger = logging.getLogger(__name__)


class EvolutionState:
    """Represents the current state of the evolution system."""
    
    def __init__(self, stage_id: str, transitions: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """
        Initialize the evolution state.
        
        Args:
            stage_id: Current stage ID
            transitions: List of transition records
            metadata: Additional metadata
        """
        self.stage_id = stage_id
        self.transitions = transitions
        self.metadata = metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.
        
        Returns:
            Dictionary representation of the state
        """
        return {
            "stage_id": self.stage_id,
            "transitions": self.transitions,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionState':
        """
        Create state from dictionary.
        
        Args:
            data: Dictionary representation of the state
            
        Returns:
            EvolutionState instance
        """
        return cls(
            stage_id=data.get("stage_id", "stage_0"),
            transitions=data.get("transitions", []),
            metadata=data.get("metadata", {})
        )


class EvolutionStateManager:
    """Manager for evolution state."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the evolution state manager.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.state_file = self._get_state_file_path()
    
    def _get_state_file_path(self) -> str:
        """
        Get the path to the state file.
        
        Returns:
            Path to the state file
        """
        # Get state file path from config or use default
        state_file = self.config.get("evolution_state_file")
        if state_file:
            return state_file
        
        # Use default path in user's home directory
        home_dir = os.path.expanduser("~")
        fei_dir = os.path.join(home_dir, ".fei")
        os.makedirs(fei_dir, exist_ok=True)
        return os.path.join(fei_dir, "evolution_state.json")
    
    def get_current_state(self) -> EvolutionState:
        """
        Get the current evolution state.
        
        Returns:
            Current evolution state
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return EvolutionState.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading evolution state: {e}")
        
        # Return default state if file doesn't exist or is invalid
        return EvolutionState(
            stage_id="stage_0",
            transitions=[],
            metadata={"created_at": datetime.now().isoformat()}
        )
    
    def save_state(self, state: EvolutionState) -> bool:
        """
        Save the evolution state.
        
        Args:
            state: Evolution state to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving evolution state: {e}")
            return False
    
    def transition_to_stage(self, stage_id: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transition to a new stage.
        
        Args:
            stage_id: Target stage ID
            metadata: Optional transition metadata
            
        Returns:
            True if transition was successful, False otherwise
        """
        # Get current state
        current_state = self.get_current_state()
        
        # Check if transition is valid
        if not self.can_transition_to(current_state.stage_id, stage_id):
            logger.error(f"Invalid transition from {current_state.stage_id} to {stage_id}")
            return False
        
        # Create transition record
        transition = {
            "from_stage": current_state.stage_id,
            "to_stage": stage_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Update state
        current_state.transitions.append(transition)
        current_state.stage_id = stage_id
        current_state.metadata["last_updated"] = datetime.now().isoformat()
        
        # Save updated state
        return self.save_state(current_state)
    
    def can_transition_to(self, from_stage: str, to_stage: str) -> bool:
        """
        Check if a transition is valid.
        
        Args:
            from_stage: Source stage ID
            to_stage: Target stage ID
            
        Returns:
            True if transition is valid, False otherwise
        """
        # Load manifest to check valid transitions
        manifest = self._load_manifest()
        
        # Get allowed transitions for the source stage
        stage_info = manifest.get("stages", {}).get(from_stage, {})
        allowed_transitions = stage_info.get("allowed_transitions", [])
        
        # Check if target stage is in allowed transitions
        return to_stage in allowed_transitions
    
    def _load_manifest(self) -> Dict[str, Any]:
        """
        Load the evolution manifest.
        
        Returns:
            Manifest data
        """
        try:
            # Get manifest path from config or use default
            manifest_path = self.config.get("evolution_manifest")
            if not manifest_path:
                # Use default path in package directory
                import fei.evolution
                package_dir = os.path.dirname(fei.evolution.__file__)
                manifest_path = os.path.join(package_dir, "manifest.json")
            
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading evolution manifest: {e}")
            return {"stages": {"stage_0": {"allowed_transitions": []}}}
    
    def get_available_stages(self) -> List[str]:
        """
        Get a list of all available stages.
        
        Returns:
            List of stage IDs
        """
        manifest = self._load_manifest()
        return list(manifest.get("stages", {}).keys())
    
    def get_stage_info(self, stage_id: str) -> Dict[str, Any]:
        """
        Get information about a stage.
        
        Args:
            stage_id: Stage ID
            
        Returns:
            Stage information
        """
        manifest = self._load_manifest()
        return manifest.get("stages", {}).get(stage_id, {})
