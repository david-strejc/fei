"""
State machine for Fei evolution system.

This module provides a more robust state machine implementation
for the evolution system, supporting multiple evolution paths.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.evolution.state import EvolutionState

logger = get_logger(__name__)


class EvolutionPath:
    """Represents an evolution path with stages and transitions."""
    
    def __init__(
        self,
        path_id: str,
        name: str,
        description: str,
        stages: Dict[str, Dict[str, Any]],
        transitions: Dict[str, List[str]]
    ):
        """
        Initialize an evolution path.
        
        Args:
            path_id: Path ID
            name: Path name
            description: Path description
            stages: Dictionary of stages
            transitions: Dictionary of allowed transitions
        """
        self.path_id = path_id
        self.name = name
        self.description = description
        self.stages = stages
        self.transitions = transitions
    
    def get_stage(self, stage_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a stage by ID.
        
        Args:
            stage_id: Stage ID
            
        Returns:
            Stage information or None if not found
        """
        return self.stages.get(stage_id)
    
    def get_allowed_transitions(self, stage_id: str) -> List[str]:
        """
        Get allowed transitions from a stage.
        
        Args:
            stage_id: Stage ID
            
        Returns:
            List of allowed transition stage IDs
        """
        return self.transitions.get(stage_id, [])
    
    def can_transition(self, from_stage: str, to_stage: str) -> bool:
        """
        Check if a transition is allowed.
        
        Args:
            from_stage: Source stage ID
            to_stage: Target stage ID
            
        Returns:
            True if transition is allowed, False otherwise
        """
        return to_stage in self.get_allowed_transitions(from_stage)
    
    def get_initial_stage(self) -> str:
        """
        Get the initial stage ID.
        
        Returns:
            Initial stage ID
        """
        # The first stage in the path is the initial stage
        if not self.stages:
            return ""
        
        return next(iter(self.stages))
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert path to dictionary.
        
        Returns:
            Dictionary representation of the path
        """
        return {
            "path_id": self.path_id,
            "name": self.name,
            "description": self.description,
            "stages": self.stages,
            "transitions": self.transitions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvolutionPath':
        """
        Create a path from dictionary.
        
        Args:
            data: Dictionary representation of the path
            
        Returns:
            EvolutionPath instance
        """
        return cls(
            path_id=data.get("path_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            stages=data.get("stages", {}),
            transitions=data.get("transitions", {})
        )


class EvolutionStateMachine:
    """
    Enhanced state machine for evolution system.
    
    This class provides a more robust state machine implementation
    for the evolution system, supporting multiple evolution paths.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the evolution state machine.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.state_file = self._get_state_file_path()
        self.manifest_file = self._get_manifest_file_path()
        self.paths: Dict[str, EvolutionPath] = {}
        
        # Load paths from manifest
        self._load_paths()
    
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
    
    def _get_manifest_file_path(self) -> str:
        """
        Get the path to the manifest file.
        
        Returns:
            Path to the manifest file
        """
        # Get manifest path from config or use default
        manifest_path = self.config.get("evolution_manifest")
        if manifest_path:
            return manifest_path
        
        # Use default path in package directory
        return os.path.join(os.path.dirname(__file__), "manifest.json")
    
    def _load_paths(self) -> None:
        """Load evolution paths from manifest."""
        try:
            if not os.path.exists(self.manifest_file):
                logger.warning(f"Manifest file not found at {self.manifest_file}")
                # Create default path
                self._create_default_path()
                return
            
            with open(self.manifest_file, 'r') as f:
                manifest = json.load(f)
            
            # Load paths
            paths_data = manifest.get("paths", {})
            for path_id, path_data in paths_data.items():
                self.paths[path_id] = EvolutionPath.from_dict({
                    "path_id": path_id,
                    **path_data
                })
            
            logger.info(f"Loaded {len(self.paths)} evolution paths from manifest")
            
            # If no paths were loaded, create default path
            if not self.paths:
                self._create_default_path()
        except Exception as e:
            logger.error(f"Error loading evolution paths: {e}")
            # Create default path
            self._create_default_path()
    
    def _create_default_path(self) -> None:
        """Create default evolution path."""
        # Create default path with stage_0
        default_path = EvolutionPath(
            path_id="default",
            name="Default Evolution Path",
            description="Default evolution path with initial bootstrap stage",
            stages={
                "stage_0": {
                    "name": "Initial Bootstrap",
                    "description": "The initial operational state of the agent before any self-evolution.",
                    "module": "fei.evolution.stages.stage_0",
                    "validation_criteria": [
                        "Basic task execution works.",
                        "Core tools are functional."
                    ]
                }
            },
            transitions={
                "stage_0": []
            }
        )
        
        self.paths["default"] = default_path
        logger.info("Created default evolution path")
    
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
            metadata={
                "created_at": datetime.now().isoformat(),
                "path_id": "default"
            }
        )
    
    def save_state(self, state: EvolutionState) -> bool:
        """
        Save the evolution state.
        
        Args:
            state: Evolution state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            # Save state
            with open(self.state_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            logger.debug(f"Saved evolution state to {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving evolution state: {e}")
            return False
    
    def get_path(self, path_id: str) -> Optional[EvolutionPath]:
        """
        Get an evolution path by ID.
        
        Args:
            path_id: Path ID
            
        Returns:
            EvolutionPath instance or None if not found
        """
        return self.paths.get(path_id)
    
    def get_current_path_id(self) -> str:
        """
        Get the current evolution path ID.
        
        Returns:
            Current path ID
        """
        state = self.get_current_state()
        return state.metadata.get("path_id", "default")
    
    def get_current_path(self) -> EvolutionPath:
        """
        Get the current evolution path.
        
        Returns:
            Current EvolutionPath instance
        """
        path_id = self.get_current_path_id()
        path = self.get_path(path_id)
        
        if not path:
            # Fall back to default path
            path = self.get_path("default")
            if not path:
                # Create default path if it doesn't exist
                self._create_default_path()
                path = self.get_path("default")
        
        return path
    
    def get_current_stage_id(self) -> str:
        """
        Get the current stage ID.
        
        Returns:
            Current stage ID
        """
        state = self.get_current_state()
        return state.stage_id
    
    def get_current_stage(self) -> Optional[Dict[str, Any]]:
        """
        Get the current stage.
        
        Returns:
            Current stage information or None if not found
        """
        path = self.get_current_path()
        stage_id = self.get_current_stage_id()
        return path.get_stage(stage_id)
    
    def get_allowed_transitions(self) -> List[Dict[str, Any]]:
        """
        Get allowed transitions from the current stage.
        
        Returns:
            List of allowed transition stages
        """
        path = self.get_current_path()
        stage_id = self.get_current_stage_id()
        allowed_stage_ids = path.get_allowed_transitions(stage_id)
        
        return [
            {
                "stage_id": stage_id,
                "name": path.get_stage(stage_id).get("name", ""),
                "description": path.get_stage(stage_id).get("description", "")
            }
            for stage_id in allowed_stage_ids
            if path.get_stage(stage_id)
        ]
    
    def can_transition_to(self, target_stage_id: str) -> bool:
        """
        Check if a transition to the target stage is allowed.
        
        Args:
            target_stage_id: Target stage ID
            
        Returns:
            True if transition is allowed, False otherwise
        """
        path = self.get_current_path()
        current_stage_id = self.get_current_stage_id()
        return path.can_transition(current_stage_id, target_stage_id)
    
    def transition_to_stage(
        self,
        target_stage_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to a new stage.
        
        Args:
            target_stage_id: Target stage ID
            metadata: Optional transition metadata
            
        Returns:
            True if transition was successful, False otherwise
        """
        # Get current state
        current_state = self.get_current_state()
        current_stage_id = current_state.stage_id
        
        # Get current path
        path = self.get_current_path()
        
        # Check if transition is valid
        if not path.can_transition(current_stage_id, target_stage_id):
            logger.error(f"Invalid transition from {current_stage_id} to {target_stage_id}")
            return False
        
        # Create transition record
        transition = {
            "from_stage": current_stage_id,
            "to_stage": target_stage_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Update state
        current_state.transitions.append(transition)
        current_state.stage_id = target_stage_id
        current_state.metadata["last_updated"] = datetime.now().isoformat()
        
        # Save updated state
        return self.save_state(current_state)
    
    def switch_path(
        self,
        target_path_id: str,
        initial_stage_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Switch to a different evolution path.
        
        Args:
            target_path_id: Target path ID
            initial_stage_id: Optional initial stage ID (defaults to path's initial stage)
            metadata: Optional transition metadata
            
        Returns:
            True if switch was successful, False otherwise
        """
        # Check if path exists
        target_path = self.get_path(target_path_id)
        if not target_path:
            logger.error(f"Evolution path not found: {target_path_id}")
            return False
        
        # Get current state
        current_state = self.get_current_state()
        current_path_id = current_state.metadata.get("path_id", "default")
        current_stage_id = current_state.stage_id
        
        # Determine initial stage
        if not initial_stage_id:
            initial_stage_id = target_path.get_initial_stage()
        
        # Check if initial stage exists
        if not target_path.get_stage(initial_stage_id):
            logger.error(f"Stage not found in path {target_path_id}: {initial_stage_id}")
            return False
        
        # Create transition record
        transition = {
            "from_path": current_path_id,
            "to_path": target_path_id,
            "from_stage": current_stage_id,
            "to_stage": initial_stage_id,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        # Update state
        current_state.transitions.append(transition)
        current_state.stage_id = initial_stage_id
        current_state.metadata["path_id"] = target_path_id
        current_state.metadata["last_updated"] = datetime.now().isoformat()
        
        # Save updated state
        return self.save_state(current_state)
    
    def get_available_paths(self) -> List[Dict[str, Any]]:
        """
        Get all available evolution paths.
        
        Returns:
            List of path information
        """
        return [
            {
                "path_id": path.path_id,
                "name": path.name,
                "description": path.description,
                "stage_count": len(path.stages)
            }
            for path in self.paths.values()
        ]
    
    def get_path_stages(self, path_id: str) -> List[Dict[str, Any]]:
        """
        Get all stages in a path.
        
        Args:
            path_id: Path ID
            
        Returns:
            List of stage information
        """
        path = self.get_path(path_id)
        if not path:
            return []
        
        return [
            {
                "stage_id": stage_id,
                "name": stage_info.get("name", ""),
                "description": stage_info.get("description", ""),
                "module": stage_info.get("module", ""),
                "validation_criteria": stage_info.get("validation_criteria", [])
            }
            for stage_id, stage_info in path.stages.items()
        ]
    
    def get_transition_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get transition history.
        
        Args:
            limit: Maximum number of transitions to return
            
        Returns:
            List of transition records
        """
        state = self.get_current_state()
        transitions = state.transitions
        
        # Sort by timestamp (newest first)
        sorted_transitions = sorted(
            transitions,
            key=lambda t: t.get("timestamp", ""),
            reverse=True
        )
        
        return sorted_transitions[:limit]
