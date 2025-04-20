"""
Path manager for Fei evolution system.

This module provides a manager for evolution paths, allowing
the creation, modification, and validation of evolution paths.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime

from fei.utils.logging import get_logger
from fei.utils.config import Config
from fei.evolution.state_machine import EvolutionPath, EvolutionStateMachine

logger = get_logger(__name__)


class EvolutionPathManager:
    """
    Manager for evolution paths.
    
    This class provides methods for creating, modifying, and
    validating evolution paths.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the evolution path manager.
        
        Args:
            config: Optional configuration
        """
        self.config = config or Config()
        self.state_machine = EvolutionStateMachine(self.config)
        self.manifest_file = self._get_manifest_file_path()
    
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
    
    def get_paths(self) -> Dict[str, EvolutionPath]:
        """
        Get all evolution paths.
        
        Returns:
            Dictionary of path ID to EvolutionPath
        """
        return self.state_machine.paths
    
    def get_path(self, path_id: str) -> Optional[EvolutionPath]:
        """
        Get an evolution path by ID.
        
        Args:
            path_id: Path ID
            
        Returns:
            EvolutionPath instance or None if not found
        """
        return self.state_machine.get_path(path_id)
    
    def create_path(
        self,
        path_id: str,
        name: str,
        description: str,
        initial_stage_id: str = "stage_0",
        initial_stage_name: str = "Initial Bootstrap",
        initial_stage_description: str = "The initial operational state of the agent before any self-evolution.",
        initial_stage_module: str = "fei.evolution.stages.stage_0"
    ) -> Optional[EvolutionPath]:
        """
        Create a new evolution path.
        
        Args:
            path_id: Path ID
            name: Path name
            description: Path description
            initial_stage_id: Initial stage ID
            initial_stage_name: Initial stage name
            initial_stage_description: Initial stage description
            initial_stage_module: Initial stage module
            
        Returns:
            Created EvolutionPath instance or None if creation failed
        """
        # Check if path already exists
        if path_id in self.state_machine.paths:
            logger.error(f"Evolution path already exists: {path_id}")
            return None
        
        # Create initial stage
        stages = {
            initial_stage_id: {
                "name": initial_stage_name,
                "description": initial_stage_description,
                "module": initial_stage_module,
                "validation_criteria": [
                    "Basic task execution works.",
                    "Core tools are functional."
                ]
            }
        }
        
        # Create transitions (initially empty)
        transitions = {
            initial_stage_id: []
        }
        
        # Create path
        path = EvolutionPath(
            path_id=path_id,
            name=name,
            description=description,
            stages=stages,
            transitions=transitions
        )
        
        # Add path to state machine
        self.state_machine.paths[path_id] = path
        
        # Save manifest
        self._save_manifest()
        
        return path
    
    def add_stage(
        self,
        path_id: str,
        stage_id: str,
        name: str,
        description: str,
        module: str,
        validation_criteria: Optional[List[str]] = None
    ) -> bool:
        """
        Add a stage to an evolution path.
        
        Args:
            path_id: Path ID
            stage_id: Stage ID
            name: Stage name
            description: Stage description
            module: Stage module
            validation_criteria: Optional validation criteria
            
        Returns:
            True if successful, False otherwise
        """
        # Get path
        path = self.get_path(path_id)
        if not path:
            logger.error(f"Evolution path not found: {path_id}")
            return False
        
        # Check if stage already exists
        if stage_id in path.stages:
            logger.error(f"Stage already exists in path {path_id}: {stage_id}")
            return False
        
        # Add stage
        path.stages[stage_id] = {
            "name": name,
            "description": description,
            "module": module,
            "validation_criteria": validation_criteria or []
        }
        
        # Initialize transitions for the new stage
        path.transitions[stage_id] = []
        
        # Save manifest
        self._save_manifest()
        
        return True
    
    def add_transition(
        self,
        path_id: str,
        from_stage_id: str,
        to_stage_id: str
    ) -> bool:
        """
        Add a transition between stages in an evolution path.
        
        Args:
            path_id: Path ID
            from_stage_id: Source stage ID
            to_stage_id: Target stage ID
            
        Returns:
            True if successful, False otherwise
        """
        # Get path
        path = self.get_path(path_id)
        if not path:
            logger.error(f"Evolution path not found: {path_id}")
            return False
        
        # Check if stages exist
        if from_stage_id not in path.stages:
            logger.error(f"Source stage not found in path {path_id}: {from_stage_id}")
            return False
        
        if to_stage_id not in path.stages:
            logger.error(f"Target stage not found in path {path_id}: {to_stage_id}")
            return False
        
        # Check if transition already exists
        if to_stage_id in path.transitions.get(from_stage_id, []):
            logger.warning(f"Transition already exists in path {path_id}: {from_stage_id} -> {to_stage_id}")
            return True
        
        # Add transition
        if from_stage_id not in path.transitions:
            path.transitions[from_stage_id] = []
        
        path.transitions[from_stage_id].append(to_stage_id)
        
        # Save manifest
        self._save_manifest()
        
        return True
    
    def remove_transition(
        self,
        path_id: str,
        from_stage_id: str,
        to_stage_id: str
    ) -> bool:
        """
        Remove a transition between stages in an evolution path.
        
        Args:
            path_id: Path ID
            from_stage_id: Source stage ID
            to_stage_id: Target stage ID
            
        Returns:
            True if successful, False otherwise
        """
        # Get path
        path = self.get_path(path_id)
        if not path:
            logger.error(f"Evolution path not found: {path_id}")
            return False
        
        # Check if stages exist
        if from_stage_id not in path.stages:
            logger.error(f"Source stage not found in path {path_id}: {from_stage_id}")
            return False
        
        if to_stage_id not in path.stages:
            logger.error(f"Target stage not found in path {path_id}: {to_stage_id}")
            return False
        
        # Check if transition exists
        if to_stage_id not in path.transitions.get(from_stage_id, []):
            logger.warning(f"Transition does not exist in path {path_id}: {from_stage_id} -> {to_stage_id}")
            return True
        
        # Remove transition
        path.transitions[from_stage_id].remove(to_stage_id)
        
        # Save manifest
        self._save_manifest()
        
        return True
    
    def remove_stage(
        self,
        path_id: str,
        stage_id: str
    ) -> bool:
        """
        Remove a stage from an evolution path.
        
        Args:
            path_id: Path ID
            stage_id: Stage ID
            
        Returns:
            True if successful, False otherwise
        """
        # Get path
        path = self.get_path(path_id)
        if not path:
            logger.error(f"Evolution path not found: {path_id}")
            return False
        
        # Check if stage exists
        if stage_id not in path.stages:
            logger.error(f"Stage not found in path {path_id}: {stage_id}")
            return False
        
        # Check if stage is the initial stage
        if stage_id == path.get_initial_stage():
            logger.error(f"Cannot remove initial stage from path {path_id}: {stage_id}")
            return False
        
        # Remove stage
        del path.stages[stage_id]
        
        # Remove transitions to and from the stage
        if stage_id in path.transitions:
            del path.transitions[stage_id]
        
        for source_stage, targets in path.transitions.items():
            if stage_id in targets:
                path.transitions[source_stage].remove(stage_id)
        
        # Save manifest
        self._save_manifest()
        
        return True
    
    def remove_path(self, path_id: str) -> bool:
        """
        Remove an evolution path.
        
        Args:
            path_id: Path ID
            
        Returns:
            True if successful, False otherwise
        """
        # Check if path exists
        if path_id not in self.state_machine.paths:
            logger.error(f"Evolution path not found: {path_id}")
            return False
        
        # Check if path is the current path
        current_path_id = self.state_machine.get_current_path_id()
        if path_id == current_path_id:
            logger.error(f"Cannot remove current evolution path: {path_id}")
            return False
        
        # Check if path is the default path
        if path_id == "default":
            logger.error("Cannot remove default evolution path")
            return False
        
        # Remove path
        del self.state_machine.paths[path_id]
        
        # Save manifest
        self._save_manifest()
        
        return True
    
    def _save_manifest(self) -> bool:
        """
        Save the manifest file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create manifest data
            manifest = {
                "paths": {}
            }
            
            # Add paths
            for path_id, path in self.state_machine.paths.items():
                manifest["paths"][path_id] = {
                    "name": path.name,
                    "description": path.description,
                    "stages": path.stages,
                    "transitions": path.transitions
                }
            
            # Save manifest
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            logger.debug(f"Saved evolution manifest to {self.manifest_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving evolution manifest: {e}")
            return False
    
    def validate_path(self, path_id: str) -> Tuple[bool, List[str]]:
        """
        Validate an evolution path.
        
        Args:
            path_id: Path ID
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        # Get path
        path = self.get_path(path_id)
        if not path:
            return False, [f"Evolution path not found: {path_id}"]
        
        errors = []
        
        # Check if path has at least one stage
        if not path.stages:
            errors.append(f"Path {path_id} has no stages")
        
        # Check if all stages have required fields
        for stage_id, stage in path.stages.items():
            if "name" not in stage:
                errors.append(f"Stage {stage_id} in path {path_id} has no name")
            
            if "description" not in stage:
                errors.append(f"Stage {stage_id} in path {path_id} has no description")
            
            if "module" not in stage:
                errors.append(f"Stage {stage_id} in path {path_id} has no module")
        
        # Check if all transitions reference valid stages
        for from_stage, to_stages in path.transitions.items():
            if from_stage not in path.stages:
                errors.append(f"Transition from invalid stage in path {path_id}: {from_stage}")
            
            for to_stage in to_stages:
                if to_stage not in path.stages:
                    errors.append(f"Transition to invalid stage in path {path_id}: {to_stage}")
        
        # Check if all stages have transitions defined
        for stage_id in path.stages:
            if stage_id not in path.transitions:
                errors.append(f"Stage {stage_id} in path {path_id} has no transitions defined")
        
        return len(errors) == 0, errors
    
    def get_path_graph(self, path_id: str) -> Dict[str, Any]:
        """
        Get a graph representation of an evolution path.
        
        Args:
            path_id: Path ID
            
        Returns:
            Graph representation of the path
        """
        # Get path
        path = self.get_path(path_id)
        if not path:
            return {"nodes": [], "edges": []}
        
        # Create nodes
        nodes = [
            {
                "id": stage_id,
                "label": stage.get("name", stage_id),
                "description": stage.get("description", ""),
                "module": stage.get("module", "")
            }
            for stage_id, stage in path.stages.items()
        ]
        
        # Create edges
        edges = []
        for from_stage, to_stages in path.transitions.items():
            for to_stage in to_stages:
                edges.append({
                    "from": from_stage,
                    "to": to_stage
                })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
