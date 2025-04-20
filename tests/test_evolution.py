"""
Tests for the evolution system
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil

from fei.evolution.state import EvolutionState, EvolutionStateManager
from fei.evolution.loader import StageLoader
from fei.utils.config import Config


class TestEvolutionSystem(unittest.TestCase):
    """Test the evolution system"""
    
    def test_evolution_state(self):
        """Test evolution state"""
        # Create a state
        state = EvolutionState(
            stage_id="stage_1",
            transitions=[
                {
                    "from_stage": "stage_0",
                    "to_stage": "stage_1",
                    "timestamp": "2023-01-01T00:00:00",
                    "metadata": {"reason": "test"}
                }
            ],
            metadata={"last_updated": "2023-01-01T00:00:00"}
        )
        
        # Convert to dictionary
        state_dict = state.to_dict()
        self.assertEqual(state_dict["stage_id"], "stage_1")
        self.assertEqual(len(state_dict["transitions"]), 1)
        self.assertEqual(state_dict["metadata"]["last_updated"], "2023-01-01T00:00:00")
        
        # Create from dictionary
        new_state = EvolutionState.from_dict(state_dict)
        self.assertEqual(new_state.stage_id, "stage_1")
        self.assertEqual(len(new_state.transitions), 1)
        self.assertEqual(new_state.metadata["last_updated"], "2023-01-01T00:00:00")
    
    def test_evolution_state_manager(self):
        """Test evolution state manager"""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a temporary state file
            state_file = os.path.join(temp_dir, "evolution_state.json")
            
            # Create a state
            state = EvolutionState(
                stage_id="stage_1",
                transitions=[
                    {
                        "from_stage": "stage_0",
                        "to_stage": "stage_1",
                        "timestamp": "2023-01-01T00:00:00",
                        "metadata": {"reason": "test"}
                    }
                ],
                metadata={"last_updated": "2023-01-01T00:00:00"}
            )
            
            # Write state to file
            with open(state_file, "w") as f:
                json.dump(state.to_dict(), f)
            
            # Create a config with the state file
            config = Config()
            config.set("evolution_state_file", state_file)
            
            # Create a state manager
            manager = EvolutionStateManager(config)
            
            # Get current state
            current_state = manager.get_current_state()
            self.assertEqual(current_state.stage_id, "stage_1")
            
            # Transition to a new stage
            with patch.object(manager, "_load_manifest") as mock_load_manifest:
                # Mock the manifest to allow transition
                mock_load_manifest.return_value = {
                    "stages": {
                        "stage_1": {
                            "allowed_transitions": ["stage_2"]
                        },
                        "stage_2": {
                            "allowed_transitions": ["stage_1"]
                        }
                    }
                }
                
                # Transition to stage_2
                result = manager.transition_to_stage("stage_2", {"reason": "test"})
                self.assertTrue(result)
                
                # Get updated state
                updated_state = manager.get_current_state()
                self.assertEqual(updated_state.stage_id, "stage_2")
                self.assertEqual(len(updated_state.transitions), 2)
                self.assertEqual(updated_state.transitions[1]["from_stage"], "stage_1")
                self.assertEqual(updated_state.transitions[1]["to_stage"], "stage_2")
                self.assertEqual(updated_state.transitions[1]["metadata"]["reason"], "test")
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)
    
    def test_stage_loader(self):
        """Test stage loader"""
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create a temporary manifest file
            manifest_file = os.path.join(temp_dir, "manifest.json")
            
            # Create a manifest
            manifest = {
                "stages": {
                    "stage_0": {
                        "name": "Initial Bootstrap",
                        "description": "Initial stage",
                        "module": "fei.evolution.stages.stage_0",
                        "allowed_transitions": ["stage_1"]
                    },
                    "stage_1": {
                        "name": "Basic Self-Improvement",
                        "description": "First evolution stage",
                        "module": "fei.evolution.stages.stage_1",
                        "allowed_transitions": ["stage_0", "stage_2"]
                    }
                }
            }
            
            # Write manifest to file
            with open(manifest_file, "w") as f:
                json.dump(manifest, f)
            
            # Create a state file
            state_file = os.path.join(temp_dir, "evolution_state.json")
            
            # Create a state
            state = EvolutionState(
                stage_id="stage_0",
                transitions=[],
                metadata={"created_at": "2023-01-01T00:00:00"}
            )
            
            # Write state to file
            with open(state_file, "w") as f:
                json.dump(state.to_dict(), f)
            
            # Create a config with the manifest and state file
            config = Config()
            config.set("evolution_manifest", manifest_file)
            config.set("evolution_state_file", state_file)
            
            # Create a stage loader
            loader = StageLoader(config)
            
            # Get current stage ID
            stage_id = loader.get_current_stage_id()
            self.assertEqual(stage_id, "stage_0")
            
            # Get stage info
            stage_info = loader.get_stage_info("stage_0")
            self.assertEqual(stage_info["name"], "Initial Bootstrap")
            
            # Get available transitions
            with patch.object(loader, "load_stage_module") as mock_load_module:
                # Mock the stage module loading
                mock_module = MagicMock()
                mock_load_module.return_value = mock_module
                
                # Mock the apply_stage_modifications function
                mock_module.apply_stage_modifications = MagicMock()
                
                # Mock assistant and tool registry
                mock_assistant = MagicMock()
                mock_tool_registry = MagicMock()
                
                # Transition to stage_1
                result = loader.transition_to_stage("stage_1", mock_assistant, mock_tool_registry)
                self.assertTrue(result)
                
                # Check if apply_stage_modifications was called
                mock_module.apply_stage_modifications.assert_called_once_with(mock_assistant, mock_tool_registry)
                
                # Get updated stage ID
                updated_stage_id = loader.get_current_stage_id()
                self.assertEqual(updated_stage_id, "stage_1")
        
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
