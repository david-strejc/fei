"""
Tests for the evolution state system.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil
from datetime import datetime

from fei.evolution.state import EvolutionState, EvolutionStateManager
from fei.utils.config import Config


class TestEvolutionState:
    """Test the evolution state."""
    
    def test_init(self):
        """Test initialization."""
        # Create state
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
        
        # Check if state is initialized correctly
        assert state.stage_id == "stage_1"
        assert len(state.transitions) == 1
        assert state.transitions[0]["from_stage"] == "stage_0"
        assert state.transitions[0]["to_stage"] == "stage_1"
        assert state.metadata["last_updated"] == "2023-01-01T00:00:00"
    
    def test_to_dict(self):
        """Test to_dict."""
        # Create state
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
        
        # Check if dictionary is correct
        assert state_dict["stage_id"] == "stage_1"
        assert len(state_dict["transitions"]) == 1
        assert state_dict["transitions"][0]["from_stage"] == "stage_0"
        assert state_dict["transitions"][0]["to_stage"] == "stage_1"
        assert state_dict["metadata"]["last_updated"] == "2023-01-01T00:00:00"
    
    def test_from_dict(self):
        """Test from_dict."""
        # Create dictionary
        state_dict = {
            "stage_id": "stage_1",
            "transitions": [
                {
                    "from_stage": "stage_0",
                    "to_stage": "stage_1",
                    "timestamp": "2023-01-01T00:00:00",
                    "metadata": {"reason": "test"}
                }
            ],
            "metadata": {"last_updated": "2023-01-01T00:00:00"}
        }
        
        # Create state from dictionary
        state = EvolutionState.from_dict(state_dict)
        
        # Check if state is correct
        assert state.stage_id == "stage_1"
        assert len(state.transitions) == 1
        assert state.transitions[0]["from_stage"] == "stage_0"
        assert state.transitions[0]["to_stage"] == "stage_1"
        assert state.metadata["last_updated"] == "2023-01-01T00:00:00"
    
    def test_from_dict_with_defaults(self):
        """Test from_dict with defaults."""
        # Create state from empty dictionary
        state = EvolutionState.from_dict({})
        
        # Check if state has default values
        assert state.stage_id == "stage_0"
        assert state.transitions == []
        assert state.metadata == {}


class TestEvolutionStateManager:
    """Test the evolution state manager."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a config with the temporary directory
        self.config = Config()
        self.config.set("evolution_state_file", os.path.join(self.temp_dir, "evolution_state.json"))
        
        # Create a state manager
        self.manager = EvolutionStateManager(self.config)
    
    def teardown_method(self):
        """Tear down the test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        # Check if manager is initialized correctly
        assert self.manager.config is self.config
        assert self.manager.state_file == os.path.join(self.temp_dir, "evolution_state.json")
    
    def test_get_current_state_no_file(self):
        """Test get_current_state with no file."""
        # Get current state
        state = self.manager.get_current_state()
        
        # Check if state has default values
        assert state.stage_id == "stage_0"
        assert state.transitions == []
        assert "created_at" in state.metadata
    
    def test_get_current_state_with_file(self):
        """Test get_current_state with file."""
        # Create a state file
        state_dict = {
            "stage_id": "stage_1",
            "transitions": [
                {
                    "from_stage": "stage_0",
                    "to_stage": "stage_1",
                    "timestamp": "2023-01-01T00:00:00",
                    "metadata": {"reason": "test"}
                }
            ],
            "metadata": {"last_updated": "2023-01-01T00:00:00"}
        }
        
        with open(self.manager.state_file, "w") as f:
            json.dump(state_dict, f)
        
        # Get current state
        state = self.manager.get_current_state()
        
        # Check if state is correct
        assert state.stage_id == "stage_1"
        assert len(state.transitions) == 1
        assert state.transitions[0]["from_stage"] == "stage_0"
        assert state.transitions[0]["to_stage"] == "stage_1"
        assert state.metadata["last_updated"] == "2023-01-01T00:00:00"
    
    def test_save_state(self):
        """Test save_state."""
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
        
        # Save state
        result = self.manager.save_state(state)
        
        # Check if save was successful
        assert result is True
        
        # Check if file was created
        assert os.path.exists(self.manager.state_file)
        
        # Check if file contains correct data
        with open(self.manager.state_file, "r") as f:
            state_dict = json.load(f)
            assert state_dict["stage_id"] == "stage_1"
            assert len(state_dict["transitions"]) == 1
            assert state_dict["transitions"][0]["from_stage"] == "stage_0"
            assert state_dict["transitions"][0]["to_stage"] == "stage_1"
            assert state_dict["metadata"]["last_updated"] == "2023-01-01T00:00:00"
    
    def test_transition_to_stage(self):
        """Test transition_to_stage."""
        # Create a state file
        state_dict = {
            "stage_id": "stage_0",
            "transitions": [],
            "metadata": {"created_at": "2023-01-01T00:00:00"}
        }
        
        with open(self.manager.state_file, "w") as f:
            json.dump(state_dict, f)
        
        # Mock _load_manifest to allow transition
        with patch.object(self.manager, "_load_manifest") as mock_load_manifest:
            mock_load_manifest.return_value = {
                "stages": {
                    "stage_0": {
                        "allowed_transitions": ["stage_1"]
                    },
                    "stage_1": {
                        "allowed_transitions": ["stage_0"]
                    }
                }
            }
            
            # Transition to stage_1
            result = self.manager.transition_to_stage("stage_1", {"reason": "test"})
            
            # Check if transition was successful
            assert result is True
            
            # Get updated state
            state = self.manager.get_current_state()
            
            # Check if state is correct
            assert state.stage_id == "stage_1"
            assert len(state.transitions) == 1
            assert state.transitions[0]["from_stage"] == "stage_0"
            assert state.transitions[0]["to_stage"] == "stage_1"
            assert state.transitions[0]["metadata"]["reason"] == "test"
            assert "last_updated" in state.metadata
    
    def test_can_transition_to(self):
        """Test can_transition_to."""
        # Mock _load_manifest
        with patch.object(self.manager, "_load_manifest") as mock_load_manifest:
            mock_load_manifest.return_value = {
                "stages": {
                    "stage_0": {
                        "allowed_transitions": ["stage_1"]
                    },
                    "stage_1": {
                        "allowed_transitions": ["stage_0", "stage_2"]
                    }
                }
            }
            
            # Check valid transitions
            assert self.manager.can_transition_to("stage_0", "stage_1") is True
            assert self.manager.can_transition_to("stage_1", "stage_0") is True
            assert self.manager.can_transition_to("stage_1", "stage_2") is True
            
            # Check invalid transitions
            assert self.manager.can_transition_to("stage_0", "stage_2") is False
            assert self.manager.can_transition_to("stage_2", "stage_0") is False
    
    def test_get_available_stages(self):
        """Test get_available_stages."""
        # Mock _load_manifest
        with patch.object(self.manager, "_load_manifest") as mock_load_manifest:
            mock_load_manifest.return_value = {
                "stages": {
                    "stage_0": {},
                    "stage_1": {},
                    "stage_2": {}
                }
            }
            
            # Get available stages
            stages = self.manager.get_available_stages()
            
            # Check if stages are correct
            assert set(stages) == {"stage_0", "stage_1", "stage_2"}
    
    def test_get_stage_info(self):
        """Test get_stage_info."""
        # Mock _load_manifest
        with patch.object(self.manager, "_load_manifest") as mock_load_manifest:
            mock_load_manifest.return_value = {
                "stages": {
                    "stage_0": {
                        "name": "Initial Bootstrap",
                        "description": "Initial stage"
                    },
                    "stage_1": {
                        "name": "Basic Self-Improvement",
                        "description": "First evolution stage"
                    }
                }
            }
            
            # Get stage info
            stage_info = self.manager.get_stage_info("stage_0")
            
            # Check if stage info is correct
            assert stage_info["name"] == "Initial Bootstrap"
            assert stage_info["description"] == "Initial stage"
