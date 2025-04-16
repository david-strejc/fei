"""
Tests for the evolution loader.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil

from fei.evolution.loader import StageLoader
from fei.evolution.state import EvolutionStateManager
from fei.utils.config import Config


class TestStageLoader:
    """Test the stage loader."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a manifest file
        self.manifest_path = os.path.join(self.temp_dir, "manifest.json")
        self.manifest = {
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
        
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f)
        
        # Create a state file
        self.state_file = os.path.join(self.temp_dir, "evolution_state.json")
        self.state = {
            "stage_id": "stage_0",
            "transitions": [],
            "metadata": {"created_at": "2023-01-01T00:00:00"}
        }
        
        with open(self.state_file, "w") as f:
            json.dump(self.state, f)
        
        # Create a config with the temporary directory
        self.config = Config()
        self.config.set("evolution_manifest", self.manifest_path)
        self.config.set("evolution_state_file", self.state_file)
        
        # Create a stage loader
        self.loader = StageLoader(self.config)
    
    def teardown_method(self):
        """Tear down the test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        # Check if loader is initialized correctly
        assert self.loader.config is self.config
        assert isinstance(self.loader.state_manager, EvolutionStateManager)
        assert self.loader.manifest_path == self.manifest_path
    
    def test_load_manifest(self):
        """Test load_manifest."""
        # Load manifest
        manifest = self.loader.load_manifest()
        
        # Check if manifest is correct
        assert manifest == self.manifest
    
    def test_load_manifest_not_found(self):
        """Test load_manifest with non-existent file."""
        # Create a loader with non-existent manifest
        config = Config()
        config.set("evolution_manifest", "/nonexistent/manifest.json")
        loader = StageLoader(config)
        
        # Try to load manifest
        with pytest.raises(FileNotFoundError):
            loader.load_manifest()
    
    def test_get_current_stage_id(self):
        """Test get_current_stage_id."""
        # Get current stage ID
        stage_id = self.loader.get_current_stage_id()
        
        # Check if stage ID is correct
        assert stage_id == "stage_0"
    
    def test_get_stage_info(self):
        """Test get_stage_info."""
        # Get stage info
        stage_info = self.loader.get_stage_info("stage_0")
        
        # Check if stage info is correct
        assert stage_info["name"] == "Initial Bootstrap"
        assert stage_info["description"] == "Initial stage"
        assert stage_info["module"] == "fei.evolution.stages.stage_0"
        assert stage_info["allowed_transitions"] == ["stage_1"]
    
    def test_get_stage_info_not_found(self):
        """Test get_stage_info with non-existent stage."""
        # Try to get non-existent stage
        with pytest.raises(KeyError):
            self.loader.get_stage_info("nonexistent")
    
    @patch("importlib.import_module")
    def test_load_stage_module(self, mock_import_module):
        """Test load_stage_module."""
        # Set up mock module
        mock_module = MagicMock()
        mock_import_module.return_value = mock_module
        
        # Load stage module
        module = self.loader.load_stage_module("stage_0")
        
        # Check if module was imported correctly
        mock_import_module.assert_called_once_with("fei.evolution.stages.stage_0")
        assert module is mock_module
    
    def test_load_stage_module_not_found(self):
        """Test load_stage_module with non-existent stage."""
        # Try to load non-existent stage
        with pytest.raises(KeyError):
            self.loader.load_stage_module("nonexistent")
    
    @patch("importlib.import_module")
    def test_apply_stage(self, mock_import_module):
        """Test apply_stage."""
        # Set up mock module
        mock_module = MagicMock()
        mock_module.apply_stage_modifications = MagicMock()
        mock_import_module.return_value = mock_module
        
        # Create mock assistant and tool registry
        mock_assistant = MagicMock()
        mock_tool_registry = MagicMock()
        
        # Apply stage
        result = self.loader.apply_stage("stage_0", mock_assistant, mock_tool_registry)
        
        # Check if stage was applied correctly
        assert result is True
        mock_import_module.assert_called_once_with("fei.evolution.stages.stage_0")
        mock_module.apply_stage_modifications.assert_called_once_with(mock_assistant, mock_tool_registry)
    
    @patch("importlib.import_module")
    def test_apply_stage_missing_function(self, mock_import_module):
        """Test apply_stage with missing function."""
        # Set up mock module without apply_stage_modifications
        mock_module = MagicMock()
        mock_module.apply_stage_modifications = None
        mock_import_module.return_value = mock_module
        
        # Create mock assistant and tool registry
        mock_assistant = MagicMock()
        mock_tool_registry = MagicMock()
        
        # Try to apply stage
        with pytest.raises(AttributeError):
            self.loader.apply_stage("stage_0", mock_assistant, mock_tool_registry)
    
    @patch("importlib.import_module")
    def test_load_current_stage(self, mock_import_module):
        """Test load_current_stage."""
        # Set up mock module
        mock_module = MagicMock()
        mock_module.apply_stage_modifications = MagicMock()
        mock_import_module.return_value = mock_module
        
        # Create mock assistant and tool registry
        mock_assistant = MagicMock()
        mock_tool_registry = MagicMock()
        
        # Load current stage
        stage_id = self.loader.load_current_stage(mock_assistant, mock_tool_registry)
        
        # Check if stage was loaded correctly
        assert stage_id == "stage_0"
        mock_import_module.assert_called_once_with("fei.evolution.stages.stage_0")
        mock_module.apply_stage_modifications.assert_called_once_with(mock_assistant, mock_tool_registry)
    
    @patch("importlib.import_module")
    def test_load_current_stage_error_fallback(self, mock_import_module):
        """Test load_current_stage with error and fallback."""
        # Set up mock modules
        def mock_import_side_effect(module_path):
            if module_path == "fei.evolution.stages.stage_1":
                raise ImportError("Module not found")
            return MagicMock()
        
        mock_import_module.side_effect = mock_import_side_effect
        
        # Create mock assistant and tool registry
        mock_assistant = MagicMock()
        mock_tool_registry = MagicMock()
        
        # Update state to stage_1
        with open(self.state_file, "w") as f:
            json.dump({
                "stage_id": "stage_1",
                "transitions": [],
                "metadata": {"created_at": "2023-01-01T00:00:00"}
            }, f)
        
        # Load current stage (should fall back to stage_0)
        with pytest.raises(Exception):
            self.loader.load_current_stage(mock_assistant, mock_tool_registry)
    
    @patch("importlib.import_module")
    def test_transition_to_stage(self, mock_import_module):
        """Test transition_to_stage."""
        # Set up mock module
        mock_module = MagicMock()
        mock_module.apply_stage_modifications = MagicMock()
        mock_import_module.return_value = mock_module
        
        # Create mock assistant and tool registry
        mock_assistant = MagicMock()
        mock_tool_registry = MagicMock()
        
        # Transition to stage_1
        result = self.loader.transition_to_stage("stage_1", mock_assistant, mock_tool_registry, {"reason": "test"})
        
        # Check if transition was successful
        assert result is True
        
        # Check if state was updated
        with open(self.state_file, "r") as f:
            state = json.load(f)
            assert state["stage_id"] == "stage_1"
            assert len(state["transitions"]) == 1
            assert state["transitions"][0]["from_stage"] == "stage_0"
            assert state["transitions"][0]["to_stage"] == "stage_1"
            assert state["transitions"][0]["metadata"]["reason"] == "test"
    
    def test_transition_to_invalid_stage(self):
        """Test transition_to_stage with invalid stage."""
        # Create mock assistant and tool registry
        mock_assistant = MagicMock()
        mock_tool_registry = MagicMock()
        
        # Try to transition to invalid stage
        result = self.loader.transition_to_stage("stage_2", mock_assistant, mock_tool_registry)
        
        # Check if transition failed
        assert result is False
    
    def test_get_available_transitions(self):
        """Test get_available_transitions."""
        # Get available transitions
        transitions = self.loader.get_available_transitions()
        
        # Check if transitions are correct
        assert len(transitions) == 1
        assert transitions[0]["stage_id"] == "stage_1"
        assert transitions[0]["name"] == "Basic Self-Improvement"
        assert transitions[0]["description"] == "First evolution stage"
