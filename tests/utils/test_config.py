"""
Tests for the configuration system.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil

from fei.utils.config import Config


class TestConfig:
    """Test the Config class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a temporary config file
        self.config_path = os.path.join(self.temp_dir, "config.json")
        self.config_data = {
            "test_key": "test_value",
            "nested": {
                "key1": "value1",
                "key2": "value2"
            }
        }
        
        with open(self.config_path, "w") as f:
            json.dump(self.config_data, f)
    
    def teardown_method(self):
        """Tear down the test."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_init_with_file(self):
        """Test initialization with config file."""
        # Create config with file
        config = Config(config_file=self.config_path)
        
        # Check if config is loaded correctly
        assert config.get("test_key") == "test_value"
        assert config.get("nested", {}).get("key1") == "value1"
        assert config.get("nested", {}).get("key2") == "value2"
    
    def test_init_with_env(self):
        """Test initialization with environment variable."""
        # Set environment variable
        with patch.dict(os.environ, {"FEI_CONFIG_PATH": self.config_path}):
            # Create config without file
            config = Config()
            
            # Check if config is loaded correctly
            assert config.get("test_key") == "test_value"
            assert config.get("nested", {}).get("key1") == "value1"
            assert config.get("nested", {}).get("key2") == "value2"
    
    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        # Create config without file
        config = Config()
        
        # Check if config has default values
        assert config.get("provider") == "anthropic"
        assert config.get("model") is None
    
    def test_get(self):
        """Test get method."""
        # Create config
        config = Config(config_file=self.config_path)
        
        # Get existing key
        assert config.get("test_key") == "test_value"
        
        # Get non-existent key with default
        assert config.get("nonexistent", "default") == "default"
        
        # Get non-existent key without default
        assert config.get("nonexistent") is None
    
    def test_set(self):
        """Test set method."""
        # Create config
        config = Config(config_file=self.config_path)
        
        # Set new key
        config.set("new_key", "new_value")
        
        # Check if key is set
        assert config.get("new_key") == "new_value"
        
        # Set existing key
        config.set("test_key", "updated_value")
        
        # Check if key is updated
        assert config.get("test_key") == "updated_value"
    
    def test_set_nested(self):
        """Test set method with nested keys."""
        # Create config
        config = Config(config_file=self.config_path)
        
        # Set nested key
        config.set("nested.key3", "value3")
        
        # Check if key is set
        assert config.get("nested", {}).get("key3") == "value3"
        
        # Set existing nested key
        config.set("nested.key1", "updated_value")
        
        # Check if key is updated
        assert config.get("nested", {}).get("key1") == "updated_value"
    
    def test_save(self):
        """Test save method."""
        # Create config
        config = Config(config_file=self.config_path)
        
        # Set some keys
        config.set("new_key", "new_value")
        config.set("nested.key3", "value3")
        
        # Save config
        config.save()
        
        # Load config from file
        with open(self.config_path, "r") as f:
            saved_data = json.load(f)
        
        # Check if config is saved correctly
        assert saved_data["new_key"] == "new_value"
        assert saved_data["nested"]["key3"] == "value3"
    
    def test_save_without_file(self):
        """Test save method without file."""
        # Create config without file
        config = Config()
        
        # Set some keys
        config.set("new_key", "new_value")
        
        # Save config (should not raise exception)
        config.save()
    
    def test_load(self):
        """Test load method."""
        # Create config
        config = Config()
        
        # Load config from file
        config.load(self.config_path)
        
        # Check if config is loaded correctly
        assert config.get("test_key") == "test_value"
        assert config.get("nested", {}).get("key1") == "value1"
        assert config.get("nested", {}).get("key2") == "value2"
    
    def test_load_nonexistent_file(self):
        """Test load method with non-existent file."""
        # Create config
        config = Config()
        
        # Load config from non-existent file
        config.load(os.path.join(self.temp_dir, "nonexistent.json"))
        
        # Check if config has default values
        assert config.get("provider") == "anthropic"
        assert config.get("model") is None
    
    def test_load_invalid_file(self):
        """Test load method with invalid file."""
        # Create an invalid config file
        invalid_path = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_path, "w") as f:
            f.write("invalid json")
        
        # Create config
        config = Config()
        
        # Load config from invalid file
        config.load(invalid_path)
        
        # Check if config has default values
        assert config.get("provider") == "anthropic"
        assert config.get("model") is None
