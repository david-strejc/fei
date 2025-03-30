#!/usr/bin/env python3
"""
Configuration utilities for Fei code assistant

This module provides secure configuration management with proper validation
and permission checks.
"""

import os
import json
import configparser
import stat
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, TypeVar, cast, Tuple
import uuid
from enum import Enum
from dotenv import load_dotenv

from fei.utils.logging import get_logger

logger = get_logger(__name__)

# Type variable for config value validation
T = TypeVar('T')

class ConfigSecurityError(Exception):
    """Exception raised for configuration security issues"""
    pass

class ConfigValidationError(Exception):
    """Exception raised for configuration validation issues"""
    pass

class ConfigValueType(Enum):
    """Types of configuration values for validation"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"


# Config schema for validation
CONFIG_SCHEMA = {
    "api": {
        "timeout": {"type": ConfigValueType.INTEGER, "default": 30, "min": 1, "max": 600},
    },
    "anthropic": {
        "api_key": {"type": ConfigValueType.STRING, "secret": True},
        "model": {"type": ConfigValueType.STRING, "default": "claude-3-7-sonnet-20250219"},
    },
    "openai": {
        "api_key": {"type": ConfigValueType.STRING, "secret": True},
        "model": {"type": ConfigValueType.STRING, "default": "gpt-4o"},
    },
    "groq": {
        "api_key": {"type": ConfigValueType.STRING, "secret": True},
        "model": {"type": ConfigValueType.STRING, "default": "groq/llama3-70b-8192"},
    },
    "google": {
        "api_key": {"type": ConfigValueType.STRING, "secret": True},
        "model": {"type": ConfigValueType.STRING, "default": "gemini/gemini-1.5-pro-latest"}, # Defaulting to a known stable model
    },
    "brave": {
        "api_key": {"type": ConfigValueType.STRING, "secret": True},
    },
    "mcp": {
        "default_server": {"type": ConfigValueType.STRING},
        "servers": {"type": ConfigValueType.DICT},
    },
    "user": {
        "default_model": {"type": ConfigValueType.STRING},
        "default_provider": {"type": ConfigValueType.STRING, "default": "anthropic"},
    },
}


class ConfigValue:
    """Configuration value with type and validation"""
    
    def __init__(
        self, 
        value_type: ConfigValueType, 
        default: Any = None, 
        secret: bool = False, 
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        choices: Optional[List[Any]] = None
    ):
        """
        Initialize configuration value
        
        Args:
            value_type: Type of value
            default: Default value
            secret: Whether the value is sensitive (API key, etc.)
            min_value: Minimum value (for numeric types)
            max_value: Maximum value (for numeric types)
            choices: Allowed values (for string types)
        """
        self.value_type = value_type
        self.default = default
        self.secret = secret
        self.min_value = min_value
        self.max_value = max_value
        self.choices = choices
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str], Any]:
        """
        Validate and convert a value
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message, converted_value)
        """
        # Handle None
        if value is None:
            return True, None, self.default
        
        # Convert and validate based on type
        if self.value_type == ConfigValueType.STRING:
            if not isinstance(value, str):
                try:
                    value = str(value)
                except Exception:
                    return False, f"Cannot convert {value} to string", None
            
            if self.choices and value not in self.choices:
                return False, f"Value must be one of: {', '.join(self.choices)}", None
                
            return True, None, value
            
        elif self.value_type == ConfigValueType.INTEGER:
            if not isinstance(value, int):
                try:
                    value = int(value)
                except Exception:
                    return False, f"Cannot convert {value} to integer", None
            
            if self.min_value is not None and value < self.min_value:
                return False, f"Value must be at least {self.min_value}", None
                
            if self.max_value is not None and value > self.max_value:
                return False, f"Value must be at most {self.max_value}", None
                
            return True, None, value
            
        elif self.value_type == ConfigValueType.FLOAT:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except Exception:
                    return False, f"Cannot convert {value} to float", None
            
            if self.min_value is not None and value < self.min_value:
                return False, f"Value must be at least {self.min_value}", None
                
            if self.max_value is not None and value > self.max_value:
                return False, f"Value must be at most {self.max_value}", None
                
            return True, None, value
            
        elif self.value_type == ConfigValueType.BOOLEAN:
            if isinstance(value, bool):
                return True, None, value
                
            if isinstance(value, str):
                if value.lower() in ["true", "yes", "1", "on"]:
                    return True, None, True
                elif value.lower() in ["false", "no", "0", "off"]:
                    return True, None, False
                    
            return False, f"Cannot convert {value} to boolean", None
            
        elif self.value_type == ConfigValueType.LIST:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    try:
                        value = value.split(",")
                    except Exception:
                        return False, f"Cannot convert {value} to list", None
            
            if not isinstance(value, list):
                return False, f"Cannot convert {value} to list", None
                
            return True, None, value
            
        elif self.value_type == ConfigValueType.DICT:
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except Exception:
                    return False, f"Cannot convert {value} to dictionary", None
            
            if not isinstance(value, dict):
                return False, f"Cannot convert {value} to dictionary", None
                
            return True, None, value
            
        # Unknown type
        return False, f"Unknown value type: {self.value_type}", None


# Create config schema
def create_config_schema() -> Dict[str, Dict[str, ConfigValue]]:
    """
    Create config schema from static definition
    
    Returns:
        Dict of section -> option -> ConfigValue
    """
    schema = {}
    
    for section, options in CONFIG_SCHEMA.items():
        schema[section] = {}
        
        for option, params in options.items():
            value_type = params["type"]
            default = params.get("default")
            secret = params.get("secret", False)
            min_value = params.get("min")
            max_value = params.get("max")
            choices = params.get("choices")
            
            schema[section][option] = ConfigValue(
                value_type=value_type,
                default=default,
                secret=secret,
                min_value=min_value,
                max_value=max_value,
                choices=choices
            )
    
    return schema

# Global config instance
_config = None

def get_config(config_path: Optional[str] = None, env_file: Optional[str] = None) -> 'Config':
    """
    Get the global configuration instance
    
    Args:
        config_path: Path to configuration file
        env_file: Path to .env file
        
    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path, env_file)
    elif config_path or env_file:
        # Reload config if paths are specified
        _config = Config(config_path, env_file)
    
    return _config

# For testing, allow resetting the config
def reset_config() -> None:
    """Reset the global configuration instance (mainly for testing)"""
    global _config
    _config = None


class Config:
    """Configuration manager for Fei code assistant"""
    
    def __init__(self, config_path: Optional[str] = None, env_file: Optional[str] = None):
        """
        Initialize configuration with secure defaults
        
        Args:
            config_path: Path to configuration file
            env_file: Path to .env file
        """
        # Set up schema
        self.schema = create_config_schema()
        
        # Load environment variables from .env file
        self.env_file = env_file or os.path.join(os.getcwd(), '.env')
        self._load_env_file()
        
        # Load INI configuration
        self.config = configparser.ConfigParser()
        self.config_path = config_path or os.path.expanduser("~/.fei.ini")
        self.load_config()
        
        # Session ID for tracking
        self.session_id = str(uuid.uuid4())
    
    def _secure_path(self, path: str) -> None:
        """
        Ensure a file path has secure permissions
        
        Args:
            path: File path to secure
            
        Raises:
            ConfigSecurityError: If path cannot be secured
        """
        try:
            # Get path info
            path_obj = Path(path)
            
            # If file exists, check permissions
            if path_obj.exists():
                mode = path_obj.stat().st_mode
                
                # Unset group and other write permissions
                if mode & (stat.S_IWGRP | stat.S_IWOTH):
                    new_mode = mode & ~(stat.S_IWGRP | stat.S_IWOTH)
                    os.chmod(path, new_mode)
                    logger.debug(f"Secured permissions for {path}")
        
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not secure permissions for {path}: {e}")
    
    def _load_env_file(self) -> None:
        """
        Load environment variables from .env file securely
        """
        # Define sensitive keys to track
        sensitive_keys = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
                         "GOOGLE_API_KEY", "BRAVE_API_KEY", "LLM_API_KEY"]
        
        # Save current environment variable state for sensitive keys
        current_env = {}
        for key in sensitive_keys:
            if key in os.environ:
                current_env[key] = os.environ[key]
        
        # First try to load from specified env_file
        if os.path.exists(self.env_file):
            # Check file permissions
            self._secure_path(self.env_file)
            
            try:
                load_dotenv(self.env_file, override=False)  # Don't override existing env vars
                logger.debug(f"Loaded environment variables from {self.env_file}")
            except Exception as e:
                logger.error(f"Error loading .env file {self.env_file}: {e}")
        
        # If not found, try default locations
        default_locations = [
            os.path.join(os.getcwd(), '.env'),
            os.path.expanduser('~/.env'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
        ]
        
        for env_path in default_locations:
            if env_path != self.env_file and os.path.exists(env_path):
                # Check file permissions
                self._secure_path(env_path)
                
                try:
                    load_dotenv(env_path, override=False)
                    logger.debug(f"Loaded environment variables from {env_path}")
                except Exception as e:
                    logger.error(f"Error loading .env file {env_path}: {e}")
        
        # Restore any manually set environment variables
        for key, value in current_env.items():
            os.environ[key] = value
    
    def load_config(self) -> None:
        """
        Load configuration from file securely
        """
        if os.path.exists(self.config_path):
            # Check file permissions
            self._secure_path(self.config_path)
            
            try:
                self.config.read(self.config_path)
                logger.debug(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        
        # Ensure all schema sections exist
        for section in self.schema:
            if section not in self.config:
                self.config[section] = {}
    
    def save_config(self) -> None:
        """
        Save configuration to file securely
        """
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            
            # Write config
            with open(self.config_path, 'w') as f:
                self.config.write(f)
            
            # Secure file permissions
            self._secure_path(self.config_path)
            
            logger.debug(f"Saved config to {self.config_path}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise ConfigSecurityError(f"Could not save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key with type validation
        
        Args:
            key: Configuration key (format: section.key)
            default: Default value if key not found
            
        Returns:
            Configuration value (converted to the appropriate type)
            
        Raises:
            ValueError: If key format is invalid
        """
        # Validate key format
        if not key or '.' not in key:
            if default is not None:
                return default
            raise ValueError("Invalid key format. Use 'section.key'")
        
        section, option = key.split('.', 1)
        
        # First check environment variables
        env_value = self._get_from_env(section, option)
        if env_value is not None:
            # If we have schema info for this key, validate the value
            if section in self.schema and option in self.schema[section]:
                config_value = self.schema[section][option]
                is_valid, error, converted = config_value.validate(env_value)
                
                if is_valid:
                    return converted
                
                logger.warning(f"Environment value for {key} is invalid: {error}")
                # Fall back to config file or default
            else:
                # No schema info, return as is
                return env_value
        
        # Check config file
        if section in self.config and option in self.config[section]:
            value = self.config[section][option]
            
            # If we have schema info, validate the value
            if section in self.schema and option in self.schema[section]:
                config_value = self.schema[section][option]
                is_valid, error, converted = config_value.validate(value)
                
                if is_valid:
                    return converted
                
                logger.warning(f"Config value for {key} is invalid: {error}")
                # Fall back to default
            else:
                # No schema info, return as is
                return value
        
        # Use schema default if available
        if section in self.schema and option in self.schema[section]:
            return self.schema[section][option].default
            
        # Fall back to provided default
        return default
    
    def _get_from_env(self, section: str, option: str) -> Optional[str]:
        """
        Get a value from environment variables with proper naming conventions
        
        Args:
            section: Config section
            option: Config option
            
        Returns:
            Environment value or None
        """
        # Standard format: FEI_SECTION_OPTION
        env_key = f"FEI_{section.upper()}_{option.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value
            
        # Special cases for API keys
        if section in ['anthropic', 'openai', 'groq', 'google', 'brave'] and option == 'api_key':
            # Provider specific format (e.g., ANTHROPIC_API_KEY, GOOGLE_API_KEY)
            env_value = os.environ.get(f"{section.upper()}_API_KEY")
            if env_value is not None:
                return env_value
                
            # Try generic LLM_API_KEY as fallback for LLM providers
            if section in ['anthropic', 'openai', 'groq', 'google']:
                env_value = os.environ.get("LLM_API_KEY")
                if env_value is not None:
                    logger.debug(f"Using LLM_API_KEY for {section}")
                    return env_value
        
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key with validation
        
        Args:
            key: Configuration key (format: section.key)
            value: Value to set
            
        Raises:
            ValueError: If key format is invalid
            ConfigValidationError: If value is invalid
        """
        # Validate key format
        if not key or '.' not in key:
            raise ValueError("Invalid key format. Use 'section.key'")
        
        section, option = key.split('.', 1)
        
        # Validate value if we have schema info
        if section in self.schema and option in self.schema[section]:
            config_value = self.schema[section][option]
            is_valid, error, converted_value = config_value.validate(value)
            
            if not is_valid:
                raise ConfigValidationError(f"Invalid value for {key}: {error}")
                
            value = converted_value
        
        # Ensure section exists
        if section not in self.config:
            self.config[section] = {}
        
        # Convert value to string for configparser
        if value is None:
            str_value = ""
        elif isinstance(value, (dict, list)):
            str_value = json.dumps(value)
        else:
            str_value = str(value)
        
        # Set value
        self.config[section][option] = str_value
        
        # Save config
        self.save_config()
    
    def delete(self, key: str) -> bool:
        """
        Delete a configuration value by key
        
        Args:
            key: Configuration key (format: section.key)
            
        Returns:
            Whether the key was deleted
            
        Raises:
            ValueError: If key format is invalid
        """
        # Validate key format
        if not key or '.' not in key:
            raise ValueError("Invalid key format. Use 'section.key'")
        
        section, option = key.split('.', 1)
        
        # Check if key exists
        if section not in self.config or option not in self.config[section]:
            return False
        
        # Delete key
        del self.config[section][option]
        
        # Save config
        self.save_config()
        
        return True
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get all values in a section with proper type conversion
        
        Args:
            section: Section name
            
        Returns:
            Dictionary of values (converted to appropriate types)
        """
        if section not in self.config:
            return {}
        
        result = {}
        
        # Convert all values according to schema if available
        for option, value in self.config[section].items():
            if section in self.schema and option in self.schema[section]:
                config_value = self.schema[section][option]
                is_valid, _, converted = config_value.validate(value)
                
                if is_valid:
                    result[option] = converted
                else:
                    # Use as-is if validation fails
                    result[option] = value
            else:
                # No schema info, use as-is
                result[option] = value
        
        return result
    
    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all configuration values with proper type conversion
        
        Returns:
            Dictionary of all values (converted to appropriate types)
        """
        result = {}
        
        for section in self.config.sections():
            result[section] = self.get_section(section)
        
        return result
    
    def get_typed(self, key: str, expected_type: Type[T], default: Optional[T] = None) -> T:
        """
        Get a configuration value with explicit type checking
        
        Args:
            key: Configuration key (format: section.key)
            expected_type: Expected type
            default: Default value if key not found or type mismatch
            
        Returns:
            Configuration value as the expected type
        """
        value = self.get(key, default)
        
        # Check type
        if not isinstance(value, expected_type) and value is not None:
            # Try to convert
            try:
                if expected_type == int:
                    value = int(value)
                elif expected_type == float:
                    value = float(value)
                elif expected_type == bool:
                    if isinstance(value, str):
                        value = value.lower() in ["true", "yes", "1", "on"]
                    else:
                        value = bool(value)
                elif expected_type == str:
                    value = str(value)
                elif expected_type == list:
                    if isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except json.JSONDecodeError:
                            value = value.split(",")
                    else:
                        value = list(value)
                elif expected_type == dict:
                    if isinstance(value, str):
                        value = json.loads(value)
                    else:
                        value = dict(value)
                else:
                    # Can't convert, use default
                    logger.warning(f"Cannot convert {key} to {expected_type.__name__}")
                    return default if default is not None else cast(T, None)
            except (ValueError, TypeError, json.JSONDecodeError):
                # Conversion failed, use default
                logger.warning(f"Cannot convert {key} to {expected_type.__name__}")
                return default if default is not None else cast(T, None)
        
        return value if value is not None else (default if default is not None else cast(T, None))
    
    def get_string(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a string configuration value"""
        return self.get_typed(key, str, default)
    
    def get_int(self, key: str, default: Optional[int] = None) -> Optional[int]:
        """Get an integer configuration value"""
        return self.get_typed(key, int, default)
    
    def get_float(self, key: str, default: Optional[float] = None) -> Optional[float]:
        """Get a float configuration value"""
        return self.get_typed(key, float, default)
    
    def get_bool(self, key: str, default: Optional[bool] = None) -> Optional[bool]:
        """Get a boolean configuration value"""
        return self.get_typed(key, bool, default)
    
    def get_list(self, key: str, default: Optional[List[Any]] = None) -> Optional[List[Any]]:
        """Get a list configuration value"""
        return self.get_typed(key, list, default)
    
    def get_dict(self, key: str, default: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get a dictionary configuration value"""
        return self.get_typed(key, dict, default)
