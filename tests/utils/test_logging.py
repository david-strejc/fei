"""
Tests for the logging system.
"""

import pytest
from unittest.mock import MagicMock, patch
import logging

from fei.utils.logging import get_logger, setup_logging


class TestLogging:
    """Test the logging system."""
    
    def test_get_logger(self):
        """Test get_logger."""
        # Get logger
        logger = get_logger("test")
        
        # Check if logger is correct
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test"
    
    def test_get_logger_with_level(self):
        """Test get_logger with level."""
        # Get logger with level
        logger = get_logger("test", level="DEBUG")
        
        # Check if logger level is correct
        assert logger.level == logging.DEBUG
    
    def test_setup_logging(self):
        """Test setup_logging."""
        # Set up logging
        setup_logging()
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Check if root logger has handlers
        assert len(root_logger.handlers) > 0
    
    def test_setup_logging_with_level(self):
        """Test setup_logging with level."""
        # Set up logging with level
        setup_logging(level="DEBUG")
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Check if root logger level is correct
        assert root_logger.level == logging.DEBUG
    
    def test_setup_logging_with_file(self):
        """Test setup_logging with file."""
        # Mock file handler
        with patch("logging.FileHandler") as mock_file_handler:
            # Set up logging with file
            setup_logging(log_file="test.log")
            
            # Check if file handler was created
            mock_file_handler.assert_called_once_with("test.log")
    
    def test_setup_logging_with_format(self):
        """Test setup_logging with format."""
        # Mock formatter
        with patch("logging.Formatter") as mock_formatter:
            # Set up logging with format
            setup_logging(format="%(message)s")
            
            # Check if formatter was created with correct format
            mock_formatter.assert_called_once_with("%(message)s")
    
    def test_setup_logging_with_date_format(self):
        """Test setup_logging with date format."""
        # Mock formatter
        with patch("logging.Formatter") as mock_formatter:
            # Set up logging with date format
            setup_logging(date_format="%Y-%m-%d")
            
            # Check if formatter was created with correct date format
            mock_formatter.assert_called_once_with(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d"
            )
