#!/usr/bin/env python3
"""
Logging utilities for Fei code assistant
"""

import os
import logging
import logging.handlers
from datetime import datetime
from typing import Optional, Dict, Any

def setup_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Set up logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    # Set log level
    if level is None:
        level = os.environ.get("FEI_LOG_LEVEL", "INFO").upper()
    
    # Configure numeric level
    numeric_level = getattr(logging, level, logging.INFO)
    
    # Basic configuration
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Add file handler if specified
    if log_file or os.environ.get("FEI_LOG_FILE"):
        log_file = log_file or os.environ.get("FEI_LOG_FILE")
        
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        )
        
        # Add to root logger
        logging.getLogger().addHandler(file_handler)
    
    # Set lower level for 3rd party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

# Configure logging
_loggers = {}

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger instance
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Set level
    if level is not None:
        logger.setLevel(level)
    else:
        # Default to INFO, but check environment variable
        env_level = os.environ.get("FEI_LOG_LEVEL", "INFO").upper()
        numeric_level = getattr(logging, env_level, logging.INFO)
        logger.setLevel(numeric_level)
    
    # Check if handlers already exist
    if logger.handlers:
        _loggers[name] = logger
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    )
    logger.addHandler(console_handler)
    
    # Create file handler if enabled
    log_file = os.environ.get("FEI_LOG_FILE")
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
        )
        logger.addHandler(file_handler)
    
    # Store logger in cache
    _loggers[name] = logger
    
    return logger