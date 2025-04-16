"""
Memory module for Fei.
This module contains the abstract interfaces and concrete implementations for memory systems.
"""

from typing import Dict, List, Optional, Any, Union

from fei.memory.base import MemoryInterface, MemoryFactory
from fei.memory.manager import MemoryManager

# Import all memory systems to register them
from fei.memory.memdir import MemdirMemory

__all__ = [
    'MemoryInterface',
    'MemoryFactory',
    'MemoryManager',
]
