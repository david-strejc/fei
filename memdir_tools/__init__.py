"""
Memdir Tools - Memory Management based on Maildir approach

A suite of tools for managing memories in a hierarchical structure similar to 
Maildir email storage, with support for metadata, flags, and categorization.

This package provides:
- Command-line tools for managing memories
- API for programmatic memory manipulation
- HTTP server for remote memory access
- Filtering and organization features
- Distributed memory system (Memorychain)
"""

__version__ = "0.3.0"  # Updated for Memorychain support

# Make key modules accessible from the package
from .utils import (
    ensure_memdir_structure,
    get_memdir_folders,
    save_memory,
    list_memories,
    move_memory,
    update_memory_flags
)

from .search import (
    SearchQuery,
    search_memories,
    parse_search_args
)

# Import Memorychain components
try:
    from .memorychain import (
        MemoryBlock,
        MemoryChain,
        MemorychainNode,
        DEFAULT_PORT
    )
except ImportError:
    # Required libraries may not be installed
    pass

# Import server module for access to Flask app
try:
    from . import server
except ImportError:
    # Flask may not be installed
    pass