#!/usr/bin/env python3
"""
Example using the modern Textual-based chat interface for Fei
"""

import sys
import os

# Add the parent directory to the path so we can import fei
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.ui.textual_chat import main

if __name__ == "__main__":
    main()