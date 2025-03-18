#!/usr/bin/env python3
"""
Main entry point for Fei code assistant
"""

import sys
import argparse
from fei.ui.cli import main as cli_main, parse_args as cli_parse_args
from fei.ui.textual_chat import main as textual_main

def main():
    """
    Main entry point that handles both CLI and Textual interfaces
    """
    # First check if --textual is in the arguments
    # We need to do this manually because we want to use different parsers
    if "--textual" in sys.argv:
        # Remove the --textual argument from sys.argv for the textual interface
        sys.argv.remove("--textual")
        # Get the app directly, don't run it yet
        app = textual_main()
        # Run the app directly
        app.run()
    else:
        # Use the default CLI with its own argument parser
        cli_main()

if __name__ == "__main__":
    main()