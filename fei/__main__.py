#!/usr/bin/env python3
"""
Main entry point for Fei code assistant
"""

import argparse

from fei.utils.logging import get_logger
from fei.ui.common import initialize_tools

def parse_args():
    """
    Parse command-line arguments for the main entry point
    """
    parser = argparse.ArgumentParser(description="Fei Code Assistant")
    parser.add_argument("--textual", action="store_true", help="Use the Textual UI instead of CLI")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--do-as-i-say", action="store_true", help="Bypass command security checks (DANGEROUS!)")
    return parser.parse_args()


def main():
    """
    Main entry point that handles both CLI and Textual interfaces
    """
    # Parse arguments
    args = parse_args()

    # Set up logging
    if args.debug:
        get_logger("fei").setLevel("DEBUG")

    # Initialize tools
    initialize_tools(do_as_i_say=args.do_as_i_say)

    # First check if --textual is in the arguments
    if args.textual:
        # Import the enhanced textual UI
        from fei.ui.enhanced_textual import main as textual_main
        # Run the textual UI
        textual_main()
    else:
        # Import the enhanced CLI UI
        from fei.ui.enhanced_cli import main as cli_main
        # Run the CLI UI
        cli_main()

if __name__ == "__main__":
    main()