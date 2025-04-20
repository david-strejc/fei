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
    parser.add_argument("--simple", action="store_true", help="Use the Simple CLI instead of Enhanced CLI")
    parser.add_argument("--chat", action="store_true", help="Use the minimal Chat interface")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--do-as-i-say", action="store_true", help="Bypass command security checks (DANGEROUS!)")
    parser.add_argument("--api-key", help="API key (defaults to provider-specific environment variable)")
    parser.add_argument("--model", help="Model to use (defaults to provider's default model)")
    parser.add_argument("--provider", help="Provider to use (anthropic, openai, groq, google, gemini)")
    parser.add_argument("-m", "--message", help="Send a single message and exit")
    parser.add_argument("--task", help="Execute a task without requiring 'continue' prompts")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations for task execution")
    parser.add_argument("--enable-evolution", action="store_true", help="Enable the evolution system (loads the current evolution stage)")
    return parser.parse_args()


def main():
    """
    Main entry point that handles CLI and Textual interfaces
    """
    # Parse arguments
    args = parse_args()

    # Set up logging
    if args.debug:
        get_logger("fei").setLevel("DEBUG")

    # Initialize tools
    initialize_tools(do_as_i_say=args.do_as_i_say)

    # Check which UI to use based on arguments
    if args.textual:
        # Import the enhanced textual UI
        from fei.ui.enhanced_textual import main as textual_main
        # Run the textual UI
        textual_main()
    elif args.chat:
        # Import the minimal chat interface
        from fei.ui.chat_cli import main as chat_cli_main
        # Run the chat interface
        chat_cli_main()
    elif args.simple:
        # Import the simple CLI UI
        from fei.ui.simple_cli import main as simple_cli_main
        # Run the simple CLI UI with the arguments
        simple_cli_main()
    else:
        # Import the enhanced CLI UI
        from fei.ui.enhanced_cli import main as cli_main
        # Run the CLI UI
        cli_main()

if __name__ == "__main__":
    main()