#!/usr/bin/env python3
"""
Simple interactive chat interface for Fei code assistant

This module provides a minimal, easy-to-use chat interface for interacting with Fei,
focusing on simplicity and conversation flow.
"""

import asyncio
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, Optional

from fei.core.assistant import Assistant
from fei.tools.enhanced_registry import EnhancedToolRegistry
from fei.utils.config import Config
from fei.utils.logging import get_logger

logger = get_logger(__name__)

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}

def colorize(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


class ChatCLI:
    """Simple interactive chat interface for Fei code assistant."""

    def __init__(self):
        """Initialize chat CLI."""
        self.config = Config()
        self.tool_registry = None
        self.assistant = None
        self.loop = asyncio.get_event_loop()

    def setup(self) -> None:
        """Set up the CLI."""
        # Set up tool registry
        self.tool_registry = self._setup_tools()

    def _setup_tools(self) -> EnhancedToolRegistry:
        """
        Set up tool registry with all tools.
        
        Returns:
            Enhanced tool registry
        """
        registry = EnhancedToolRegistry()
        
        # Use the common tool setup function
        from fei.ui.common import setup_tool_registry
        setup_tool_registry(registry)
        
        return registry

    def setup_assistant(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                       provider: Optional[str] = None) -> Assistant:
        """
        Set up assistant.
        
        Args:
            api_key: Optional API key
            model: Optional model name
            provider: Optional provider name
            
        Returns:
            Assistant instance
        """
        # Create a consent UI provider function
        def ask_consent(message: str, details: Optional[Dict[str, Any]] = None) -> bool:
            """Ask for user consent."""
            print(colorize("\n--- Consent Required ---", "yellow"))
            print(message)
            
            if details:
                print(colorize("\nDetails:", "cyan"))
                for key, value in details.items():
                    print(f"  {key}: {value}")
            
            while True:
                try:
                    response = input(colorize("\nDo you consent? (yes/no): ", "bold"))
                    response = response.strip().lower()
                    if response in ['yes', 'y']:
                        print(colorize("Consent granted.", "green"))
                        return True
                    elif response in ['no', 'n']:
                        print(colorize("Consent denied.", "red"))
                        return False
                    else:
                        print(colorize("Please answer 'yes' or 'no'.", "yellow"))
                except EOFError:
                    print(colorize("\nInput stream closed. Denying consent.", "red"))
                    return False
                except Exception as e:
                    logger.error(f"Error during CLI consent input: {e}", exc_info=True)
                    print(colorize(f"An error occurred: {e}. Denying consent.", "red"))
                    return False
        
        # Create the assistant
        return Assistant(
            config=self.config,
            api_key=api_key,
            model=model,
            provider=provider,
            tool_registry=self.tool_registry,
            consent_ui_provider=ask_consent
        )

    def print_welcome(self) -> None:
        """Print welcome message."""
        print(colorize("\n🐉 Fei Code Assistant - Chat Interface 🐉", "bold"))
        print(colorize("Type your messages and press Enter to chat with Fei", "cyan"))
        print(colorize(f"Connected to {self.assistant.provider} using model {self.assistant.model}", "green"))
        print(colorize("Type '/help' to see available commands.", "cyan"))

    def print_help(self) -> None:
        """Print help message."""
        print(colorize("\nAvailable commands:", "bold"))
        print(colorize("  /help", "yellow") + " - Show this help message")
        print(colorize("  /exit", "yellow") + " - Exit the application")
        print(colorize("  /clear", "yellow") + " - Clear conversation history")
        print(colorize("  /model", "yellow") + " - Show current model information")
        print(colorize("\nJust type your message to chat with Fei.", "cyan"))

    def print_model_info(self) -> None:
        """Print current model information."""
        print(colorize("\nCurrent model information:", "bold"))
        print(colorize(f"Provider: ", "yellow") + self.assistant.provider)
        print(colorize(f"Model: ", "yellow") + self.assistant.model)

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self.assistant:
            self.assistant.reset_conversation()
            print(colorize("Conversation history cleared.", "cyan"))

    async def chat_loop(self) -> None:
        """Interactive chat loop."""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input(colorize("\nYou: ", "bold"))
                
                # Check for commands
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command in ['exit', 'quit']:
                        print(colorize("\nGoodbye!", "bold"))
                        break
                    
                    elif command == 'help':
                        self.print_help()
                        continue
                    
                    elif command == 'clear':
                        self.clear_conversation()
                        continue
                    
                    elif command == 'model':
                        self.print_model_info()
                        continue
                    
                    else:
                        print(colorize(f"Unknown command: {command}", "red"))
                        print(colorize("Type '/help' to see available commands.", "cyan"))
                        continue
                
                # Skip empty messages
                if not user_input.strip():
                    continue
                
                # Display thinking indicator
                print(colorize("Fei is thinking...", "cyan"))
                
                # Get response from assistant
                response = await self.assistant.chat(user_input)
                
                # Display the response
                print(colorize("\nFei: ", "bold"))
                print(response)
                
            except KeyboardInterrupt:
                print(colorize("\nInterrupted. Type '/exit' to quit.", "yellow"))
                continue
                
            except Exception as e:
                print(colorize(f"\nError: {e}", "red"))

    async def run(self, args: argparse.Namespace) -> None:
        """
        Run the CLI.
        
        Args:
            args: Command-line arguments
        """
        # Set up assistant
        self.assistant = self.setup_assistant(args.api_key, args.model, args.provider)
        logger.info(f"Using provider: {self.assistant.provider}, model: {self.assistant.model}")
        
        # Start interactive chat mode
        await self.chat_loop()

    def shutdown(self) -> None:
        """Shut down the CLI."""
        # Clean up tool registry
        if self.tool_registry:
            self.tool_registry.cleanup()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Fei Code Assistant - Chat Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--api-key", help="API key (defaults to provider-specific environment variable)")
    parser.add_argument("--model", help="Model to use (defaults to provider's default model)")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "groq", "google", "gemini"],
        help="LLM provider to use (defaults to .env PROVIDER or anthropic)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--do-as-i-say", 
        action="store_true", 
        help="DANGEROUS: Bypass all command security checks in the Shell tool"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    if args.debug:
        get_logger("fei").setLevel("DEBUG")
    
    # Initialize tools
    from fei.ui.common import initialize_tools
    initialize_tools(do_as_i_say=args.do_as_i_say)
    
    # Create and set up CLI
    cli = ChatCLI()
    cli.setup()
    
    try:
        # Set up signal handlers
        loop = asyncio.get_event_loop()
        
        # Register signal handlers
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda: print(colorize("\nInterrupted. Exiting...", "yellow"))
            )
        
        # Run the CLI
        asyncio.run(cli.run(args))
    except KeyboardInterrupt:
        print(colorize("\nExiting...", "cyan"))
    finally:
        cli.shutdown()


if __name__ == "__main__":
    main()
