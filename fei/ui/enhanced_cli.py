#!/usr/bin/env python3
"""
Enhanced command-line interface for Fei code assistant
"""

import os
import sys
import asyncio
import argparse
import readline
import json
import signal
from pathlib import Path
from typing import Dict, List, Any, Optional, Deque, Callable, Union
from collections import deque

# Try to import prompt_toolkit for better input handling
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.formatted_text import FormattedText
    PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    PROMPT_TOOLKIT_AVAILABLE = False

from fei.ui.base import UIInterface, ChatUIInterface, ConsentUIInterface, CommandUIInterface
from fei.core.assistant import Assistant
from fei.tools.enhanced_registry import EnhancedToolRegistry
from fei.utils.config import Config
from fei.utils.logging import get_logger
from fei.evolution import load_stage

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


class CLIChatUI(ChatUIInterface):
    """CLI implementation of chat UI interface."""

    def __init__(self):
        """Initialize CLI chat UI."""
        self.thinking_indicator = None

    def display_message(self, message: str, role: str = "assistant") -> None:
        """
        Display a message.

        Args:
            message: Message content
            role: Message role (user, assistant, system)
        """
        if role == "assistant":
            print(colorize("\nFei: ", "bold"))
            print(message)
        elif role == "user":
            print(colorize("\nYou: ", "bold"))
            print(message)
        elif role == "system":
            print(colorize("\nSystem: ", "cyan"))
            print(message)
        else:
            print(colorize(f"\n{role.capitalize()}: ", "magenta"))
            print(message)

    def get_input(self, prompt: Optional[str] = None) -> str:
        """
        Get user input.

        Args:
            prompt: Optional prompt to display

        Returns:
            User input
        """
        prompt = prompt or colorize("\nYou: ", "bold")

        if PROMPT_TOOLKIT_AVAILABLE:
            # Get user input with prompt_toolkit for better handling
            prompt_text = FormattedText([('bold', prompt)])
            return self.prompt_session.prompt(prompt_text)
        else:
            # Fallback to basic approach with readline
            sys.stdout.write("\n")  # Add newline for consistent appearance
            return input(prompt)

    def display_error(self, error: str) -> None:
        """
        Display an error message.

        Args:
            error: Error message
        """
        print(colorize(f"\nError: {error}", "red"))

    def display_thinking(self) -> None:
        """Display a thinking indicator."""
        print(colorize("Fei is thinking...", "cyan"))

    def clear_thinking(self) -> None:
        """Clear the thinking indicator."""
        # In CLI, we don't need to clear the thinking indicator
        pass


class CLIConsentUI(ConsentUIInterface):
    """CLI implementation of consent UI interface."""

    def __init__(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Initialize CLI consent UI.

        Args:
            loop: Optional event loop for async operations
        """
        self.loop = loop or asyncio.get_event_loop()

    def ask_consent(self, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ask for user consent.

        Args:
            message: Consent message
            details: Optional details about the action

        Returns:
            True if consent was given, False otherwise
        """
        print(colorize("\n--- MCP Consent Required ---", "yellow"))
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


class CLICommandUI(CommandUIInterface):
    """CLI implementation of command UI interface."""

    def __init__(self):
        """Initialize CLI command UI."""
        self.commands = {}

    def register_command(self, name: str, handler: Callable, help_text: str) -> None:
        """
        Register a command.

        Args:
            name: Command name
            handler: Command handler function
            help_text: Command help text
        """
        self.commands[name] = {
            "handler": handler,
            "help": help_text
        }

    def execute_command(self, command: str, args: List[str]) -> Any:
        """
        Execute a command.

        Args:
            command: Command name
            args: Command arguments

        Returns:
            Command result
        """
        if command not in self.commands:
            print(colorize(f"Unknown command: {command}", "red"))
            return None

        try:
            return self.commands[command]["handler"](*args)
        except Exception as e:
            print(colorize(f"Error executing command: {e}", "red"))
            return None

    def display_help(self) -> None:
        """Display help for all commands."""
        print(colorize("\nAvailable commands:", "bold"))
        for name, command in self.commands.items():
            print(colorize(f"  {name}: ", "yellow") + command["help"])


class EnhancedCLI(UIInterface):
    """Enhanced command-line interface for Fei code assistant."""

    def __init__(self):
        """Initialize enhanced CLI."""
        self.config = Config()
        self.tool_registry = None
        self.assistant = None
        self.history = deque(maxlen=100)
        self.history_file = None
        self.loop = asyncio.get_event_loop()

        self._chat_ui = CLIChatUI()
        self._consent_ui = CLIConsentUI(self.loop)
        self._command_ui = CLICommandUI()

    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the UI.

        Args:
            config: Optional configuration dictionary
        """
        if config:
            for key, value in config.items():
                self.config.set(key, value)

        # Set up tool registry
        self.tool_registry = self._setup_tools()

        # Set up history
        self.history_file = self._get_history_file_path()
        self._setup_history()

        # Register commands
        self._register_commands()

    def _get_history_file_path(self) -> Path:
        """
        Get the path to the history file.

        Returns:
            Path to the history file
        """
        # Create .fei directory in user's home directory if it doesn't exist
        home_dir = Path.home()
        fei_dir = home_dir / ".fei"
        fei_dir.mkdir(exist_ok=True)

        # Return path to history file
        return fei_dir / "history.json"

    def _setup_history(self) -> None:
        """Set up command history using readline or prompt_toolkit."""
        # Load history from file if it exists
        self._load_history()

        # Set up history directory
        history_dir = Path.home() / ".fei"
        history_dir.mkdir(exist_ok=True)

        if PROMPT_TOOLKIT_AVAILABLE:
            # Use prompt_toolkit for better handling
            prompt_history_file = str(history_dir / "prompt_history")
            self._chat_ui.prompt_session = PromptSession(
                history=FileHistory(prompt_history_file),
                auto_suggest=AutoSuggestFromHistory(),
                enable_history_search=True,
                mouse_support=True,
                complete_while_typing=True
            )
        else:
            # Fall back to readline if prompt_toolkit is not available
            readline_history_file = history_dir / "readline_history"
            try:
                readline.read_history_file(readline_history_file)
            except FileNotFoundError:
                pass

            # Register to save readline history at exit
            import atexit
            atexit.register(readline.write_history_file, readline_history_file)

            # Configure readline
            readline.set_history_length(100)
            readline.parse_and_bind('tab: complete')
            readline.set_completer(lambda text, state: None)

    def _load_history(self) -> None:
        """Load message history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                    self.history = deque(history_data, maxlen=100)
                    logger.debug(f"Loaded {len(self.history)} messages from history")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save message history to file."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(list(self.history), f)
                logger.debug(f"Saved {len(self.history)} messages to history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

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

    def _setup_assistant(self, api_key: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None) -> Assistant:
        """
        Set up assistant.

        Args:
            api_key: Optional API key
            model: Optional model name
            provider: Optional provider name

        Returns:
            Assistant instance
        """
        return Assistant(
            config=self.config,
            api_key=api_key,
            model=model,
            provider=provider,
            tool_registry=self.tool_registry,
            consent_ui_provider=self._consent_ui.ask_consent
        )

    def _register_commands(self) -> None:
        """Register CLI commands."""
        self._command_ui.register_command("exit", lambda: sys.exit(0), "Exit the application")
        self._command_ui.register_command("quit", lambda: sys.exit(0), "Exit the application")
        self._command_ui.register_command("clear", self._clear_conversation, "Clear conversation history")
        self._command_ui.register_command("history", self._show_history, "Show conversation history")
        self._command_ui.register_command("help", self._command_ui.display_help, "Show help")

    def _clear_conversation(self) -> None:
        """Clear conversation history."""
        if self.assistant:
            self.assistant.reset_conversation()
            print(colorize("Conversation history cleared.", "cyan"))

    def _show_history(self) -> None:
        """Display command history."""
        if not self.history:
            print(colorize("No history available.", "yellow"))
            return

        print(colorize("\nCommand History:", "bold"))
        for i, entry in enumerate(self.history, 1):
            print(colorize(f"\n[{i}] You: ", "yellow"))
            print(entry["user"])
            print(colorize(f"\nFei: ", "cyan"))
            print(entry["assistant"][:200] + "..." if len(entry["assistant"]) > 200 else entry["assistant"])
            print(colorize("─" * 50, "blue"))

    def _print_welcome(self) -> None:
        """Print welcome message."""
        print(colorize("\n🐉 Fei Code Assistant 🐉", "bold"))
        print(colorize("Advanced code manipulation with AI assistance", "cyan"))
        print(colorize(f"Connected to {self.assistant.provider} using model {self.assistant.model}", "green"))
        print(colorize("Type your message and press Enter to send.", "cyan"))
        print(colorize("Type 'exit' or 'quit' to exit, 'clear' to clear conversation history.", "cyan"))

    def _print_available_tools(self) -> None:
        """Print available tools."""
        print(colorize("\nAvailable tools:", "bold"))
        for tool in self.tool_registry.get_tools():
            print(colorize(f"- {tool['name']}: ", "yellow") + tool["description"].split("\n")[0])

    async def _chat_loop(self) -> None:
        """Interactive chat loop."""
        self._print_welcome()
        self._print_available_tools()

        # Add history info to welcome message
        history_count = len(self.history)
        if history_count > 0:
            print(colorize(f"\nLoaded {history_count} messages from history. Use up/down arrows to navigate.", "cyan"))
            print(colorize("Type 'history' to view previous conversations.", "cyan"))

        while True:
            # Get user input
            message = self._chat_ui.get_input()

            # Check for commands
            if message.startswith("/"):
                command_parts = message[1:].split()
                command = command_parts[0]
                args = command_parts[1:]
                self._command_ui.execute_command(command, args)
                continue

            if message.lower() in ["exit", "quit"]:
                # Save history before exiting
                self._save_history()
                print(colorize("\nGoodbye!", "bold"))
                break

            if message.lower() == "clear":
                self._clear_conversation()
                continue

            if message.lower() == "history":
                self._show_history()
                continue

            if not message.strip():
                continue

            self._chat_ui.display_thinking()

            try:
                response = await self.assistant.chat(message)

                # Add message and response to history
                self.history.append({
                    "user": message,
                    "assistant": response,
                    "timestamp": self.loop.time()
                })

                # Save history after each interaction
                self._save_history()

                # Display the response
                self._chat_ui.display_message(response, "assistant")
            except Exception as e:
                self._chat_ui.display_error(str(e))

    async def _process_single_message(self, message: str) -> None:
        """
        Process a single message and exit.

        Args:
            message: Message to process
        """
        try:
            self._chat_ui.display_message(message, "user")
            self._chat_ui.display_thinking()

            response = await self.assistant.chat(message)

            # Add message and response to history
            self.history.append({
                "user": message,
                "assistant": response,
                "timestamp": self.loop.time()
            })

            # Save history
            self._save_history()

            # Display the response
            self._chat_ui.display_message(response, "assistant")
        except Exception as e:
            self._chat_ui.display_error(str(e))

    async def _process_continuous_task(self, task: str, max_iterations: int = 10) -> None:
        """
        Process a continuous task without requiring user prompts.

        Args:
            task: Task to process
            max_iterations: Maximum number of iterations
        """
        from fei.core.task_executor import TaskExecutor

        # Create the task executor with the assistant
        executor = TaskExecutor(
            assistant=self.assistant,
            on_message=lambda msg: self._chat_ui.display_message(msg, "assistant")
        )

        print(colorize(f"\nStarting continuous task execution (max {max_iterations} iterations):", "cyan"))
        print(colorize("Task: ", "bold") + task)

        # Enhance the task prompt to guide the LLM for autonomous execution
        enhanced_task = f"""
Your goal is to complete the following task autonomously: {task}

Follow these instructions:
1. Break the task down into logical steps based on the request.
2. Execute each step sequentially using the available tools (like write_to_file, edit_handler, etc.).
3. For coding tasks, generate the necessary code for each step and use file tools to save or modify the code file.
4. Do NOT ask the user for the code or instructions for each step. Generate it yourself based on the overall task.
5. After completing a step that involves using a tool (like writing to a file), briefly confirm the action taken.
6. Continue executing steps until the entire task is finished.
7. When the entire task is complete, and only then, include the exact text [TASK_COMPLETE] at the very end of your final message.
"""

        # Execute the task
        print(colorize("\nExecuting task...", "cyan"))
        result = await executor.execute_task(enhanced_task, max_iterations)

        print(colorize(f"\nTask Status: {result}", "green"))

    async def run(self) -> None:
        """Run the UI."""
        # Parse command-line arguments
        args = self._parse_args()

        # Set up logging
        if args.debug:
            get_logger("fei").setLevel("DEBUG")

        # Set up assistant
        self.assistant = self._setup_assistant(args.api_key, args.model, args.provider)
        logger.info(f"Using provider: {self.assistant.provider}, model: {self.assistant.model}")

        # Load the current evolution stage
        try:
            loaded_stage_id = load_stage(self.assistant, self.tool_registry)
            logger.info(f"Evolution Stage {loaded_stage_id} loaded.")
        except Exception as e:
            logger.error(f"Failed to load evolution stage: {e}", exc_info=True)
            print(colorize(f"Error loading evolution stage: {e}. Proceeding with baseline.", "red"))

        # Process based on arguments
        if args.task:
            # Continuous task execution mode
            await self._process_continuous_task(args.task, args.max_iterations)
        elif args.message:
            # Single message mode
            await self._process_single_message(args.message)
        else:
            # Interactive chat mode
            await self._chat_loop()

    def _parse_args(self) -> argparse.Namespace:
        """
        Parse command-line arguments.

        Returns:
            Parsed arguments
        """
        parser = argparse.ArgumentParser(description="Fei Code Assistant - Advanced code manipulation with AI assistance")

        parser.add_argument("--api-key", help="API key (defaults to provider-specific environment variable)")
        parser.add_argument("--model", help="Model to use (defaults to provider's default model)")
        parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "groq", "google"], help="LLM provider to use")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        parser.add_argument("-m", "--message", help="Send a single message and exit")
        parser.add_argument("--task", help="Execute a task without requiring 'continue' prompts")
        parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations for continuous task execution (default: 10)")
        parser.add_argument("--do-as-i-say", action="store_true", help="DANGEROUS: Bypass all command security checks in the Shell tool")

        return parser.parse_args()

    def shutdown(self) -> None:
        """Shut down the UI."""
        # Save history
        self._save_history()

        # Clean up tool registry
        if self.tool_registry:
            self.tool_registry.cleanup()

    @property
    def chat(self) -> ChatUIInterface:
        """Get the chat UI interface."""
        return self._chat_ui

    @property
    def consent(self) -> ConsentUIInterface:
        """Get the consent UI interface."""
        return self._consent_ui

    @property
    def command(self) -> CommandUIInterface:
        """Get the command UI interface."""
        return self._command_ui


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fei Code Assistant CLI")
    parser.add_argument("--api-key", help="API key for the provider")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument("--provider", help="Provider to use (anthropic, openai, etc.)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--do-as-i-say", action="store_true", help="Bypass command security checks (DANGEROUS!)")
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

    cli = EnhancedCLI()
    cli.setup()

    try:
        asyncio.run(cli.run())
    except KeyboardInterrupt:
        print(colorize("\nExiting...", "cyan"))
    finally:
        cli.shutdown()


if __name__ == "__main__":
    main()
