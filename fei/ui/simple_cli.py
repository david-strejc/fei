#!/usr/bin/env python3
"""
Simple command-line interface for Fei code assistant

This module provides a streamlined CLI experience for interacting with Fei,
focusing on simplicity and ease of use while maintaining all core functionality.
"""

import asyncio
import argparse
import json
import signal
import readline
from pathlib import Path
from typing import Dict, Any, Optional
from collections import deque

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
    "gray": "\033[90m",
}

def colorize(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"


class SimpleCLI:
    """Simple command-line interface for Fei code assistant."""

    def __init__(self):
        """Initialize simple CLI."""
        self.config = Config()
        self.tool_registry = None
        self.assistant = None
        self.history = deque(maxlen=100)
        self.history_file = None
        self.loop = asyncio.get_event_loop()

        # Set up readline for command history
        self._setup_readline()

    def _setup_readline(self) -> None:
        """Set up readline for command history."""
        # Create .fei directory in user's home directory if it doesn't exist
        history_dir = Path.home() / ".fei"
        history_dir.mkdir(exist_ok=True)

        # Set up readline history file
        readline_history_file = history_dir / "simple_cli_history"
        try:
            readline.read_history_file(readline_history_file)
        except FileNotFoundError:
            pass

        # Register to save readline history at exit
        import atexit
        atexit.register(readline.write_history_file, readline_history_file)

        # Configure readline
        readline.set_history_length(1000)
        readline.parse_and_bind('tab: complete')

    def setup(self) -> None:
        """Set up the CLI."""
        # Set up tool registry
        self.tool_registry = self._setup_tools()

        # Set up history file
        history_dir = Path.home() / ".fei"
        history_dir.mkdir(exist_ok=True)
        self.history_file = history_dir / "simple_cli_history.json"

        # Load history
        self._load_history()

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
        print(colorize("\n🐉 Fei Code Assistant - Simple CLI 🐉", "bold"))
        print(colorize("Advanced code manipulation with AI assistance", "cyan"))
        print(colorize(f"Connected to {self.assistant.provider} using model {self.assistant.model}", "green"))
        print(colorize("Type your message and press Enter to send.", "cyan"))
        print(colorize("Type '/help' to see available commands.", "cyan"))

    def print_help(self) -> None:
        """Print help message."""
        print(colorize("\nAvailable commands:", "bold"))
        print(colorize("  /help", "yellow") + " - Show this help message")
        print(colorize("  /exit", "yellow") + " - Exit the application")
        print(colorize("  /clear", "yellow") + " - Clear conversation history")
        print(colorize("  /history", "yellow") + " - Show conversation history")
        print(colorize("  /tools", "yellow") + " - Show available tools")
        print(colorize("  /model", "yellow") + " - Show current model information")
        print(colorize("  /task <task>", "yellow") + " - Execute a task without requiring 'continue' prompts")
        print(colorize("  /save <filename>", "yellow") + " - Save conversation to a file")
        print(colorize("\nJust type your message to chat with Fei.", "cyan"))

    def print_tools(self) -> None:
        """Print available tools."""
        print(colorize("\nAvailable tools:", "bold"))
        for tool in self.tool_registry.get_tools():
            print(colorize(f"- {tool['name']}: ", "yellow") + tool["description"].split("\n")[0])

    def print_model_info(self) -> None:
        """Print current model information."""
        print(colorize("\nCurrent model information:", "bold"))
        print(colorize(f"Provider: ", "yellow") + self.assistant.provider)
        print(colorize(f"Model: ", "yellow") + self.assistant.model)

        # Get additional model information if available
        try:
            from fei.models.config import ModelConfig
            model_config = ModelConfig(self.config)
            model_info = model_config.get_model_info(self.assistant.provider, self.assistant.model)

            if model_info:
                print(colorize(f"Context length: ", "yellow") + str(model_info.get("context_length", "Unknown")))
                print(colorize(f"Capabilities: ", "yellow") + ", ".join(model_info.get("capabilities", [])))
        except Exception as e:
            logger.error(f"Failed to get model information: {e}")

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self.assistant:
            self.assistant.reset_conversation()
            print(colorize("Conversation history cleared.", "cyan"))

    def show_history(self) -> None:
        """Display conversation history."""
        if not self.history:
            print(colorize("No history available.", "yellow"))
            return

        print(colorize("\nConversation History:", "bold"))
        for i, entry in enumerate(self.history, 1):
            print(colorize(f"\n[{i}] You: ", "yellow"))
            print(entry["user"])
            print(colorize(f"\nFei: ", "cyan"))
            print(entry["assistant"][:200] + "..." if len(entry["assistant"]) > 200 else entry["assistant"])
            print(colorize("─" * 50, "blue"))

    def save_conversation(self, filename: str) -> None:
        """
        Save conversation to a file.

        Args:
            filename: Name of the file to save to
        """
        if not self.history:
            print(colorize("No conversation to save.", "yellow"))
            return

        # Ensure the filename has a .txt extension
        if not filename.endswith('.txt'):
            filename += '.txt'

        try:
            with open(filename, 'w') as f:
                f.write("# Fei Conversation Log\n\n")
                for entry in self.history:
                    f.write(f"## User\n\n{entry['user']}\n\n")
                    f.write(f"## Fei\n\n{entry['assistant']}\n\n")
                    f.write("---\n\n")

            print(colorize(f"Conversation saved to {filename}", "green"))
        except Exception as e:
            print(colorize(f"Error saving conversation: {e}", "red"))

    async def execute_task(self, task: str, max_iterations: int = 10) -> None:
        """
        Execute a task without requiring user prompts.

        Args:
            task: Task to execute
            max_iterations: Maximum number of iterations
        """
        from fei.core.task_executor import TaskExecutor

        # Create the task executor with the assistant
        executor = TaskExecutor(
            assistant=self.assistant,
            on_message=lambda msg: print(colorize("\nFei: ", "bold") + msg)
        )

        print(colorize(f"\nStarting task execution (max {max_iterations} iterations):", "cyan"))
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

    async def chat_loop(self) -> None:
        """Interactive chat loop."""
        self.print_welcome()

        # Add history info to welcome message
        history_count = len(self.history)
        if history_count > 0:
            print(colorize(f"\nLoaded {history_count} messages from history.", "cyan"))
            print(colorize("Type '/history' to view previous conversations.", "cyan"))

        while True:
            try:
                # Get user input
                user_input = input(colorize("\nYou: ", "bold"))

                # Check for commands
                if user_input.startswith('/'):
                    command_parts = user_input[1:].split(maxsplit=1)
                    command = command_parts[0].lower()
                    args = command_parts[1] if len(command_parts) > 1 else ""

                    if command in ['exit', 'quit']:
                        # Save history before exiting
                        self._save_history()
                        print(colorize("\nGoodbye!", "bold"))
                        break

                    elif command == 'help':
                        self.print_help()
                        continue

                    elif command == 'clear':
                        self.clear_conversation()
                        continue

                    elif command == 'history':
                        self.show_history()
                        continue

                    elif command == 'tools':
                        self.print_tools()
                        continue

                    elif command == 'model':
                        self.print_model_info()
                        continue

                    elif command == 'task':
                        if not args:
                            print(colorize("Please specify a task.", "yellow"))
                            continue
                        await self.execute_task(args)
                        continue

                    elif command == 'save':
                        if not args:
                            print(colorize("Please specify a filename.", "yellow"))
                            continue
                        self.save_conversation(args)
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

                # Add message and response to history
                self.history.append({
                    "user": user_input,
                    "assistant": response,
                    "timestamp": self.loop.time()
                })

                # Save history after each interaction
                self._save_history()

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

        # Evolution is completely decoupled - we don't load it by default
        # Users can explicitly enable evolution if needed
        if args.enable_evolution:
            try:
                # Import only when needed to maintain decoupling
                from fei.evolution import load_stage
                loaded_stage_id = load_stage(self.assistant, self.tool_registry)
                logger.info(f"Evolution Stage {loaded_stage_id} loaded.")
            except Exception as e:
                logger.error(f"Failed to load evolution stage: {e}", exc_info=True)
                print(colorize(f"Error loading evolution stage: {e}. Proceeding with baseline.", "red"))

        # Process based on arguments
        if args.task:
            # Task execution mode
            await self.execute_task(args.task, args.max_iterations)
        elif args.message:
            # Single message mode
            print(colorize("\nYou: ", "bold") + args.message)
            print(colorize("Fei is thinking...", "cyan"))

            response = await self.assistant.chat(args.message)

            # Add message and response to history
            self.history.append({
                "user": args.message,
                "assistant": response,
                "timestamp": self.loop.time()
            })

            # Save history
            self._save_history()

            # Display the response
            print(colorize("\nFei: ", "bold"))
            print(response)
        else:
            # Interactive chat mode
            await self.chat_loop()

    def shutdown(self) -> None:
        """Shut down the CLI."""
        # Save history
        self._save_history()

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
        description="Fei Code Assistant - Simple CLI",
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
    parser.add_argument("-m", "--message", help="Send a single message and exit")
    parser.add_argument("--task", help="Execute a task without requiring 'continue' prompts")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum iterations for task execution"
    )
    parser.add_argument(
        "--do-as-i-say",
        action="store_true",
        help="DANGEROUS: Bypass all command security checks in the Shell tool"
    )
    parser.add_argument(
        "--enable-evolution",
        action="store_true",
        help="Enable the evolution system (loads the current evolution stage)"
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
    cli = SimpleCLI()
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
