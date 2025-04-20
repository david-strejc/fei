#!/usr/bin/env python3
"""
Enhanced Textual-based chat interface for Fei code assistant
"""

import os
import sys
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Iterable, Callable, Union

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.reactive import reactive
from textual.widgets import (
    Header, Footer, Input, Button, Static,
    RichLog, LoadingIndicator, Markdown
)
from textual.binding import Binding
from textual.suggester import SuggestFromList, Suggester
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.console import RenderableType

# Import the autocomplete library
try:
    from textual_autocomplete import AutoComplete, DropdownItem
    AUTOCOMPLETE_AVAILABLE = True
except ImportError:
    AUTOCOMPLETE_AVAILABLE = False

from fei.ui.base import UIInterface, ChatUIInterface, ConsentUIInterface, CommandUIInterface
from fei.core.assistant import Assistant
from fei.tools.enhanced_registry import EnhancedToolRegistry
from fei.utils.config import Config
from fei.utils.logging import get_logger
from fei.evolution import load_stage

logger = get_logger(__name__)


class ChatMessage(Static):
    """A chat message display widget"""

    def __init__(
        self,
        content: str,
        sender: str = "user",
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.content = content
        self.sender = sender  # 'user' or 'assistant'

    def compose(self) -> ComposeResult:
        """Compose the message with appropriate styling"""
        # Different styling based on sender
        if self.sender == "user":
            style = "rgb(174,225,252)"  # Light blue
            panel_title = "You"
        else:
            style = "rgb(0,180,0)"  # Green
            panel_title = "Assistant"

        # Handle markdown for assistant messages
        if self.sender == "assistant":
            rendered_content = RichMarkdown(self.content)
        else:
            rendered_content = Text(self.content)

        # Create a panel with the content
        panel = Panel(
            rendered_content,
            title=panel_title,
            border_style=style,
            title_align="left",
            padding=(0, 1),
            # Make panel subtle
            highlight=False
        )

        yield Static(
            panel,
            classes=f"message {self.sender}"
        )


class ChatContainer(Vertical):
    """Container for all chat messages with proper scrolling"""

    def compose(self) -> ComposeResult:
        """Nothing to compose initially - messages will be added dynamically"""
        yield from []  # Return an empty iterable instead of None

    async def add_message(self, content: str, sender: str = "user") -> None:
        """Add a new message to the chat container and scroll to it"""
        message = ChatMessage(content, sender=sender)
        await self.mount(message)

        # Use the appropriate scrolling method based on Textual version
        try:
            # Try to scroll directly without await
            self.scroll_end(animate=False)
        except (AttributeError, TypeError):
            try:
                # Try standard scroll_to method
                self.scroll_to(y=1000000)  # Scroll to a very large value to ensure we're at the bottom
            except Exception as e:
                # Last resort, just log the error and continue
                print(f"Scrolling error: {e}")
                pass


class InputArea(Horizontal):
    """Bottom area containing input field and send button"""

    def compose(self) -> ComposeResult:
        """Compose the input area"""
        # Create the input with auto-suggestion if available
        if AUTOCOMPLETE_AVAILABLE:
            input_widget = AutoComplete(
                Input(
                    placeholder="Type your message here...",
                    id="message-input"
                ),
                id="autocomplete"
            )
        else:
            input_widget = Input(
                placeholder="Type your message here...",
                id="message-input"
            )

        yield input_widget
        yield Button("Send", id="send-button", variant="primary")


class TextualChatUI(ChatUIInterface):
    """Textual implementation of chat UI interface."""

    def __init__(self, app: 'EnhancedTextualApp'):
        """
        Initialize Textual chat UI.

        Args:
            app: Textual app instance
        """
        self.app = app

    def display_message(self, message: str, role: str = "assistant") -> None:
        """
        Display a message.

        Args:
            message: Message content
            role: Message role (user, assistant, system)
        """
        # Schedule the message to be added to the chat container
        asyncio.create_task(self.app.add_message(message, role))

    def get_input(self, prompt: Optional[str] = None) -> str:
        """
        Get user input.

        Args:
            prompt: Optional prompt to display

        Returns:
            User input
        """
        # This is not directly applicable in Textual UI
        # Input is handled through the UI events
        return ""

    def display_error(self, error: str) -> None:
        """
        Display an error message.

        Args:
            error: Error message
        """
        # Schedule the error message to be added to the chat container
        asyncio.create_task(self.app.add_message(f"Error: {error}", "system"))

    def display_thinking(self) -> None:
        """Display a thinking indicator."""
        self.app.is_loading = True

    def clear_thinking(self) -> None:
        """Clear the thinking indicator."""
        self.app.is_loading = False


class TextualConsentUI(ConsentUIInterface):
    """Textual implementation of consent UI interface."""

    def __init__(self, app: 'EnhancedTextualApp'):
        """
        Initialize Textual consent UI.

        Args:
            app: Textual app instance
        """
        self.app = app

    def ask_consent(self, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ask for user consent.

        Args:
            message: Consent message
            details: Optional details about the action

        Returns:
            True if consent was given, False otherwise
        """
        # This is a synchronous method, but we need to interact with the UI
        # We'll use a future to wait for the user's response
        future = asyncio.Future()

        # Schedule the consent dialog to be shown
        asyncio.create_task(self.app.show_consent_dialog(message, details, future))

        # Wait for the future to be resolved
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(future)
        except Exception as e:
            logger.error(f"Error during consent dialog: {e}")
            return False


class TextualCommandUI(CommandUIInterface):
    """Textual implementation of command UI interface."""

    def __init__(self, app: 'EnhancedTextualApp'):
        """
        Initialize Textual command UI.

        Args:
            app: Textual app instance
        """
        self.app = app
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
            asyncio.create_task(self.app.add_message(f"Unknown command: {command}", "system"))
            return None

        try:
            return self.commands[command]["handler"](*args)
        except Exception as e:
            asyncio.create_task(self.app.add_message(f"Error executing command: {e}", "system"))
            return None

    def display_help(self) -> None:
        """Display help for all commands."""
        help_text = "# Available Commands\n\n"
        for name, command in self.commands.items():
            help_text += f"- **{name}**: {command['help']}\n"

        asyncio.create_task(self.app.add_message(help_text, "system"))


class EnhancedTextualApp(App):
    """Enhanced Textual-based chat interface for Fei code assistant"""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+d", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear Chat"),
        Binding("ctrl+f", "search_memories", "Search Memories"),
    ]

    # Set the app colors to work with terminal theme
    COLORS = {
        "background": "transparent",
        "primary": "rgb(174,225,252)",  # Light blue for accents
        "secondary": "grey",  # More subtle color
        "accent": "rgb(174,225,252)",  # Light blue
        "success": "rgb(0,180,0)",  # Green
        "warning": "rgb(255,170,0)",  # Amber
        "error": "rgb(200,0,0)",  # Dark red
        "surface": "transparent",  # Use terminal background
        "panel": "transparent",  # Use terminal background
    }

    CSS = """
    Screen {
        background: transparent;
    }

    App {
        background: transparent;
    }

    #chat-container {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1 1;
        margin-bottom: 10;  /* Increased bottom margin to push content up */
        background: transparent;
    }

    .message {
        margin: 1 0;
        width: 100%;
    }

    .user {
        margin-right: 0;
        margin-left: 20;
    }

    .assistant {
        margin-left: 0;
        margin-right: 20;
    }

    #input-area {
        dock: bottom;
        width: 100%;
        height: 7;  /* Further increased height to make sure bottom is visible */
        background: transparent;  /* Transparent background */
        border-top: heavy $accent;
        padding: 0 1;  /* Reduced top padding from 1 to 0 */
        offset-y: -2;  /* Reduced offset to show the bottom part */
    }

    #message-input {
        width: 1fr;
        height: 3;  /* Increased height */
        border: tall $accent;
        background: transparent;  /* Transparent background */
        margin: 1 1 1 0;  /* Added vertical margins to center it */
        padding: 0 1;
    }

    #send-button {
        width: auto;
        min-width: 10;
        height: 3;  /* Match height with input */
        margin: 1 0 1 0;  /* Match vertical margins with input */
        background: transparent;
        border: tall $accent;
        color: $text;
    }

    LoadingIndicator {
        align: center middle;
        height: 1;
        margin: 1 0;
        display: none;  /* Hide by default */
    }

    /* Styling for autocomplete */
    #autocomplete {
        width: 1fr;
    }

    Dropdown {
        background: $panel;
        border: tall $primary-darken-2;
        padding: 0;
        max-height: 20;
    }
    """

    is_loading = reactive(False)

    def __init__(self):
        """Initialize enhanced Textual app."""
        super().__init__()
        self.config = Config()
        self.tool_registry = None
        self.assistant = None

        self._chat_ui = TextualChatUI(self)
        self._consent_ui = TextualConsentUI(self)
        self._command_ui = TextualCommandUI(self)

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

        # Register commands
        self._register_commands()

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
        """Register UI commands."""
        self._command_ui.register_command("clear", self.action_clear, "Clear conversation history")
        self._command_ui.register_command("help", self._command_ui.display_help, "Show help")
        self._command_ui.register_command("search", self.action_search_memories, "Search memories")

    def compose(self) -> ComposeResult:
        """Compose the app layout"""
        yield Header(show_clock=True)

        # Main chat container
        chat_container = ChatContainer(id="chat-container")
        yield chat_container

        # Loading indicator
        yield LoadingIndicator()

        # Input area
        yield InputArea(id="input-area")

        yield Footer()

    def on_mount(self) -> None:
        """When app is mounted"""
        # Hide loading indicator initially
        loading = self.query_one(LoadingIndicator)
        loading.display = False

        # Add welcome message
        welcome = f"""# 🐉 Fei Code Assistant 🐉

Advanced code manipulation with AI assistance

Connected to {self.assistant.provider} using model {self.assistant.model}

Type your message and press Enter to send."""

        # Schedule the welcome message to be added
        asyncio.create_task(self.add_message(welcome, "assistant"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle submitted input"""
        if not event.value.strip():
            return

        self.handle_user_message(event.value)
        # Clear the input value
        event.input.value = ""

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        if event.button.id == "send-button":
            if AUTOCOMPLETE_AVAILABLE:
                input_widget = self.query_one(AutoComplete)
                message_input = input_widget.input
            else:
                message_input = self.query_one(Input)

            if message_input.value.strip():
                self.handle_user_message(message_input.value)
                message_input.value = ""

    def watch_is_loading(self, is_loading: bool) -> None:
        """Watch the is_loading reactive attribute"""
        loading = self.query_one(LoadingIndicator)
        loading.display = is_loading

    def action_clear(self) -> None:
        """Clear the chat history"""
        chat_container = self.query_one("#chat-container")
        chat_container.remove_children()
        self.assistant.reset_conversation()

        # Add welcome message back
        welcome = f"Chat history cleared. Connected to {self.assistant.provider} using model {self.assistant.model}."
        asyncio.create_task(self.add_message(welcome, "assistant"))

    def action_search_memories(self) -> None:
        """Open memory search prompt"""
        if AUTOCOMPLETE_AVAILABLE:
            input_widget = self.query_one(AutoComplete)
            message_input = input_widget.input
        else:
            message_input = self.query_one(Input)

        message_input.value = "/mem search "
        message_input.focus()

    async def add_message(self, content: str, sender: str = "user") -> None:
        """
        Add a message to the chat container.

        Args:
            content: Message content
            sender: Message sender (user, assistant, system)
        """
        chat_container = self.query_one("#chat-container")
        await chat_container.add_message(content, sender)

    async def handle_user_message(self, message: str) -> None:
        """
        Handle a user message.

        Args:
            message: User message
        """
        # Check if it's a command
        if message.startswith("/"):
            command_parts = message[1:].split()
            command = command_parts[0]
            args = command_parts[1:]
            self._command_ui.execute_command(command, args)
            return

        # Add user message to chat
        await self.add_message(message, "user")

        # Show loading indicator
        self.is_loading = True

        try:
            # Get response from assistant
            response = await self.assistant.chat(message)

            # Add assistant message to chat
            await self.add_message(response, "assistant")
        except Exception as e:
            # Display error message
            await self.add_message(f"Error: {e}", "system")
        finally:
            # Hide loading indicator
            self.is_loading = False

    async def show_consent_dialog(self, message: str, details: Optional[Dict[str, Any]], future: asyncio.Future) -> None:
        """
        Show a consent dialog.

        Args:
            message: Consent message
            details: Optional details about the action
            future: Future to resolve with the user's response
        """
        # For now, we'll just add a message to the chat and wait for the user to respond
        consent_message = f"# Consent Required\n\n{message}\n\n"

        if details:
            consent_message += "## Details\n\n"
            for key, value in details.items():
                consent_message += f"- **{key}**: {value}\n"

        consent_message += "\nType 'yes' or 'no' to respond."

        await self.add_message(consent_message, "system")

        # Create a function to handle the user's response
        async def handle_consent_response(response: str) -> None:
            response = response.strip().lower()
            if response in ['yes', 'y']:
                await self.add_message("Consent granted.", "system")
                future.set_result(True)
            elif response in ['no', 'n']:
                await self.add_message("Consent denied.", "system")
                future.set_result(False)
            else:
                await self.add_message("Please answer 'yes' or 'no'.", "system")
                # Wait for another response
                if AUTOCOMPLETE_AVAILABLE:
                    input_widget = self.query_one(AutoComplete)
                    message_input = input_widget.input
                else:
                    message_input = self.query_one(Input)

                message_input.focus()

        # Set up a one-time event handler for the input
        def on_input_submitted(event: Input.Submitted) -> None:
            asyncio.create_task(handle_consent_response(event.value))
            # Remove this event handler
            self.query_one(Input).remove_class_handlers(Input.Submitted, on_input_submitted)

        # Add the event handler
        if AUTOCOMPLETE_AVAILABLE:
            self.query_one(AutoComplete).input.add_class_handler(Input.Submitted, on_input_submitted)
        else:
            self.query_one(Input).add_class_handler(Input.Submitted, on_input_submitted)

    def run(self) -> None:
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
            # We'll display this error in the UI after it's mounted

        # Run the app
        super().run()

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
        parser.add_argument("--do-as-i-say", action="store_true", help="DANGEROUS: Bypass all command security checks in the Shell tool")

        return parser.parse_args()

    def shutdown(self) -> None:
        """Shut down the UI."""
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
    parser = argparse.ArgumentParser(description="Fei Code Assistant Textual UI")
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

    app = EnhancedTextualApp()
    app.setup()

    try:
        app.run()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
