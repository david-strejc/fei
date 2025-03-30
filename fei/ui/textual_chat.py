#!/usr/bin/env python3
"""
Textual-based modern chat interface for Fei code assistant
"""

import os
import sys
import asyncio
import argparse
from typing import Dict, List, Any, Optional, Iterable

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
# Dropdown seems removed or moved in textual-autocomplete v4+
from textual_autocomplete import AutoComplete, DropdownItem 

from fei.core.assistant import Assistant
from fei.core.mcp import MCPManager
from fei.tools.registry import ToolRegistry
from fei.tools.handlers import (
    glob_tool_handler,
    grep_tool_handler,
    view_handler,
    edit_handler,
    replace_handler,
    ls_handler
)
from fei.tools.definitions import TOOL_DEFINITIONS
from fei.tools.memory_tools import create_memory_tools
from fei.utils.config import Config
from fei.utils.logging import get_logger

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

class MemoryCommandSuggester(Suggester):
    """Custom suggester for memory commands"""
    
    def __init__(self):
        """Initialize the memory command suggester"""
        super().__init__(use_cache=True, case_sensitive=False)
        
        # List of main memory commands
        self.main_commands = [
            "/mem help",
            "/mem search ",
            "/mem list",
            "/mem view ",
            "/mem save ",
            "/mem tag ",
            "/mem server start",
            "/mem server stop",
            "/mem server status"
        ]
        
        # Sub-suggestions based on command context
        self.folder_suggestions = [
            ".Projects", 
            ".Archive", 
            ".Tags"
        ]
        
        # Common tag suggestions
        self.tag_suggestions = [
            "python",
            "javascript",
            "learning",
            "important",
            "code",
            "conversation",
            "fei",
            "bookmark"
        ]
    
    async def get_suggestion(self, value: str) -> Optional[str]:
        """
        Try to get completion suggestion for the input value
        
        Args:
            value: Current input value
            
        Returns:
            Suggestion string or None
        """
        if not value:
            return None
            
        # For main command suggestions
        if value.startswith("/"):
            for cmd in self.main_commands:
                if cmd.startswith(value) and cmd != value:
                    return cmd
        
        # For folder suggestions after "/mem list "
        if value.startswith("/mem list "):
            prefix = "/mem list "
            folder_part = value[len(prefix):]
            
            for folder in self.folder_suggestions:
                if folder.startswith(folder_part) and folder != folder_part:
                    return prefix + folder
        
        # For tag suggestions after "/mem tag "
        if value.startswith("/mem tag "):
            prefix = "/mem tag "
            tag_part = value[len(prefix):]
            
            for tag in self.tag_suggestions:
                if tag.startswith(tag_part) and tag != tag_part:
                    return prefix + tag
                    
        # No suggestion found
        return None

class MemoryCommandDropdown:
    """Factory for creating dropdown items for memory commands"""
    
    @staticmethod
    def get_dropdown_items() -> List[DropdownItem]:
        """Get the dropdown items for memory commands"""
        return [
            DropdownItem(main="/mem help", left_meta="Show available memory commands"),
            DropdownItem(main="/mem search", left_meta="Search memories with query syntax"),
            DropdownItem(main="/mem list", left_meta="List memories in a folder"),
            DropdownItem(main="/mem view", left_meta="View a specific memory by ID"),
            DropdownItem(main="/mem save", left_meta="Save conversation as a memory"),
            DropdownItem(main="/mem tag", left_meta="Search memories by tag"),
            DropdownItem(main="/mem server start", left_meta="Start the Memdir server"),
            DropdownItem(main="/mem server stop", left_meta="Stop the Memdir server"),
            DropdownItem(main="/mem server status", left_meta="Check Memdir server status"),
        ]

class InputArea(Horizontal):
    """Bottom area containing input field and send button"""
    
    def compose(self) -> ComposeResult:
        """Compose the input area"""
        # Create the input with auto-suggestion
        input_widget = Input(
            placeholder="Type your message here...",
            id="message-input",
            suggester=MemoryCommandSuggester()
        )
        
        yield input_widget
        yield Button("Send", id="send-button", variant="primary")

class FeiChatApp(App):
    """Modern Textual-based chat interface for Fei code assistant"""
    
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
    
    /* Styling for memory command autocomplete */
    #memory-command-autocomplete {
        width: 1fr;
    }
    
    Dropdown {
        background: $panel;
        border: tall $primary-darken-2;
        padding: 0;
        max-height: 20;
    }
    
    .autocomplete--highlight-match {
        color: $accent-lighten-2;
        text-style: bold;
    }
    
    .autocomplete--selection-cursor {
        background: $primary-darken-1;
        color: $text;
    }
    
    .autocomplete--left-column {
        color: $text-muted;
    }
    
    .autocomplete--right-column {
        color: $accent;
        text-style: italic;
    }
    """
    
    is_loading = reactive(False)
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        model: Optional[str] = None, 
        provider: Optional[str] = None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = Config()
        self.tool_registry = self.setup_tools()
        self.assistant = self.setup_assistant(api_key, model, provider)
        
    def setup_tools(self) -> ToolRegistry:
        """Set up tool registry with all tools"""
        registry = ToolRegistry()
        
        # Use the centralized tool registration function
        from fei.tools.code import create_code_tools
        create_code_tools(registry)
        
        # Register memory tools
        create_memory_tools(registry)
        
        return registry
    
    def setup_assistant(self, api_key: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None) -> Assistant:
        """Set up assistant"""
        return Assistant(
            config=self.config,
            api_key=api_key,
            model=model,
            provider=provider,
            tool_registry=self.tool_registry
        )
    
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

Memory system available - use `/mem help` for commands or press Ctrl+F to search

Type your message and press Enter to send."""
        
        # Schedule the welcome message to be added
        asyncio.create_task(self.add_assistant_message(welcome))
        
        # Set up dynamic tag suggestions based on available memory tags
        self._update_memory_suggestions()
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle submitted input"""
        # The event could come from the autocomplete's input
        if not event.value.strip():
            return
            
        self.handle_user_message(event.value)
        # Clear the input value
        event.input.value = ""
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        if event.button.id == "send-button":
            input_widget = self.query_one(AutoComplete)
            message_input = input_widget.input
            if message_input.value.strip():
                self.handle_user_message(message_input.value)
                message_input.value = ""
    
    def action_clear(self) -> None:
        """Clear the chat history"""
        chat_container = self.query_one("#chat-container")
        chat_container.remove_children()
        self.assistant.reset_conversation()
        
        # Add welcome message back
        welcome = f"Chat history cleared. Connected to {self.assistant.provider} using model {self.assistant.model}."
        asyncio.create_task(self.add_assistant_message(welcome))
        
    def action_search_memories(self) -> None:
        """Open memory search prompt"""
        input_widget = self.query_one(AutoComplete)
        message_input = input_widget.input
        message_input.value = "/mem search "
        message_input.focus()
        
    def _update_memory_suggestions(self) -> None:
        """Update memory suggestions with available tags and folders"""
        try:
            # Try to get memory tags from memdir connector
            from fei.tools.memory_tools import memory_list_handler, memory_search_handler
            
            # Get a list of unique tags from memories
            search_result = memory_search_handler({"query": "#", "limit": 100})
            tags = set()
            
            if not isinstance(search_result, dict) or "error" in search_result:
                return
                
            for memory in search_result.get("results", []):
                memory_tags = memory.get("tags", "")
                if memory_tags:
                    tags.update([tag.strip() for tag in memory_tags.split(",")])
            
            # Get list of folders
            list_result = memory_list_handler({})
            folders = set()
            
            if not isinstance(list_result, dict) or "error" in list_result:
                return
                
            # Extract folder names from memories
            for memory in list_result.get("memories", []):
                folder = memory.get("folder", "")
                if folder:
                    folders.add(folder)
            
            # Update the suggester with the discovered tags and folders
            # Find the memory command suggester
            input_widget = self.query_one("#message-input", Input)
            if hasattr(input_widget, "suggester") and isinstance(input_widget.suggester, MemoryCommandSuggester):
                suggester = input_widget.suggester
                
                # Update tag suggestions
                if tags:
                    suggester.tag_suggestions = list(tags)
                
                # Update folder suggestions
                if folders:
                    suggester.folder_suggestions = list(folders)
        except Exception as e:
            logger.error(f"Error updating memory suggestions: {e}")
            
    def on_auto_complete_selected(self, event) -> None:
        """Handle autocomplete selection event"""
        # Get input widget and send the message if it's a complete command
        selected_item = event.item
        selected_value = selected_item.main.plain if hasattr(selected_item, 'main') else ""
        
        # Simple commands can be executed immediately
        if selected_value == "/mem help":
            self.handle_user_message(selected_value)
        elif selected_value == "/mem list":
            self.handle_user_message(selected_value)
        # Commands that need more input get the cursor placed at the right position
        elif selected_value in ["/mem search", "/mem view", "/mem save", "/mem tag"]:
            # Add a space and focus the input to let the user add the rest
            input_widget = self.query_one("#message-input", Input)
            input_widget.value = f"{selected_value} "
            input_widget.focus()
            input_widget.cursor_position = len(input_widget.value)
    
    def handle_user_message(self, message: str) -> None:
        """Handle a message from the user"""
        # Check for special commands
        if message.lower() in ["exit", "quit"]:
            self.exit()
            return
            
        if message.lower() == "clear":
            self.action_clear()
            return
        
        # Check for memory commands
        if message.lower().startswith("/mem"):
            asyncio.create_task(self.handle_memory_command(message))
            return
        
        # Add user message to the chat
        asyncio.create_task(self.add_user_message(message))
        
        # Process with assistant (in background)
        asyncio.create_task(self.process_with_assistant(message))
        
    async def handle_memory_command(self, command: str) -> None:
        """
        Handle memory-related commands
        
        Supported commands:
        - /mem search <query> - Search memories
        - /mem list - List memories
        - /mem view <memory_id> - View a specific memory
        - /mem save <subject> - Save conversation as a memory
        - /mem help - Show memory command help
        """
        # Add command to chat as user message
        await self.add_user_message(command)
        
        # Process command
        parts = command.split(maxsplit=2)
        
        if len(parts) < 2:
            await self.add_assistant_message("Invalid memory command. Use '/mem help' for available commands.")
            return
        
        subcommand = parts[1].lower()
        
        # Check if Memdir server is running before executing commands (except help)
        if subcommand != "help":
            try:
                from fei.tools.memdir_connector import MemdirConnector
                connector = MemdirConnector()
                # Try to start the server automatically - always start it regardless of check
                # Create connector with auto-start enabled
                connector = MemdirConnector(auto_start=True)
                result = connector.start_server_command()
                
                # Always continue with the command, as the actual memory operations will
                # now handle server auto-start internally
                if result["status"] != "already_running":
                    await self.add_assistant_message(
                        f"**Starting Memdir server:** {result['message']}"
                    )
            except Exception as e:
                await self.add_assistant_message(f"**Error checking Memdir connection:** {str(e)}")
                return
        
        # Show help
        if subcommand == "help":
            help_text = """**Memory System Commands:**

- `/mem search <query>` - Search memories with query syntax
- `/mem list [folder]` - List memories in a folder
- `/mem view <memory_id>` - View a specific memory
- `/mem save <subject>` - Save conversation as a memory
- `/mem tag <tag>` - Search memories by tag
- `/mem help` - Show this help message
- `/mem server start` - Start the Memdir server
- `/mem server stop` - Stop the Memdir server
- `/mem server status` - Check Memdir server status

**Examples:**
- `/mem search #python` - Search memories with tag 'python'
- `/mem search "machine learning"` - Search for 'machine learning' in subject/content 
- `/mem list` - List memories in the root folder
- `/mem list .Projects` - List memories in the Projects folder
- `/mem save "Important Python Tips"` - Save the conversation with a subject
- `/mem server start` - Start the server if it's not running
"""
            await self.add_assistant_message(help_text)
            return
            
        # List memories
        if subcommand == "list":
            folder = parts[2] if len(parts) > 2 else ""
            
            try:
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                # Call the memory_list tool handler directly
                from fei.tools.memory_tools import memory_list_handler
                result = memory_list_handler({"folder": folder, "limit": 10})
                
                if "error" in result:
                    await self.add_assistant_message(f"**Error:** {result['error']}")
                    # Since we're already auto-starting in the handler directly,
                    # we don't need to do another attempt here
                    return
                
                # Format the output
                count = result.get("count", 0)
                memories = result.get("memories", [])
                
                if count == 0:
                    output = f"No memories found in {folder or 'root folder'}."
                else:
                    output = f"**Memories in {folder or 'root folder'}:**\n\n"
                    for memory in memories:
                        subject = memory.get("subject", "No subject")
                        memory_id = memory.get("id", "")
                        tags = memory.get("tags", "")
                        tags_display = f" [tags: {tags}]" if tags else ""
                        
                        output += f"- **{subject}**{tags_display} (ID: `{memory_id}`)\n"
                    
                    if count > 10:
                        output += f"\n*Showing 10 of {count} memories. Use more specific search to narrow results.*"
                
                await self.add_assistant_message(output)
                
            except Exception as e:
                await self.add_assistant_message(f"**Error listing memories:** {str(e)}")
            finally:
                # Hide loading indicator
                self.is_loading = False
                loading.display = False
            
            return
            
        # Search memories
        if subcommand == "search" and len(parts) > 2:
            query = parts[2]
            
            try:
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                # Call the memory_search tool handler directly
                from fei.tools.memory_tools import memory_search_handler
                result = memory_search_handler({"query": query, "limit": 10})
                
                if "error" in result:
                    await self.add_assistant_message(f"**Error:** {result['error']}")
                    # Since we're already auto-starting in the handler directly,
                    # we don't need to do another attempt here
                    return
                
                # Format the output
                count = result.get("count", 0)
                memories = result.get("results", [])
                
                if count == 0:
                    output = f"No memories found matching query: '{query}'."
                else:
                    output = f"**Search results for '{query}':**\n\n"
                    for memory in memories:
                        subject = memory.get("subject", "No subject")
                        memory_id = memory.get("id", "")
                        tags = memory.get("tags", "")
                        tags_display = f" [tags: {tags}]" if tags else ""
                        
                        output += f"- **{subject}**{tags_display} (ID: `{memory_id}`)\n"
                    
                    if count > 10:
                        output += f"\n*Showing 10 of {count} results. Use more specific search to narrow results.*"
                
                await self.add_assistant_message(output)
                
            except Exception as e:
                await self.add_assistant_message(f"**Error searching memories:** {str(e)}")
            finally:
                # Hide loading indicator
                self.is_loading = False
                loading.display = False
            
            return
            
        # View a memory
        if subcommand == "view" and len(parts) > 2:
            memory_id = parts[2]
            
            try:
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                # Call the memory_view tool handler directly
                from fei.tools.memory_tools import memory_view_handler
                result = memory_view_handler({"memory_id": memory_id})
                
                if "error" in result:
                    await self.add_assistant_message(f"**Error:** {result['error']}")
                    # Since we're already auto-starting in the handler directly,
                    # we don't need to do another attempt here
                    return
                
                # Format the output
                subject = result.get("subject", "No subject")
                content = result.get("content", "")
                tags = result.get("tags", "")
                date = result.get("date", "")
                priority = result.get("priority", "")
                status = result.get("status", "")
                
                output = f"# {subject}\n\n"
                
                if tags:
                    output += f"**Tags:** {tags}\n"
                if date:
                    output += f"**Date:** {date}\n"
                if priority:
                    output += f"**Priority:** {priority}\n"
                if status:
                    output += f"**Status:** {status}\n"
                
                output += f"\n---\n\n{content}"
                
                await self.add_assistant_message(output)
                
            except Exception as e:
                await self.add_assistant_message(f"**Error viewing memory:** {str(e)}")
            finally:
                # Hide loading indicator
                self.is_loading = False
                loading.display = False
            
            return
            
        # Save conversation as a memory
        if subcommand == "save" and len(parts) > 2:
            subject = parts[2]
            
            # Collect conversation history
            chat_container = self.query_one("#chat-container")
            messages = chat_container.query("ChatMessage")
            
            conversation = ""
            for message in messages:
                sender = message.sender
                content = message.content
                conversation += f"**{sender.capitalize()}:** {content}\n\n"
            
            try:
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                # Call the memory_create tool handler directly
                from fei.tools.memory_tools import memory_create_handler
                result = memory_create_handler({
                    "subject": subject,
                    "content": conversation,
                    "tags": "conversation,fei",
                    "priority": "medium"
                })
                
                if "error" in result:
                    await self.add_assistant_message(f"**Error:** {result['error']}")
                    # Since we're already auto-starting in the handler directly,
                    # we don't need to do another attempt here
                    return
                
                memory_id = result.get("memory_id", "")
                await self.add_assistant_message(f"Conversation saved as memory with subject: '{subject}'\nMemory ID: `{memory_id}`")
                
            except Exception as e:
                await self.add_assistant_message(f"**Error saving memory:** {str(e)}")
            finally:
                # Hide loading indicator
                self.is_loading = False
                loading.display = False
            
            return
            
        # Search by tag
        if subcommand == "tag" and len(parts) > 2:
            tag = parts[2]
            
            try:
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                # Call the memory_search_by_tag tool handler directly
                from fei.tools.memory_tools import memory_search_by_tag_handler
                result = memory_search_by_tag_handler({"tag": tag})
                
                if "error" in result:
                    await self.add_assistant_message(f"**Error:** {result['error']}")
                    # Since we're already auto-starting in the handler directly,
                    # we don't need to do another attempt here
                    return
                
                # Format the output
                count = result.get("count", 0)
                memories = result.get("results", [])
                
                if count == 0:
                    output = f"No memories found with tag: '{tag}'."
                else:
                    output = f"**Memories with tag '{tag}':**\n\n"
                    for memory in memories:
                        subject = memory.get("subject", "No subject")
                        memory_id = memory.get("id", "")
                        
                        output += f"- **{subject}** (ID: `{memory_id}`)\n"
                    
                    if count > 10:
                        output += f"\n*Showing 10 of {count} results. Use more specific search to narrow results.*"
                
                await self.add_assistant_message(output)
                
            except Exception as e:
                await self.add_assistant_message(f"**Error searching by tag:** {str(e)}")
            finally:
                # Hide loading indicator
                self.is_loading = False
                loading.display = False
            
            return
        
        # Server commands
        if subcommand == "server":
            if len(parts) < 3:
                await self.add_assistant_message("**Error:** Missing server command (start, stop, or status)")
                return
                
            server_cmd = parts[2].lower()
            
            if server_cmd == "start":
                # Try to start the server
                from fei.tools.memdir_connector import MemdirConnector
                from fei.tools.memory_tools import memdir_server_start_handler
                
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                try:
                    result = memdir_server_start_handler({})
                    
                    if result.get("status") == "started":
                        await self.add_assistant_message(f"**{result.get('message', 'Server started successfully')}**")
                    elif result.get("status") == "already_running":
                        await self.add_assistant_message(f"**{result.get('message', 'Server is already running')}**")
                    else:
                        await self.add_assistant_message(f"**Error:** {result.get('message', 'Failed to start server')}")
                except Exception as e:
                    await self.add_assistant_message(f"**Error starting server:** {str(e)}")
                finally:
                    # Hide loading indicator
                    self.is_loading = False
                    loading.display = False
                
                return
                
            elif server_cmd == "stop":
                # Try to stop the server
                from fei.tools.memdir_connector import MemdirConnector
                from fei.tools.memory_tools import memdir_server_stop_handler
                
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                try:
                    result = memdir_server_stop_handler({})
                    
                    if result.get("status") == "stopped":
                        await self.add_assistant_message(f"**{result.get('message', 'Server stopped successfully')}**")
                    elif result.get("status") == "not_running":
                        await self.add_assistant_message(f"**{result.get('message', 'Server is not running')}**")
                    else:
                        await self.add_assistant_message(f"**Error:** {result.get('message', 'Failed to stop server')}")
                except Exception as e:
                    await self.add_assistant_message(f"**Error stopping server:** {str(e)}")
                finally:
                    # Hide loading indicator
                    self.is_loading = False
                    loading.display = False
                
                return
                
            elif server_cmd == "status":
                # Check server status
                from fei.tools.memdir_connector import MemdirConnector
                from fei.tools.memory_tools import memdir_server_status_handler
                
                # Show loading indicator
                loading = self.query_one(LoadingIndicator)
                self.is_loading = True
                loading.display = True
                
                try:
                    result = memdir_server_status_handler({})
                    status = result.get("status", "unknown")
                    
                    if status == "running":
                        port = result.get("port", "unknown")
                        await self.add_assistant_message(f"**Memdir Server Status:** Running on port {port}")
                    elif status == "stopped":
                        await self.add_assistant_message("**Memdir Server Status:** Stopped (not running)")
                    else:
                        await self.add_assistant_message(f"**Memdir Server Status:** {result.get('message', 'Unknown')}")
                except Exception as e:
                    await self.add_assistant_message(f"**Error checking server status:** {str(e)}")
                finally:
                    # Hide loading indicator
                    self.is_loading = False
                    loading.display = False
                
                return
            else:
                await self.add_assistant_message(f"**Unknown server command:** '{server_cmd}'. Use 'start', 'stop', or 'status'.")
                return
                
        # Unknown subcommand
        await self.add_assistant_message(f"Unknown memory command: '{subcommand}'. Use '/mem help' for available commands.")
    
    async def add_user_message(self, message: str) -> None:
        """Add a user message to the chat"""
        try:
            chat_container = self.query_one("#chat-container")
            await chat_container.add_message(message, sender="user")
        except Exception as e:
            # If there's an error adding the message, fall back to direct mount
            logger.error(f"Error adding user message: {e}")
            try:
                container = self.query_one("#chat-container")
                message_widget = ChatMessage(message, sender="user")
                await container.mount(message_widget)
            except Exception as inner_e:
                logger.error(f"Critical error displaying message: {inner_e}")
    
    async def add_assistant_message(self, message: str) -> None:
        """Add an assistant message to the chat"""
        try:
            chat_container = self.query_one("#chat-container")
            await chat_container.add_message(message, sender="assistant")
        except Exception as e:
            # If there's an error adding the message, fall back to direct mount
            logger.error(f"Error adding assistant message: {e}")
            try:
                container = self.query_one("#chat-container")
                message_widget = ChatMessage(message, sender="assistant")
                await container.mount(message_widget)
            except Exception as inner_e:
                logger.error(f"Critical error displaying message: {inner_e}")
    
    async def process_with_assistant(self, message: str) -> None:
        """Process a message with the assistant"""
        # Show loading indicator
        loading = self.query_one(LoadingIndicator)
        self.is_loading = True
        loading.display = True  # Show loading indicator
        
        try:
            # Get response from assistant
            response = await self.assistant.chat(message)
            
            if response is None:
                response = "Sorry, I couldn't generate a response. Please try again."
                
            # Add assistant message to chat
            await self.add_assistant_message(response)
            
        except Exception as e:
            # Handle error
            error_message = f"**Error:** {str(e)}"
            await self.add_assistant_message(error_message)
            logger.error(f"Assistant error: {e}", exc_info=True)
            
        finally:
            # Hide loading indicator
            self.is_loading = False
            loading.display = False  # Hide loading indicator
            
            # Refocus input
            self.query_one("#message-input").focus()

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Fei Code Assistant - Modern Textual Chat Interface")
    
    parser.add_argument("--api-key", help="API key (defaults to provider-specific environment variable)")
    parser.add_argument("--model", help="Model to use (defaults to provider's default model)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "groq"], help="LLM provider to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    return parser.parse_args()

def main() -> None:
    """Main entry point"""
    args = parse_args()
    
    # Set up logging
    if args.debug:
        get_logger("fei").setLevel("DEBUG")
    
    try:
        # Create the Textual app
        app = FeiChatApp(
            api_key=args.api_key,
            model=args.model,
            provider=args.provider
        )
        
        # Just return the app directly - let the caller handle the execution
        # This allows better integration when called from CLI
        return app
            
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            raise

if __name__ == "__main__":
    app = main()
    app.run()
