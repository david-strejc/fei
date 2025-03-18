#!/usr/bin/env python3
"""
Command-line interface for Fei code assistant
"""

import os
import sys
import asyncio
import argparse
import readline
import atexit
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Deque
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

class CLI:
    """Command-line interface for Fei code assistant"""
    
    def __init__(self):
        """Initialize CLI"""
        self.config = Config()
        self.tool_registry = self.setup_tools()
        self.assistant = None
        self.history = deque(maxlen=100)  # Store last 100 messages
        self.history_file = self.get_history_file_path()
        self.setup_history()
    
    def get_history_file_path(self) -> Path:
        """Get the path to the history file"""
        # Create .fei directory in user's home directory if it doesn't exist
        home_dir = Path.home()
        fei_dir = home_dir / ".fei"
        fei_dir.mkdir(exist_ok=True)
        
        # Return path to history file
        return fei_dir / "history.json"

    def setup_history(self) -> None:
        """Set up command history using readline or prompt_toolkit"""
        # Load history from file if it exists
        self.load_history()
        
        # Set up history directory
        history_dir = Path.home() / ".fei"
        history_dir.mkdir(exist_ok=True)
        
        if PROMPT_TOOLKIT_AVAILABLE:
            # Use prompt_toolkit for better handling
            prompt_history_file = str(history_dir / "prompt_history")
            self.prompt_session = PromptSession(
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
            atexit.register(readline.write_history_file, readline_history_file)
            
            # Configure readline
            readline.set_history_length(100)
            
            # Set readline completer to avoid automatic completion
            readline.parse_and_bind('tab: complete')
            readline.set_completer(lambda text, state: None)
        
    def load_history(self) -> None:
        """Load message history from file"""
        try:
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    history_data = json.load(f)
                    self.history = deque(history_data, maxlen=100)
                    logger.debug(f"Loaded {len(self.history)} messages from history")
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            
    def save_history(self) -> None:
        """Save message history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(list(self.history), f)
                logger.debug(f"Saved {len(self.history)} messages to history")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def setup_tools(self) -> ToolRegistry:
        """Set up tool registry with all tools"""
        registry = ToolRegistry()
        
        # Use the centralized tool registration function
        from fei.tools.code import create_code_tools
        create_code_tools(registry)
        
        return registry
    
    def setup_assistant(self, api_key: Optional[str] = None, model: Optional[str] = None, provider: Optional[str] = None) -> Assistant:
        """Set up assistant"""
        # The API key check is now handled inside the Assistant class
        # based on the selected provider
        
        return Assistant(
            config=self.config,
            api_key=api_key,
            model=model,
            provider=provider,
            tool_registry=self.tool_registry
        )
    
    def print_welcome(self) -> None:
        """Print welcome message"""
        print(colorize("\n🐉 Fei Code Assistant 🐉", "bold"))
        print(colorize("Advanced code manipulation with AI assistance", "cyan"))
        print(colorize(f"Connected to {self.assistant.provider} using model {self.assistant.model}", "green"))
        print(colorize("Type your message and press Enter to send.", "cyan"))
        print(colorize("Type 'exit' or 'quit' to exit, 'clear' to clear conversation history.", "cyan"))
    
    def print_available_tools(self) -> None:
        """Print available tools"""
        print(colorize("\nAvailable tools:", "bold"))
        for tool in self.tool_registry.get_tools():
            print(colorize(f"- {tool['name']}: ", "yellow") + tool["description"].split("\n")[0])
    
    async def chat_loop(self) -> None:
        """Interactive chat loop"""
        self.print_welcome()
        self.print_available_tools()
        
        # Add history info to welcome message
        history_count = len(self.history)
        if history_count > 0:
            print(colorize(f"\nLoaded {history_count} messages from history. Use up/down arrows to navigate.", "cyan"))
            print(colorize("Type 'history' to view previous conversations.", "cyan"))
        
        while True:
            # Get input using prompt_toolkit or fallback to readline
            if PROMPT_TOOLKIT_AVAILABLE:
                # Get user input with prompt_toolkit for better handling
                prompt_text = FormattedText([('bold', '\nYou: ')])
                message = self.prompt_session.prompt(prompt_text)
            else:
                # Fallback to basic approach with readline
                prompt = colorize("\nYou: ", "bold")
                sys.stdout.write("\n")  # Add newline for consistent appearance
                message = input(prompt)
                
                # Add message to readline history if not empty
                if message.strip():
                    readline.add_history(message)
            
            if message.lower() in ["exit", "quit"]:
                # Save history before exiting
                self.save_history()
                print(colorize("\nGoodbye!", "bold"))
                break
            
            if message.lower() == "clear":
                self.assistant.reset_conversation()
                print(colorize("Conversation history cleared.", "cyan"))
                continue
                
            if message.lower() == "history":
                self.show_history()
                continue
            
            if not message.strip():
                continue
            
            print(colorize("Fei is thinking...", "cyan"))
            
            try:
                response = await self.assistant.chat(message)
                
                # Add message and response to history
                self.history.append({
                    "user": message,
                    "assistant": response,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # Save history after each interaction
                self.save_history()
                
                # Make sure we display the response appropriately
                print(colorize("\nFei: ", "bold"))
                
                # Check if the response is empty or None
                if not response or response.strip() == "None":
                    # Look at the last few messages in the conversation to find tool results
                    tool_outputs = []
                    conversation = self.assistant.conversation
                    
                    # Go through the last few messages to extract tool results
                    for msg in reversed(conversation[-5:] if len(conversation) >= 5 else conversation):
                        if msg.get("role") == "tool":
                            try:
                                # Parse tool content if it's a string
                                content = msg.get("content", "")
                                if isinstance(content, str) and content.startswith("{") and content.endswith("}"):
                                    import json
                                    tool_result = json.loads(content)
                                    if "stdout" in tool_result and tool_result["stdout"]:
                                        tool_outputs.append(f"\nCommand output:\n{tool_result['stdout']}")
                            except Exception:
                                pass
                    
                    # Display tool outputs or a message about missing output
                    if tool_outputs:
                        for output in tool_outputs:
                            print(output)
                    else:
                        print("Command executed, but no output was returned.")
                else:
                    # Display the normal response
                    print(response)
            except Exception as e:
                print(colorize(f"\nError: {e}", "red"))
                
    def show_history(self) -> None:
        """Display command history"""
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
    
    async def process_single_message(self, message: str) -> None:
        """Process a single message and exit"""
        try:
            # Add message to history if using readline and not empty
            if not PROMPT_TOOLKIT_AVAILABLE and message.strip():
                readline.add_history(message)
            
            response = await self.assistant.chat(message)
            
            # Add message and response to history
            self.history.append({
                "user": message,
                "assistant": response,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Save history
            self.save_history()
            
            # Check if the response is empty or None
            if not response or response.strip() == "None":
                # Look at the last few messages in the conversation to find tool results
                tool_outputs = []
                conversation = self.assistant.conversation
                
                # Go through the last few messages to extract tool results
                for msg in reversed(conversation[-5:] if len(conversation) >= 5 else conversation):
                    if msg.get("role") == "tool":
                        try:
                            # Parse tool content if it's a string
                            content = msg.get("content", "")
                            if isinstance(content, str) and content.startswith("{") and content.endswith("}"):
                                import json
                                tool_result = json.loads(content)
                                if "stdout" in tool_result and tool_result["stdout"]:
                                    tool_outputs.append(f"Command output:\n{tool_result['stdout']}")
                        except Exception:
                            pass
                
                # Display tool outputs or a message about missing output
                if tool_outputs:
                    for output in tool_outputs:
                        print(output)
                else:
                    print("Command executed, but no output was returned.")
            else:
                # Display the normal response
                print(response)
        except Exception as e:
            print(colorize(f"Error: {e}", "red"))
    
    async def process_continuous_task(self, task: str, max_iterations: int = 10) -> None:
        """Process a continuous task without requiring user prompts"""
        from fei.core.task_executor import TaskExecutor
        
        # Create the task executor with the assistant
        executor = TaskExecutor(
            assistant=self.assistant,
            on_message=lambda msg: print(colorize("\nFei: ", "bold") + "\n" + msg)
        )
        
        print(colorize(f"\nStarting continuous task execution (max {max_iterations} iterations):", "cyan"))
        print(colorize("Task: ", "bold") + task)
        
        # Enhance the task prompt to guide the LLM
        enhanced_task = f"""
I'll help you with this task: {task}

IMPORTANT: I'll work on this task in steps. After each step, I'll wait for your response.
When the task is completely finished, include the exact text [TASK_COMPLETE] at the end of your message.
"""
        
        # Execute the task
        print(colorize("\nExecuting task...", "cyan"))
        result = await executor.execute_task(enhanced_task, max_iterations)
        
        print(colorize(f"\nTask Status: {result}", "green"))

    async def run(self, args: argparse.Namespace) -> None:
        """Run CLI with parsed arguments"""
        # Set up logging
        if args.debug:
            get_logger("fei").setLevel("DEBUG")
        
        # Set up assistant
        self.assistant = self.setup_assistant(args.api_key, args.model, args.provider)
        
        logger.info(f"Using provider: {self.assistant.provider}, model: {self.assistant.model}")
        
        # Process based on arguments
        if args.task:
            # Continuous task execution mode
            await self.process_continuous_task(args.task, args.max_iterations)
        elif args.message:
            # Single message mode
            await self.process_single_message(args.message)
        else:
            # Interactive chat mode
            await self.chat_loop()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Fei Code Assistant - Advanced code manipulation with AI assistance")
    
    parser.add_argument("--api-key", help="API key (defaults to provider-specific environment variable)")
    parser.add_argument("--model", help="Model to use (defaults to provider's default model)")
    parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "groq"], help="LLM provider to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("-m", "--message", help="Send a single message and exit")
    parser.add_argument("--textual", action="store_true", help="Use the modern Textual-based chat interface")
    parser.add_argument("--task", help="Execute a task without requiring 'continue' prompts")
    parser.add_argument("--max-iterations", type=int, default=10, help="Maximum iterations for continuous task execution (default: 10)")
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Ask command - combines search and LLM
    ask_parser = subparsers.add_parser("ask", help="Ask a question with internet search capabilities")
    ask_parser.add_argument("question", help="Your question")
    ask_parser.add_argument("--provider", default="anthropic", choices=["anthropic", "openai", "groq"], help="LLM provider to use")
    ask_parser.add_argument("--search", action="store_true", default=True, help="Enable search capabilities")
    
    # History command
    history_parser = subparsers.add_parser("history", help="View conversation history")
    history_parser.add_argument("--mode", choices=["chat", "ask"], default="chat", help="Which history to view (chat or ask)")
    history_parser.add_argument("--limit", type=int, default=10, help="Number of history entries to show")
    history_parser.add_argument("--clear", action="store_true", help="Clear history")
    history_parser.add_argument("--load", type=int, help="Load a specific history entry into a new conversation")
    
    # MCP server commands
    mcp_parser = subparsers.add_parser("mcp", help="MCP server management")
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command", help="MCP commands")
    
    # List MCP servers
    list_parser = mcp_subparsers.add_parser("list", help="List MCP servers")
    
    # Add MCP server
    add_parser = mcp_subparsers.add_parser("add", help="Add MCP server")
    add_parser.add_argument("id", help="Server ID")
    add_parser.add_argument("url", help="Server URL")
    
    # Remove MCP server
    remove_parser = mcp_subparsers.add_parser("remove", help="Remove MCP server")
    remove_parser.add_argument("id", help="Server ID")
    
    # Set default MCP server
    set_default_parser = mcp_subparsers.add_parser("set-default", help="Set default MCP server")
    set_default_parser.add_argument("id", help="Server ID")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the web using Brave Search")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--count", type=int, default=5, help="Number of results (1-20, default 5)")
    search_parser.add_argument("--offset", type=int, default=0, help="Pagination offset (default 0)")
    
    return parser.parse_args()


async def handle_history_command(args: argparse.Namespace) -> None:
    """Handle history command"""
    home_dir = Path.home()
    fei_dir = home_dir / ".fei"
    fei_dir.mkdir(exist_ok=True)
    
    mode = args.mode
    history_file = fei_dir / f"{mode}_history.json"
    
    if args.clear:
        # Clear history
        if history_file.exists():
            history_file.unlink()
            print(colorize(f"{mode.capitalize()} history cleared.", "green"))
        return
        
    # Check if history file exists
    if not history_file.exists():
        print(colorize(f"No {mode} history found.", "yellow"))
        return
        
    # Load history
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except Exception as e:
        print(colorize(f"Error loading history: {e}", "red"))
        return
        
    # Display history
    if not history:
        print(colorize(f"No {mode} history found.", "yellow"))
        return
    
    # Load specific entry and start a new conversation
    if args.load is not None:
        load_idx = args.load - 1  # Convert from 1-based to 0-based index
        
        if load_idx < 0 or load_idx >= len(history):
            print(colorize(f"Invalid history index: {args.load}", "red"))
            return
            
        # Load the entry
        entry = history[load_idx]
        
        if mode == "chat":
            # Start a new chat with this message
            cli = CLI()
            cli.setup_assistant()
            
            print(colorize("\nLoading chat from history...", "cyan"))
            print(colorize(f"Original user message: {entry['user']}", "yellow"))
            
            # Run async chat with the loaded message
            # Instead of asyncio.run, use await directly since we're already in an async function
            await cli.process_single_message(entry['user'])
            return
        else:  # ask mode
            # Start a new ask session with this question
            question = entry['question']
            print(colorize(f"Loading question from history: {question}", "cyan"))
            
            # Create args for ask command
            ask_args = argparse.Namespace()
            ask_args.question = question
            ask_args.provider = "anthropic"
            ask_args.search = True
            
            # Run the ask command
            # Use await directly since we're already in an async function
            await handle_ask_command(ask_args)
            return
        
    # Limit number of entries
    history = history[-args.limit:] if len(history) > args.limit else history
    
    print(colorize(f"\n{mode.capitalize()} History:", "bold"))
    
    for i, entry in enumerate(history, 1):
        if mode == "chat":
            print(colorize(f"\n[{i}] You: ", "yellow"))
            print(entry["user"])
            print(colorize("\nFei: ", "cyan"))
            print(entry["assistant"][:200] + "..." if len(entry["assistant"]) > 200 else entry["assistant"])
        else:  # ask mode
            print(colorize(f"\n[{i}] Question: ", "yellow"))
            print(entry["question"])
            print(colorize("\nAnswer: ", "cyan"))
            print(entry["answer"][:200] + "..." if len(entry["answer"]) > 200 else entry["answer"])
        
        print(colorize("─" * 50, "blue"))

async def handle_mcp_command(args: argparse.Namespace) -> None:
    """Handle MCP commands"""
    config = Config()
    mcp_manager = MCPManager(config)
    
    if args.mcp_command == "list":
        # List MCP servers
        servers = mcp_manager.list_servers()
        if servers:
            print(colorize("MCP Servers:", "bold"))
            for server in servers:
                print(f"  {server['id']}: {server['url']}")
        else:
            print(colorize("No MCP servers configured.", "yellow"))
    
    elif args.mcp_command == "add":
        # Add MCP server
        if mcp_manager.add_server(args.id, args.url):
            print(colorize(f"Added MCP server: {args.id}", "green"))
        else:
            print(colorize(f"Server ID already exists: {args.id}", "red"))
    
    elif args.mcp_command == "remove":
        # Remove MCP server
        if mcp_manager.remove_server(args.id):
            print(colorize(f"Removed MCP server: {args.id}", "green"))
        else:
            print(colorize(f"Server not found: {args.id}", "red"))
    
    elif args.mcp_command == "set-default":
        # Set default MCP server
        if mcp_manager.set_default_server(args.id):
            print(colorize(f"Set default MCP server: {args.id}", "green"))
        else:
            print(colorize(f"Server not found: {args.id}", "red"))

async def search_brave(query: str, count: int = 5, offset: int = 0) -> dict:
    """
    Search with Brave Search API
    
    Args:
        query: Search query
        count: Number of results
        offset: Pagination offset
        
    Returns:
        Search results
    """
    # Import requests for direct API call
    import requests
    import os
    
    # Make direct API call to Brave Search
    api_key = os.environ.get("BRAVE_API_KEY", "BSABGuCvrv8TWsq-MpBTip9bnRi6JUg")
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": count, "offset": offset}
    
    response = requests.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers=headers,
        params=params
    )
    
    response.raise_for_status()
    return response.json()

async def handle_search_command(args: argparse.Namespace) -> None:
    """Handle search command"""
    config = Config()
    
    try:
        print(colorize(f"Searching for: {args.query}", "bold"))
        
        # Use direct search without trying MCP at all
        results = await search_brave(args.query, args.count, args.offset)
        
        # Display results
        print(colorize("\nSearch Results:", "bold"))
        for i, result in enumerate(results.get("web", {}).get("results", []), 1):
            print(colorize(f"{i}. {result.get('title')}", "green"))
            print(f"   URL: {colorize(result.get('url'), 'blue')}")
            print(f"   {result.get('description')}")
            print()
    
    except Exception as e:
        print(colorize(f"Search failed: {e}", "red"))
        
async def handle_ask_command(args: argparse.Namespace) -> None:
    """Handle ask command with internet search"""
    config = Config()
    
    try:
        # Initialize readline history for ask command
        home_dir = Path.home()
        fei_dir = home_dir / ".fei"
        fei_dir.mkdir(exist_ok=True)
        readline_history_file = fei_dir / "ask_history"
        
        try:
            readline.read_history_file(readline_history_file)
        except FileNotFoundError:
            pass
        
        # Register to save readline history at exit
        atexit.register(readline.write_history_file, readline_history_file)
        readline.set_history_length(100)
        
        # Add this question to readline history if we're not using prompt_toolkit
        if not PROMPT_TOOLKIT_AVAILABLE and args.question.strip():
            readline.add_history(args.question)
        
        # Save user questions and answers
        history_file = fei_dir / "ask_history.json"
        history = []
        
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ask history: {e}")
            
        # Create tool registry
        tool_registry = ToolRegistry()
        
        # Create assistant
        assistant = Assistant(
            provider=args.provider,
            tool_registry=tool_registry
        )
        
        # System prompt for search-enhanced assistance
        system_prompt = """You are a helpful assistant with internet search capabilities.
When asked about current information, first use the provided search results to find up-to-date information.
Always look at the search results before giving your answer, especially if the question is about current events,
technologies, or facts that might have changed recently.

When you use information from search results, cite the source in your answer."""
        
        print(colorize(f"Question: {args.question}", "bold"))
        
        search_results = None
        
        # Search the internet if enabled
        if args.search:
            print(colorize("Searching the web...", "cyan"))
            
            try:
                # Use direct search - bypass MCP completely
                search_results = await search_brave(args.question, count=5)
                
                # Format search results for the LLM
                search_context = "\n\nHere are some search results that may help answer the question:\n\n"
                
                for i, result in enumerate(search_results.get("web", {}).get("results", []), 1):
                    search_context += f"[{i}] {result.get('title')}\n"
                    search_context += f"URL: {result.get('url')}\n"
                    search_context += f"Summary: {result.get('description')}\n\n"
                
                # Add search results to the question
                enhanced_question = args.question + search_context
                
            except Exception as e:
                print(colorize(f"Search failed: {e}", "red"))
                enhanced_question = args.question
        else:
            enhanced_question = args.question
            
        print(colorize("Generating answer...", "cyan"))
            
        # Get assistant response
        response = await assistant.chat(enhanced_question, system_prompt=system_prompt)
        
        # Save to history
        history.append({
            "question": args.question,
            "answer": response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Save history
        try:
            with open(history_file, 'w') as f:
                json.dump(history, f)
        except Exception as e:
            logger.error(f"Failed to save ask history: {e}")
        
        # Display response
        print(colorize("\nAnswer:", "bold"))
        print(response)
    
    except Exception as e:
        print(colorize(f"Error: {e}", "red"))


# Global event loop check function to handle asyncio properly
def run_async_command(coro):
    """Run an async command with proper event loop handling"""
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_running_loop()
        # We're in an event loop, we should not be using asyncio.run
        # Instead, use a helper method from the asyncio module to gather multiple tasks
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, use asyncio.run
        return asyncio.run(coro)

def main() -> None:
    """Main entry point"""
    args = parse_args()
    
    # Set up logging
    if args.debug:
        get_logger("fei").setLevel("DEBUG")
    
    try:
        # Check if textual interface is requested
        if args.textual:
            # Import here to avoid circular imports
            from fei.ui.textual_chat import main as textual_main
            
            # Get the app directly, don't run it yet
            app = textual_main()
            
            # Run the app directly, don't use asyncio.run()
            app.run()
            return
            
        # Handle specific commands
        if args.command == "mcp":
            run_async_command(handle_mcp_command(args))
        elif args.command == "search":
            run_async_command(handle_search_command(args))
        elif args.command == "ask":
            run_async_command(handle_ask_command(args))
        elif args.command == "history":
            run_async_command(handle_history_command(args))
        else:
            # Default to chat
            cli = CLI()
            run_async_command(cli.run(args))
            
    except KeyboardInterrupt:
        print(colorize("\nInterrupted by user. Goodbye!", "bold"))
    except Exception as e:
        print(colorize(f"\nError: {e}", "red"))
        if args.debug:
            raise

if __name__ == "__main__":
    main()