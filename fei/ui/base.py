"""
Base UI interfaces for Fei.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable


class ConsentUIInterface(ABC):
    """Abstract interface for consent UI."""
    
    @abstractmethod
    def ask_consent(self, message: str, details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Ask for user consent.
        
        Args:
            message: Consent message
            details: Optional details about the action
            
        Returns:
            True if consent was given, False otherwise
        """
        pass


class ChatUIInterface(ABC):
    """Abstract interface for chat UI."""
    
    @abstractmethod
    def display_message(self, message: str, role: str = "assistant") -> None:
        """
        Display a message.
        
        Args:
            message: Message content
            role: Message role (user, assistant, system)
        """
        pass
    
    @abstractmethod
    def get_input(self, prompt: Optional[str] = None) -> str:
        """
        Get user input.
        
        Args:
            prompt: Optional prompt to display
            
        Returns:
            User input
        """
        pass
    
    @abstractmethod
    def display_error(self, error: str) -> None:
        """
        Display an error message.
        
        Args:
            error: Error message
        """
        pass
    
    @abstractmethod
    def display_thinking(self) -> None:
        """Display a thinking indicator."""
        pass
    
    @abstractmethod
    def clear_thinking(self) -> None:
        """Clear the thinking indicator."""
        pass


class CommandUIInterface(ABC):
    """Abstract interface for command UI."""
    
    @abstractmethod
    def register_command(self, name: str, handler: Callable, help_text: str) -> None:
        """
        Register a command.
        
        Args:
            name: Command name
            handler: Command handler function
            help_text: Command help text
        """
        pass
    
    @abstractmethod
    def execute_command(self, command: str, args: List[str]) -> Any:
        """
        Execute a command.
        
        Args:
            command: Command name
            args: Command arguments
            
        Returns:
            Command result
        """
        pass
    
    @abstractmethod
    def display_help(self) -> None:
        """Display help for all commands."""
        pass


class UIInterface(ABC):
    """Abstract interface for UI."""
    
    @abstractmethod
    def setup(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the UI.
        
        Args:
            config: Optional configuration dictionary
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """Run the UI."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shut down the UI."""
        pass
    
    @property
    @abstractmethod
    def chat(self) -> ChatUIInterface:
        """Get the chat UI interface."""
        pass
    
    @property
    @abstractmethod
    def consent(self) -> ConsentUIInterface:
        """Get the consent UI interface."""
        pass
    
    @property
    @abstractmethod
    def command(self) -> CommandUIInterface:
        """Get the command UI interface."""
        pass
