"""
Conversation manager for Fei code assistant.

This module contains the ConversationManager class, which handles
conversation state and message formatting.
"""

import json
from typing import Dict, List, Optional, Any

from fei.utils.logging import get_logger
from fei.core.mediator import Mediator

logger = get_logger(__name__)


class ConversationManager:
    """
    Manages conversation state and message formatting.
    
    This class is responsible for:
    1. Maintaining conversation history
    2. Formatting messages for different providers
    3. Adding user, assistant, and tool messages
    """
    
    def __init__(self, provider: str, mediator: Optional[Mediator] = None):
        """
        Initialize conversation manager.
        
        Args:
            provider: Provider name for proper formatting
            mediator: Mediator for component communication
        """
        self.provider = provider
        self.mediator = mediator
        self.conversation = []
        self.last_message_id = None
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation.
        
        Args:
            message: User message content
        """
        self.conversation.append({"role": "user", "content": message})
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("user_message_added", {
                "message": message,
                "conversation_length": len(self.conversation)
            })
    
    def add_assistant_message(self, content: str, tool_calls: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: Assistant message content
            tool_calls: Optional tool calls for the message
        """
        assistant_message = {"role": "assistant", "content": content}
        
        if tool_calls:
            if isinstance(assistant_message["content"], str):
                assistant_message["content"] = [{"type": "text", "text": assistant_message["content"]}]
                
            # Add tool_calls field to match OpenAI format    
            assistant_message["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["input"] if isinstance(tc["input"], str) else json.dumps(tc["input"])
                    }
                } for tc in tool_calls
            ]
        
        self.conversation.append(assistant_message)
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("assistant_message_added", {
                "content": content,
                "has_tool_calls": bool(tool_calls),
                "tool_call_count": len(tool_calls) if tool_calls else 0,
                "conversation_length": len(self.conversation)
            })
    
    def add_tool_results(self, tool_results: List[Dict[str, Any]]) -> None:
        """
        Add tool results to the conversation.
        
        Args:
            tool_results: Tool execution results
        """
        if self.provider == "anthropic":
            # Anthropic tool results format
            for result in tool_results:
                tool_result_message = {
                    "role": "tool", 
                    "tool_call_id": result["tool_use_id"],
                    "name": result["name"],
                    "content": result["content"]
                }
                # Add each tool result as a separate message
                self.conversation.append(tool_result_message)
        else:
            # Format for other providers
            for result in tool_results:
                # Add each result as a separate tool message for better visibility
                self.conversation.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["content"]
                })
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("tool_results_added", {
                "result_count": len(tool_results),
                "conversation_length": len(self.conversation)
            })
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """
        Get current conversation messages.
        
        Returns:
            List of conversation messages
        """
        return self.conversation
    
    def reset(self) -> None:
        """Reset the conversation history."""
        self.conversation = []
        self.last_message_id = None
        
        # Notify mediator if available
        if self.mediator:
            self.mediator.notify("conversation_reset", {})
