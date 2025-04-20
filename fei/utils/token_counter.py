#!/usr/bin/env python3
"""
Token counting utilities for Fei

This module provides utilities for counting tokens in text and messages
for different LLM providers.
"""

import json
from typing import Dict, List, Any, Optional, Union, Callable

# Try to import tiktoken for OpenAI models
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from fei.utils.logging import get_logger

logger = get_logger(__name__)


class TokenCounter:
    """
    Token counter for different LLM providers
    
    This class provides methods for counting tokens in text and messages
    for different LLM providers.
    """
    
    # Default token counts per character for different providers
    # These are rough estimates when tiktoken is not available
    DEFAULT_TOKENS_PER_CHAR = {
        "openai": 0.25,
        "anthropic": 0.25,
        "google": 0.25,
        "groq": 0.25,
        "default": 0.25
    }
    
    # OpenAI model to encoding mapping
    OPENAI_MODEL_TO_ENCODING = {
        # GPT-4 models
        "gpt-4": "cl100k_base",
        "gpt-4-0314": "cl100k_base",
        "gpt-4-0613": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-32k-0314": "cl100k_base",
        "gpt-4-32k-0613": "cl100k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4-0125-preview": "cl100k_base",
        "gpt-4-1106-preview": "cl100k_base",
        "gpt-4-vision-preview": "cl100k_base",
        "gpt-4o": "cl100k_base",
        "gpt-4o-mini": "cl100k_base",
        
        # GPT-3.5 models
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-0301": "cl100k_base",
        "gpt-3.5-turbo-0613": "cl100k_base",
        "gpt-3.5-turbo-1106": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-16k-0613": "cl100k_base",
        "gpt-3.5-turbo-instruct": "cl100k_base",
        
        # Default for unknown models
        "default": "cl100k_base"
    }
    
    def __init__(self, provider: str = "default", model: Optional[str] = None):
        """
        Initialize token counter
        
        Args:
            provider: Provider name (openai, anthropic, google, etc.)
            model: Model name (optional)
        """
        self.provider = provider.lower()
        self.model = model
        
        # Initialize tiktoken encoder if available
        self.tiktoken_encoder = None
        if TIKTOKEN_AVAILABLE and (self.provider == "openai" or self.provider == "default"):
            try:
                if model:
                    # Get encoding for specific model
                    encoding_name = self.OPENAI_MODEL_TO_ENCODING.get(
                        model, self.OPENAI_MODEL_TO_ENCODING["default"]
                    )
                    self.tiktoken_encoder = tiktoken.get_encoding(encoding_name)
                else:
                    # Use default encoding
                    self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            except Exception as e:
                logger.warning(f"Failed to initialize tiktoken encoder: {e}")
                self.tiktoken_encoder = None
    
    def count_tokens_in_text(self, text: str) -> int:
        """
        Count tokens in text
        
        Args:
            text: Text to count tokens in
            
        Returns:
            Token count
        """
        if not text:
            return 0
            
        # Use tiktoken if available
        if self.tiktoken_encoder:
            return len(self.tiktoken_encoder.encode(text))
            
        # Fall back to character-based estimation
        tokens_per_char = self.DEFAULT_TOKENS_PER_CHAR.get(
            self.provider, self.DEFAULT_TOKENS_PER_CHAR["default"]
        )
        return int(len(text) * tokens_per_char)
    
    def count_tokens_in_messages(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count tokens in a list of messages
        
        Args:
            messages: List of messages to count tokens in
            
        Returns:
            Token count
        """
        if not messages:
            return 0
            
        total_tokens = 0
        
        for message in messages:
            # Count tokens in message content
            content = message.get("content", "")
            
            # Handle content that could be a list (for OpenAI messages with multiple parts)
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        # Handle different content types
                        if part.get("type") == "text":
                            total_tokens += self.count_tokens_in_text(part.get("text", ""))
                        # Add other content types as needed
                    elif isinstance(part, str):
                        total_tokens += self.count_tokens_in_text(part)
            else:
                total_tokens += self.count_tokens_in_text(content)
            
            # Count tokens in function calls if present
            if "tool_calls" in message:
                for tool_call in message["tool_calls"]:
                    # Add tokens for function name
                    if "function" in tool_call:
                        total_tokens += self.count_tokens_in_text(tool_call["function"].get("name", ""))
                        
                        # Add tokens for function arguments
                        args = tool_call["function"].get("arguments", "")
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        total_tokens += self.count_tokens_in_text(args)
            
            # Add tokens for message role
            total_tokens += self.count_tokens_in_text(message.get("role", ""))
            
            # Add tokens for message name if present
            if "name" in message:
                total_tokens += self.count_tokens_in_text(message["name"])
        
        # Add a small overhead for message formatting
        total_tokens += len(messages) * 4
        
        return total_tokens
