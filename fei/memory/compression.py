"""
Memory compression for Fei memory system.

This module provides utilities for compressing and summarizing
memories to reduce storage requirements and improve retrieval.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple

from fei.utils.logging import get_logger
from fei.utils.config import Config

logger = get_logger(__name__)


class MemoryCompressor:
    """
    Memory compressor for summarizing and compressing memories.
    
    This class provides methods for summarizing and compressing
    memories to reduce storage requirements and improve retrieval.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the memory compressor.
        
        Args:
            config: Configuration object
            provider: Provider to use (anthropic, openai, etc.)
            model: Model to use
            api_key: API key (overrides config)
        """
        self.config = config or Config()
        
        # Set up provider
        self.provider = provider or self.config.get("compression_provider", "anthropic")
        
        # Set up model
        self.model = model or self._get_default_model()
        
        # Set up API key
        self.api_key = api_key or self._get_api_key()
    
    def _get_default_model(self) -> str:
        """
        Get the default model for the provider.
        
        Returns:
            Default model name
        """
        # Provider-specific default models
        defaults = {
            "anthropic": "claude-3-haiku-20240307",
            "openai": "gpt-3.5-turbo",
            "google": "gemini-1.5-flash"
        }
        
        return (
            self.config.get(f"{self.provider}.compression_model") or
            self.config.get("compression_model") or
            defaults.get(self.provider, "claude-3-haiku-20240307")
        )
    
    def _get_api_key(self) -> Optional[str]:
        """
        Get the API key for the provider.
        
        Returns:
            API key or None
        """
        # Provider-specific API key environment variables
        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY"
        }
        
        key_env_var = key_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
        
        return (
            self.config.get(f"{self.provider}.api_key") or
            self.config.get("api_key") or
            self.config.get(key_env_var)
        )
    
    async def summarize(
        self,
        content: str,
        max_length: Optional[int] = None,
        preserve_keywords: Optional[List[str]] = None
    ) -> str:
        """
        Summarize memory content.
        
        Args:
            content: Memory content to summarize
            max_length: Maximum length of summary (in characters)
            preserve_keywords: Keywords to preserve in summary
            
        Returns:
            Summarized content
            
        Raises:
            Exception: If summarization fails
        """
        try:
            # Import here to avoid circular imports
            from litellm import completion as litellm_completion
            
            # Prepare prompt
            prompt = self._create_summarization_prompt(content, max_length, preserve_keywords)
            
            # Set up parameters
            params = {
                "model": f"{self.provider}/{self.model}",
                "messages": [
                    {"role": "system", "content": "You are a memory summarization assistant. Your task is to create concise, accurate summaries of text while preserving key information."},
                    {"role": "user", "content": prompt}
                ],
                "api_key": self.api_key,
                "max_tokens": 1000
            }
            
            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: litellm_completion(**params)
            )
            
            # Extract summary
            summary = self._extract_response_content(response)
            
            return summary
        except Exception as e:
            logger.error(f"Error summarizing memory: {e}")
            # Return original content on error
            return content
    
    def _create_summarization_prompt(
        self,
        content: str,
        max_length: Optional[int] = None,
        preserve_keywords: Optional[List[str]] = None
    ) -> str:
        """
        Create a prompt for summarization.
        
        Args:
            content: Memory content to summarize
            max_length: Maximum length of summary (in characters)
            preserve_keywords: Keywords to preserve in summary
            
        Returns:
            Summarization prompt
        """
        prompt = "Please summarize the following text:\n\n"
        prompt += content
        
        if max_length:
            prompt += f"\n\nPlease keep the summary under {max_length} characters."
        
        if preserve_keywords:
            keywords_str = ", ".join(preserve_keywords)
            prompt += f"\n\nMake sure to preserve these key concepts in the summary: {keywords_str}."
        
        prompt += "\n\nSummary:"
        
        return prompt
    
    def _extract_response_content(self, response: Any) -> str:
        """
        Extract content from a response.
        
        Args:
            response: Response from the LLM provider
            
        Returns:
            Response content
        """
        try:
            # Handle different response formats
            if hasattr(response, 'choices') and response.choices:
                # OpenAI format
                message = response.choices[0].message
                
                if hasattr(message, 'content') and message.content is not None:
                    return message.content
                
                # Handle content list (Anthropic via LiteLLM)
                if hasattr(message, 'content') and isinstance(message.content, list):
                    text_parts = []
                    for content_block in message.content:
                        if content_block.get('type') == 'text':
                            text_parts.append(content_block.get('text', ''))
                    return ''.join(text_parts)
            
            # Direct Anthropic format
            if hasattr(response, 'content') and isinstance(response.content, list):
                text_parts = []
                for content_block in response.content:
                    if content_block.type == 'text':
                        text_parts.append(content_block.text)
                return ''.join(text_parts)
            
            # Simple content attribute
            if hasattr(response, 'content') and isinstance(response.content, str):
                return response.content
            
            # Fallback to string representation
            return str(response)
        except Exception as e:
            logger.error(f"Error extracting response content: {e}")
            return ""
    
    async def compress_memories(
        self,
        memories: List[Dict[str, Any]],
        max_memories: int = 5,
        strategy: str = "summarize"
    ) -> Dict[str, Any]:
        """
        Compress multiple memories into a single memory.
        
        Args:
            memories: List of memories to compress
            max_memories: Maximum number of memories to include
            strategy: Compression strategy ("summarize", "concatenate", "extract_key_points")
            
        Returns:
            Compressed memory
            
        Raises:
            Exception: If compression fails
        """
        try:
            # Limit number of memories
            if len(memories) > max_memories:
                # Sort by timestamp if available
                sorted_memories = sorted(
                    memories,
                    key=lambda m: m.get("metadata", {}).get("timestamp", ""),
                    reverse=True
                )
                memories_to_compress = sorted_memories[:max_memories]
            else:
                memories_to_compress = memories
            
            # Apply compression strategy
            if strategy == "summarize":
                return await self._compress_by_summarization(memories_to_compress)
            elif strategy == "concatenate":
                return self._compress_by_concatenation(memories_to_compress)
            elif strategy == "extract_key_points":
                return await self._compress_by_key_points(memories_to_compress)
            else:
                logger.warning(f"Unknown compression strategy: {strategy}, falling back to summarization")
                return await self._compress_by_summarization(memories_to_compress)
        except Exception as e:
            logger.error(f"Error compressing memories: {e}")
            # Return a simple concatenation on error
            return self._compress_by_concatenation(memories_to_compress)
    
    async def _compress_by_summarization(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compress memories by summarizing their content.
        
        Args:
            memories: List of memories to compress
            
        Returns:
            Compressed memory
        """
        # Concatenate memory contents
        combined_content = "\n\n".join([
            f"Memory {i+1}: {memory.get('content', '')}"
            for i, memory in enumerate(memories)
        ])
        
        # Summarize combined content
        summary = await self.summarize(combined_content)
        
        # Create compressed memory
        compressed_memory = {
            "content": summary,
            "metadata": {
                "compressed": True,
                "compression_strategy": "summarize",
                "original_count": len(memories),
                "original_ids": [memory.get("id") for memory in memories if "id" in memory],
                "timestamp": memories[0].get("metadata", {}).get("timestamp", "")
            }
        }
        
        return compressed_memory
    
    def _compress_by_concatenation(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compress memories by concatenating their content.
        
        Args:
            memories: List of memories to compress
            
        Returns:
            Compressed memory
        """
        # Concatenate memory contents
        combined_content = "\n\n".join([
            f"Memory {i+1}: {memory.get('content', '')}"
            for i, memory in enumerate(memories)
        ])
        
        # Create compressed memory
        compressed_memory = {
            "content": combined_content,
            "metadata": {
                "compressed": True,
                "compression_strategy": "concatenate",
                "original_count": len(memories),
                "original_ids": [memory.get("id") for memory in memories if "id" in memory],
                "timestamp": memories[0].get("metadata", {}).get("timestamp", "")
            }
        }
        
        return compressed_memory
    
    async def _compress_by_key_points(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compress memories by extracting key points.
        
        Args:
            memories: List of memories to compress
            
        Returns:
            Compressed memory
        """
        try:
            # Import here to avoid circular imports
            from litellm import completion as litellm_completion
            
            # Prepare prompt
            prompt = "Please extract the key points from these memories:\n\n"
            for i, memory in enumerate(memories):
                prompt += f"Memory {i+1}: {memory.get('content', '')}\n\n"
            
            prompt += "Key points (in bullet point format):"
            
            # Set up parameters
            params = {
                "model": f"{self.provider}/{self.model}",
                "messages": [
                    {"role": "system", "content": "You are a memory extraction assistant. Your task is to extract key points from memories while preserving important information."},
                    {"role": "user", "content": prompt}
                ],
                "api_key": self.api_key,
                "max_tokens": 1000
            }
            
            # Use proper async pattern with executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: litellm_completion(**params)
            )
            
            # Extract key points
            key_points = self._extract_response_content(response)
            
            # Create compressed memory
            compressed_memory = {
                "content": key_points,
                "metadata": {
                    "compressed": True,
                    "compression_strategy": "extract_key_points",
                    "original_count": len(memories),
                    "original_ids": [memory.get("id") for memory in memories if "id" in memory],
                    "timestamp": memories[0].get("metadata", {}).get("timestamp", "")
                }
            }
            
            return compressed_memory
        except Exception as e:
            logger.error(f"Error extracting key points: {e}")
            # Fall back to concatenation on error
            return self._compress_by_concatenation(memories)
