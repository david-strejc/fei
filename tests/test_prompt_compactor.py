#!/usr/bin/env python3
"""
Tests for the Prompt Compactor
"""

import os
import sys
import unittest
import json
from unittest.mock import patch, MagicMock
import asyncio

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.core.prompt_compactor import PromptCompactor
from fei.utils.config import Config


class TestPromptCompactor(unittest.TestCase):
    """Test cases for the Prompt Compactor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = Config()
        
        # Create a prompt compactor with test settings
        self.compactor = PromptCompactor(
            config=self.config,
            provider="anthropic",
            model="claude-3-7-sonnet-20250219",
            tail_k=3,
            target_tokens=1000,
            enabled=True
        )
        
        # Sample messages for testing
        self.sample_messages = [
            {"role": "user", "content": "Hello, I need help with Python."},
            {"role": "assistant", "content": "I'd be happy to help with Python. What do you need?"},
            {"role": "user", "content": "How do I read a file?"},
            {"role": "assistant", "content": "You can use the `open()` function to read a file in Python."},
            {"role": "user", "content": "Can you show an example?"},
            {"role": "assistant", "content": "Sure, here's an example:\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\nprint(content)\n```"},
        ]
    
    @patch('fei.core.prompt_compactor.PromptCompactor.summarize_async')
    def test_add_message(self, mock_summarize):
        """Test adding messages to the compactor"""
        # Set up mock
        mock_summarize.return_value = "Conversation about Python file handling."
        
        # Add messages
        for msg in self.sample_messages:
            self.compactor.add_message(msg)
        
        # Check that messages were added to raw_log
        self.assertEqual(len(self.compactor.raw_log), len(self.sample_messages))
    
    def test_context_for_llm(self):
        """Test getting context for LLM"""
        # Add messages
        for msg in self.sample_messages:
            self.compactor.add_message(msg)
        
        # Get context
        context = self.compactor.context_for_llm()
        
        # Check that context has system message and tail messages
        self.assertEqual(len(context), 4)  # system + 3 tail messages
        self.assertEqual(context[0]["role"], "system")
    
    @patch('fei.core.prompt_compactor.PromptCompactor.summarize_async')
    def test_compaction(self, mock_summarize):
        """Test compaction logic"""
        # Set up mock
        mock_summarize.return_value = "Conversation about Python file handling."
        
        # Create a compactor with very low token target to force compaction
        compactor = PromptCompactor(
            config=self.config,
            provider="anthropic",
            model="claude-3-7-sonnet-20250219",
            tail_k=2,
            target_tokens=10,  # Very low to force compaction
            enabled=True
        )
        
        # Patch _needs_compaction to always return True after 4 messages
        with patch.object(compactor, '_needs_compaction', return_value=True):
            # Add messages
            for msg in self.sample_messages:
                compactor.add_message(msg)
            
            # Check that compaction was triggered
            self.assertTrue(mock_summarize.called)
            
            # Check that we have the right number of messages after compaction
            self.assertLessEqual(len(compactor.raw_log), 4)  # Should have at most 4 messages
    
    def test_get_compacted_messages(self):
        """Test getting compacted messages"""
        # Set summary
        self.compactor.summary = "This is a test summary."
        
        # Get compacted messages
        compacted = self.compactor.get_compacted_messages(self.sample_messages)
        
        # Check that we have the right structure
        self.assertEqual(len(compacted), 4)  # system + 3 tail messages
        self.assertEqual(compacted[0]["role"], "system")
        self.assertTrue("This is a test summary" in compacted[0]["content"])
        
        # Check that tail messages are preserved
        self.assertEqual(compacted[1:], self.sample_messages[-3:])
    
    def test_reset(self):
        """Test resetting the compactor"""
        # Add messages
        for msg in self.sample_messages:
            self.compactor.add_message(msg)
        
        # Set a custom summary
        self.compactor.summary = "Custom summary."
        
        # Reset
        self.compactor.reset()
        
        # Check that state was reset
        self.assertEqual(len(self.compactor.raw_log), 0)
        self.assertEqual(self.compactor.summary, "The dialogue has not been summarized yet.")


if __name__ == '__main__':
    unittest.main()
