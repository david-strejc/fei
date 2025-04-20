"""
Tests for the task executor.
"""

import pytest
from unittest.mock import MagicMock, patch
import asyncio

from fei.core.task_executor import TaskExecutor


class TestTaskExecutor:
    """Test the TaskExecutor class."""
    
    def setup_method(self):
        """Set up the test."""
        # Create mock assistant
        self.assistant = MagicMock()
        
        # Set up mock chat method
        async def mock_chat(message):
            if "TASK_COMPLETE" in message:
                return "Task completed"
            return "Working on task..."
        
        self.assistant.chat = MagicMock(side_effect=mock_chat)
        
        # Create mock on_message callback
        self.on_message = MagicMock()
        
        # Create task executor
        self.executor = TaskExecutor(
            assistant=self.assistant,
            on_message=self.on_message
        )
    
    @pytest.mark.asyncio
    async def test_execute_task_completion(self):
        """Test execute_task with task completion."""
        # Set up mock chat to complete task on second iteration
        async def mock_chat(message):
            if self.assistant.chat.call_count == 2:
                return "Task completed [TASK_COMPLETE]"
            return "Working on task..."
        
        self.assistant.chat = MagicMock(side_effect=mock_chat)
        
        # Execute task
        result = await self.executor.execute_task("Test task", max_iterations=5)
        
        # Check if assistant.chat was called with correct arguments
        assert self.assistant.chat.call_count == 2
        
        # First call should be with the task
        args1, kwargs1 = self.assistant.chat.call_args_list[0]
        assert "Test task" in args1[0]
        
        # Second call should be with "continue"
        args2, kwargs2 = self.assistant.chat.call_args_list[1]
        assert "continue" in args2[0].lower()
        
        # Check if on_message was called with correct arguments
        assert self.on_message.call_count == 2
        self.on_message.assert_any_call("Working on task...")
        self.on_message.assert_any_call("Task completed [TASK_COMPLETE]")
        
        # Check if result is correct
        assert result == "COMPLETED"
    
    @pytest.mark.asyncio
    async def test_execute_task_max_iterations(self):
        """Test execute_task with max iterations reached."""
        # Set up mock chat to never complete task
        async def mock_chat(message):
            return "Working on task..."
        
        self.assistant.chat = MagicMock(side_effect=mock_chat)
        
        # Execute task
        result = await self.executor.execute_task("Test task", max_iterations=3)
        
        # Check if assistant.chat was called with correct arguments
        assert self.assistant.chat.call_count == 3
        
        # First call should be with the task
        args1, kwargs1 = self.assistant.chat.call_args_list[0]
        assert "Test task" in args1[0]
        
        # Subsequent calls should be with "continue"
        args2, kwargs2 = self.assistant.chat.call_args_list[1]
        assert "continue" in args2[0].lower()
        
        args3, kwargs3 = self.assistant.chat.call_args_list[2]
        assert "continue" in args3[0].lower()
        
        # Check if on_message was called with correct arguments
        assert self.on_message.call_count == 3
        for i in range(3):
            self.on_message.assert_any_call("Working on task...")
        
        # Check if result is correct
        assert result == "MAX_ITERATIONS"
    
    @pytest.mark.asyncio
    async def test_execute_task_with_exception(self):
        """Test execute_task with exception."""
        # Set up mock chat to raise exception
        async def mock_chat(message):
            raise ValueError("Test error")
        
        self.assistant.chat = MagicMock(side_effect=mock_chat)
        
        # Execute task
        result = await self.executor.execute_task("Test task", max_iterations=3)
        
        # Check if assistant.chat was called
        self.assistant.chat.assert_called_once()
        
        # Check if on_message was not called
        self.on_message.assert_not_called()
        
        # Check if result is correct
        assert result == "ERROR"
    
    @pytest.mark.asyncio
    async def test_execute_task_with_no_callback(self):
        """Test execute_task with no callback."""
        # Create task executor without callback
        executor = TaskExecutor(assistant=self.assistant)
        
        # Set up mock chat to complete task on second iteration
        async def mock_chat(message):
            if self.assistant.chat.call_count == 2:
                return "Task completed [TASK_COMPLETE]"
            return "Working on task..."
        
        self.assistant.chat = MagicMock(side_effect=mock_chat)
        
        # Execute task
        result = await executor.execute_task("Test task", max_iterations=5)
        
        # Check if assistant.chat was called with correct arguments
        assert self.assistant.chat.call_count == 2
        
        # Check if result is correct
        assert result == "COMPLETED"
