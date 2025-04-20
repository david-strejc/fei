#!/usr/bin/env python3
"""
Continuous task execution implementation for Fei

This module provides a TaskExecutor class that continues executing
a task until completion without requiring user intervention.
"""

import os
import json
import asyncio
import sys
import time
import uuid
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable, Tuple
from dataclasses import dataclass

from fei.core.assistant import Assistant
from fei.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class TaskContext:
    """Context for task execution"""
    task_id: str
    iteration: int
    start_time: float
    messages: List[str]
    tool_outputs: List[Dict[str, Any]]
    

class TaskExecutionError(Exception):
    """Exception raised for task execution errors"""
    pass


class TaskExecutor:
    """
    TaskExecutor class for continuous task execution
    
    This class takes an Assistant instance and runs a task continuously
    until completion is detected or the task is manually interrupted.
    """
    
    def __init__(
        self,
        assistant: Assistant,
        task_completion_signal: str = "[TASK_COMPLETE]",
        on_message: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the TaskExecutor
        
        Args:
            assistant: The Assistant instance to use for execution
            task_completion_signal: Special string that signals task completion
            on_message: Optional callback function to invoke for each message
        """
        self.assistant = assistant
        self.task_completion_signal = task_completion_signal
        self.on_message = on_message or (lambda msg: print(msg))
        
        # Task context for maintaining state between iterations
        self.task_contexts = {}
    
    async def _process_assistant_response(
        self, 
        response: str, 
        context: TaskContext
    ) -> Tuple[str, bool]:
        """
        Process assistant response
        
        Args:
            response: Assistant response
            context: Task context
            
        Returns:
            Tuple of (processed_response, is_complete)
        """
        # Check if response is None or empty
        if not response or response.strip() == "None":
            # Check for tool outputs in the conversation
            tool_outputs = self._extract_tool_outputs()
            
            if tool_outputs:
                # Add to context
                context.tool_outputs.extend(tool_outputs)
                
                # Create a response message with tool outputs
                response = "I've executed the command(s). Here are the results:\n\n" + "\n\n".join(
                    output.get("display_text", f"Command output:\n{output.get('stdout', '')}")
                    for output in tool_outputs
                )
            else:
                response = "Command executed, but no output was returned."
        
        # Check for completion signal
        is_complete = False
        if response and self.task_completion_signal in response:
            # Remove the signal from the output
            response = response.replace(self.task_completion_signal, "").strip()
            is_complete = True
        
        # Add to message history
        context.messages.append(response)
        
        return response, is_complete
    
    def _extract_tool_outputs(self) -> List[Dict[str, Any]]:
        """
        Extract tool outputs from conversation
        
        Returns:
            List of tool outputs
        """
        tool_outputs = []
        conversation = self.assistant.conversation
        
        # Go through the last few messages to extract tool results
        for msg in reversed(conversation[-5:] if len(conversation) >= 5 else conversation):
            if msg.get("role") == "tool":
                try:
                    # Parse tool content if it's a string
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.startswith("{") and content.endswith("}"):
                        try:
                            tool_result = json.loads(content)
                            
                            # Format output for display
                            display_text = None
                            if "stdout" in tool_result and tool_result["stdout"]:
                                display_text = f"Command output:\n{tool_result['stdout']}"
                            
                            tool_outputs.append({
                                "tool_name": msg.get("name", "unknown"),
                                "tool_id": msg.get("tool_call_id", ""),
                                "result": tool_result,
                                "stdout": tool_result.get("stdout", ""),
                                "stderr": tool_result.get("stderr", ""),
                                "display_text": display_text
                            })
                        except json.JSONDecodeError:
                            # Not JSON, use as is
                            tool_outputs.append({
                                "tool_name": msg.get("name", "unknown"),
                                "tool_id": msg.get("tool_call_id", ""),
                                "result": content,
                                "display_text": f"Tool output:\n{content}"
                            })
                except Exception as e:
                    logger.warning(f"Error parsing tool result: {e}")
        
        return tool_outputs
    
    async def _execute_task_iteration(
        self, 
        task: str, 
        context: TaskContext,
        interactive: bool = False
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Execute a single task iteration
        
        Args:
            task: Task to execute
            context: Task context
            interactive: Whether to use interactive mode
            
        Returns:
            Tuple of (response, is_complete, next_task)
        """
        try:
            # Get response from assistant
            response = await self.assistant.chat(task)
            
            # Process response
            processed_response, is_complete = await self._process_assistant_response(response, context)
            
            # Display the response
            self.on_message(processed_response)
            
            # If interactive, get user input
            next_task = None
            if interactive and not is_complete:
                user_input = input("\nType 'continue' to proceed, 'stop' to end, or enter custom instructions: ")
                
                if user_input.lower() == 'stop':
                    return processed_response, True, None
                elif user_input.lower() == 'continue':
                    next_task = "Continue with the next step of the task."
                else:
                    # Use custom input as the next task instruction
                    next_task = user_input
            
            return processed_response, is_complete, next_task
            
        except Exception as e:
            error_msg = f"Error in task execution: {str(e)}"
            logger.error(error_msg)
            self.on_message(f"Error: {str(e)}")
            raise TaskExecutionError(error_msg)
    
    async def execute_task(self, initial_task: str, max_iterations: int = 10) -> str:
        """
        Execute a task continuously until completion
        
        Args:
            initial_task: The initial task description
            max_iterations: Maximum number of iterations before stopping
            
        Returns:
            Final message or status
        """
        # Create task context
        task_id = str(uuid.uuid4())
        context = TaskContext(
            task_id=task_id,
            iteration=0,
            start_time=time.time(),
            messages=[],
            tool_outputs=[]
        )
        self.task_contexts[task_id] = context
        
        # Initial task
        current_task = initial_task
        
        try:
            while context.iteration < max_iterations:
                context.iteration += 1
                logger.info(f"Task iteration {context.iteration}/{max_iterations}")

                # --- Refactored Loop Logic ---
                logger.debug(f"Sending task/continuation to LLM: {current_task}")

                # Get response and potential tool calls from assistant
                response_content, tool_calls = await self.assistant._send_message_to_llm(
                    self.assistant.conversation_manager.get_messages() + [{"role": "user", "content": current_task}], # Send current history + task
                    tools=self.assistant.get_tools()
                )

                # Add the user message (task instruction) and assistant response (potentially with tool calls)
                self.assistant.conversation_manager.add_user_message(current_task) # Add the instruction that was sent
                self.assistant.conversation_manager.add_assistant_message(response_content, tool_calls) # Add LLM response

                processed_response, is_complete = await self._process_assistant_response(response_content, context)
                self.on_message(processed_response) # Display intermediate response

                if is_complete:
                    logger.info("Task completed successfully (completion signal found).")
                    elapsed_time = time.time() - context.start_time
                    return f"Task completed in {context.iteration} iterations ({elapsed_time:.2f}s)"

                # If there were tool calls, execute them and get the continuation response
                if tool_calls:
                    logger.info(f"Processing {len(tool_calls)} tool calls for iteration {context.iteration}")
                    tool_results = await self.assistant.process_tool_calls(tool_calls)
                    self.assistant.conversation_manager.add_tool_results(tool_results)

                    logger.info("Continuing conversation with tool results...")
                    # Send continuation message to get final response after tool execution
                    # The _send_continuation method already adds the final assistant response to history
                    final_answer_after_tools = await self.assistant._send_continuation(self.assistant.get_tools())

                    processed_final_answer, is_complete_after_tools = await self._process_assistant_response(final_answer_after_tools, context)
                    self.on_message(processed_final_answer) # Display final answer for this iteration

                    if is_complete_after_tools:
                        logger.info("Task completed successfully (completion signal found after tool use).")
                        elapsed_time = time.time() - context.start_time
                        return f"Task completed in {context.iteration} iterations ({elapsed_time:.2f}s)"

                # Set next task instruction for the loop
                current_task = "Continue with the next step of the task based on the conversation history."
                # --- End Refactored Loop Logic ---

                # Add a small delay to avoid spinning too fast
                await asyncio.sleep(0.5)

            # Max iterations reached
            logger.warning(f"Task reached max iterations ({max_iterations})")
            return f"Task did not complete within {max_iterations} iterations"

        finally:
            # Clean up task context
            self.task_contexts.pop(task_id, None)
    
    async def execute_interactive(self, initial_task: str) -> str:
        """
        Execute a task with interactive control
        
        This version allows the user to type 'continue', 'stop', or custom input
        between iterations.
        
        Args:
            initial_task: The initial task description
            
        Returns:
            Final message or status
        """
        # Create task context
        task_id = str(uuid.uuid4())
        context = TaskContext(
            task_id=task_id,
            iteration=0,
            start_time=time.time(),
            messages=[],
            tool_outputs=[]
        )
        self.task_contexts[task_id] = context
        
        # Initial task
        current_task = initial_task
        
        try:
            while True:
                context.iteration += 1
                logger.info(f"Task iteration {context.iteration} (interactive)")
                
                # Execute iteration
                _, is_complete, next_task = await self._execute_task_iteration(
                    current_task, 
                    context,
                    interactive=True
                )
                
                # Check if task is complete
                if is_complete:
                    logger.info("Task completed successfully")
                    elapsed_time = time.time() - context.start_time
                    return f"Task completed in {context.iteration} iterations ({elapsed_time:.2f}s)"
                
                # Check if task was stopped
                if next_task is None:
                    logger.info("Task stopped by user")
                    return "Task stopped by user"
                
                # Set next task
                current_task = next_task
                
        finally:
            # Clean up task context
            self.task_contexts.pop(task_id, None)
