#!/usr/bin/env python3
"""
FEI Memorychain Integration Example

This example demonstrates how to integrate the Memorychain distributed memory system
with FEI (Flying Dragon of Adaptability), enabling:

1. Memory-based conversations with FEI
2. Saving conversation highlights to the Memorychain
3. Referencing memories in conversations using #mem:id syntax
4. Sharing memories across multiple FEI instances

Prerequisites:
- A running Memorychain node (start with `python -m memdir_tools.memorychain_cli start`)
- FEI installed and configured

Usage:
python fei_memorychain_example.py
"""

import os
import sys
import json
import time
import re
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union, Set

# Import core FEI components
from fei.core.assistant import Agent
from fei.utils.config import load_api_key

# Import Memorychain connector
from fei.tools.memorychain_connector import MemorychainConnector, get_connector

class MemorychainFEIAssistant:
    """
    FEI assistant with Memorychain integration
    
    This class extends FEI with Memorychain capabilities, allowing it to:
    - Save conversation highlights to the distributed memory chain
    - Reference and retrieve memories using #mem: syntax
    - Share memories across multiple FEI instances
    """
    
    # Status constants
    STATUS_IDLE = "idle"
    STATUS_BUSY = "busy"
    STATUS_WORKING_ON_TASK = "working_on_task"
    STATUS_SOLUTION_PROPOSED = "solution_proposed"
    STATUS_TASK_COMPLETED = "task_completed"
    
    def __init__(self, 
               api_key: Optional[str] = None, 
               model: str = "claude-3-opus-20240229", 
               node_address: Optional[str] = None):
        """
        Initialize the MemorychainFEI assistant
        
        Args:
            api_key: API key for the LLM (Claude API)
            model: Model to use
            node_address: Memorychain node address (ip:port)
        """
        # Initialize FEI assistant
        self.api_key = api_key or load_api_key()
        self.model = model
        self.assistant = Agent(api_key=self.api_key, model=self.model)
        
        # Initialize Memorychain connector
        self.memorychain = get_connector(node_address)
        
        # Initialize conversation history
        self.conversation = []
        
        # Initial system prompt
        self.system_prompt = """
You are an AI assistant with access to a distributed memory system called Memorychain. 
You can access memories using #mem:id syntax. When you see this syntax, I'll provide the memory content.
You can also suggest saving important information to the memory system.

Commands:
- save: Save part of the conversation as a memory
- search [query]: Search for memories
- list: List recent memories
- view [id]: View a specific memory
- status: Show the status of all nodes in the network
- help: Show available commands

Respond directly and helpfully to the user's questions, referencing memories when appropriate.
"""
        # Update assistant with system prompt
        self.assistant.update_system(self.system_prompt)
        
        # Update node status with AI model and idle status
        try:
            self.memorychain.update_status(
                status=self.STATUS_IDLE,
                ai_model=self.model,
                load=0.0
            )
            print(f"Connected to Memorychain node and updated status")
        except Exception as e:
            print(f"Warning: Could not update status: {e}")
    
    def handle_command(self, command: str) -> str:
        """
        Process special memory commands
        
        Args:
            command: Command string
            
        Returns:
            Response message
        """
        # Parse the command
        parts = command.strip().split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        try:
            # save - Save a memory
            if cmd == "save":
                subject = "Conversation Highlight"
                
                # Get last few turns of conversation (up to 5)
                last_turns = self.conversation[-10:]
                
                # Format content from these turns
                content = ""
                for turn in last_turns:
                    role = turn.get("role", "")
                    msg = turn.get("content", "")
                    
                    if role == "user":
                        content += f"User: {msg}\n\n"
                    elif role == "assistant":
                        content += f"Assistant: {msg}\n\n"
                
                # Try to extract a good subject line from first user message
                for turn in last_turns:
                    if turn.get("role") == "user":
                        text = turn.get("content", "")
                        # Use first ~5 words as subject
                        words = text.split()
                        if words:
                            subject = " ".join(words[:5])
                            if len(words) > 5:
                                subject += "..."
                        break
                
                # Determine tags from content (could be more sophisticated)
                tags = "conversation"
                
                # Add to memorychain
                result = self.memorychain.add_memory(
                    subject=subject,
                    content=content,
                    tags=tags,
                    flags="F"  # Flagged by default
                )
                
                memory_id = ""
                # Extract memory ID from result
                if isinstance(result, dict):
                    message = result.get("message", "")
                    # Try to extract the memory ID from message
                    id_match = re.search(r'memory accepted.*?([a-z0-9-]+)', message, re.IGNORECASE)
                    if id_match:
                        memory_id = id_match.group(1)
                
                return f"Saved to Memorychain with subject: '{subject}'\nReference with: #mem:{memory_id}"
            
            # search - Search for memories
            elif cmd == "search":
                if not args:
                    return "Please provide a search query. Usage: search [query]"
                
                memories = self.memorychain.search_memories(args)
                
                if not memories:
                    return f"No memories found matching query: '{args}'"
                    
                # Format results
                response = f"Found {len(memories)} memories matching '{args}':\n\n"
                
                for memory in memories:
                    metadata = memory.get("metadata", {})
                    headers = memory.get("headers", {})
                    
                    memory_id = metadata.get("unique_id", "unknown")
                    subject = headers.get("Subject", "No subject")
                    tags = headers.get("Tags", "")
                    
                    response += f"- {subject} (#{memory_id})\n"
                    if tags:
                        response += f"  Tags: {tags}\n"
                    response += f"  Reference with: #mem:{memory_id}\n\n"
                
                return response
            
            # list - List recent memories
            elif cmd == "list":
                # Get chain and extract memories (skip genesis block)
                chain = self.memorychain.get_chain()
                
                # Take last 10 (excluding genesis block)
                recent_blocks = [block for block in chain if block["index"] > 0][-10:]
                recent_memories = [block["memory_data"] for block in recent_blocks]
                
                if not recent_memories:
                    return "No memories found in the chain."
                
                # Format results
                response = f"Recent memories in the chain ({len(recent_memories)}):\n\n"
                
                for memory in recent_memories:
                    metadata = memory.get("metadata", {})
                    headers = memory.get("headers", {})
                    
                    memory_id = metadata.get("unique_id", "unknown")
                    subject = headers.get("Subject", "No subject")
                    timestamp = metadata.get("timestamp", 0)
                    date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
                    
                    response += f"- {subject} (#{memory_id}, {date_str})\n"
                    response += f"  Reference with: #mem:{memory_id}\n\n"
                
                return response
            
            # view - View a specific memory
            elif cmd == "view":
                if not args:
                    return "Please provide a memory ID. Usage: view [id]"
                
                memory = self.memorychain.get_memory_by_id(args)
                
                if not memory:
                    return f"Memory not found: {args}"
                
                # Return formatted memory
                return self.memorychain.format_memory(memory)
            
            # status - Show node status
            elif cmd == "status":
                try:
                    # Get network status
                    network_status = self.memorychain.get_network_status()
                    nodes = network_status.get("nodes", {})
                    
                    if not nodes:
                        return "No nodes found in the network."
                        
                    response = "\n=== Memorychain Network Status ===\n"
                    response += f"Total nodes: {len(nodes)}\n"
                    response += f"Network load: {network_status.get('network_load', 0.0):.2f}\n\n"
                    response += "Node Details:\n"
                    
                    for node_id, status in nodes.items():
                        status_str = status.get("status", "unknown")
                        model = status.get("ai_model", "unknown")
                        load = status.get("load", 0.0)
                        task = status.get("current_task_id", "-")
                        
                        response += f"- Node: {node_id}\n"
                        response += f"  Status: {status_str}\n"
                        response += f"  Model: {model}\n"
                        response += f"  Load: {load:.2f}\n"
                        response += f"  Task: {task}\n\n"
                        
                    return response
                    
                except Exception as e:
                    return f"Error retrieving network status: {e}"
            
            # help - Show available commands
            elif cmd == "help":
                return """
Available Memorychain commands:

- save: Save recent conversation as a memory
- search [query]: Search for memories containing the query
- list: Show recent memories in the chain
- view [id]: View details of a specific memory by ID
- status: Show the status of all nodes in the network
- help: Show this help message

You can reference memories in your messages using #mem:id syntax.
"""
            
            # Unknown command
            else:
                return f"Unknown command: {cmd}. Type 'help' to see available commands."
                
        except Exception as e:
            return f"Error executing command '{cmd}': {e}"
    
    def process_memory_references(self, message: str) -> str:
        """
        Process memory references in a message
        
        Args:
            message: Message that may contain memory references
            
        Returns:
            Message with memory references expanded
        """
        # Extract all memory IDs
        memory_ids = self.memorychain.extract_memory_references(message)
        
        if not memory_ids:
            return message
        
        # Replace each memory reference
        for memory_id in memory_ids:
            memory = self.memorychain.get_memory_by_id(memory_id)
            
            if not memory:
                continue
                
            # Format reference for replacement
            ref_pattern = fr'(?:#mem:{memory_id}|{{mem:{memory_id}}})'
            
            # Create replacement text
            headers = memory.get("headers", {})
            subject = headers.get("Subject", "No subject")
            
            # Replace the reference with the formatted memory
            memory_text = self.memorychain.format_memory(memory)
            replacement = f"Memory {memory_id} ({subject}):\n\n{memory_text}"
            
            # Replace in the message
            message = re.sub(ref_pattern, replacement, message)
            
        return message
    
    def chat(self, message: str) -> str:
        """
        Chat with the assistant
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        try:
            # Check if it's a command
            if message.startswith("/"):
                command = message[1:]
                return self.handle_command(command)
            
            # Set status to busy when processing a request
            self.memorychain.update_status(
                status=self.STATUS_BUSY,
                current_task_id=f"Processing message: {message[:30]}...",
                load=0.7
            )
            
            # Add user message to conversation
            self.conversation.append({
                "role": "user",
                "content": message
            })
            
            # Process the message with the assistant
            response = self.assistant.chat(message)
            
            # Process any memory references in the response
            response = self.process_memory_references(response)
            
            # Add assistant response to conversation
            self.conversation.append({
                "role": "assistant",
                "content": response
            })
            
            # Return to idle status
            self.memorychain.update_status(
                status=self.STATUS_IDLE,
                current_task_id=None,
                load=0.0
            )
            
            return response
            
        except Exception as e:
            # Ensure we set status back to idle on error
            self.memorychain.update_status(
                status=self.STATUS_IDLE,
                current_task_id=None,
                load=0.0
            )
            raise e
    
    def update_model(self, model: str):
        """
        Update the AI model information
        
        Args:
            model: New AI model being used
        """
        try:
            # Update both local model and report to network
            self.model = model
            self.assistant = Agent(api_key=self.api_key, model=self.model)
            
            # Update the network status
            self.memorychain.update_status(ai_model=model)
            return f"Updated AI model to: {model}"
        except Exception as e:
            return f"Error updating model: {e}"
    
    def start_interactive(self):
        """Run an interactive chat session"""
        print("FEI-Memorychain Interactive Chat")
        print("Type '/help' for a list of commands or '/exit' to quit")
        print("Type '/status' to see network status")
        print("Type '/model <name>' to change AI model")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ["/exit", "/quit"]:
                    print("Goodbye!")
                    break
                
                # Special command to update model
                if user_input.startswith("/model "):
                    model_name = user_input[7:].strip()
                    if model_name:
                        result = self.update_model(model_name)
                        print(f"\n{result}")
                        continue
                    
                # Process the user input
                response = self.chat(user_input)
                
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")

def main():
    """Main entry point"""
    # Load API key from environment or config
    api_key = load_api_key()
    
    # Get node address from environment or use default
    node_address = os.environ.get("MEMORYCHAIN_NODE", "localhost:6789")
    
    # Create the assistant
    assistant = MemorychainFEIAssistant(api_key=api_key, node_address=node_address)
    
    # Start interactive session
    assistant.start_interactive()

if __name__ == "__main__":
    main()