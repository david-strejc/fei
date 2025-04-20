#!/usr/bin/env python3
"""
FEI Status Reporting Example

This example demonstrates how to use the status reporting functionality of the Memorychain
system to report and monitor the status of FEI instances in a network.

The example shows how to:
1. Report the AI model being used by a FEI instance
2. Update the status of a FEI instance (busy, idle, working on task, etc.)
3. Monitor the status of other FEI instances in the network
4. Display a network-wide status overview

Prerequisites:
- A running Memorychain node (start with `python -m memdir_tools.memorychain_cli start`)
- FEI installed and configured

Usage:
python fei_status_reporting_example.py
"""

import os
import sys
import json
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import core FEI components
from fei.core.assistant import Agent
from fei.utils.config import load_api_key

# Import Memorychain connector
from fei.tools.memorychain_connector import MemorychainConnector, get_connector

class StatusReportingFEI:
    """
    FEI assistant with status reporting capabilities
    
    This class demonstrates how to integrate FEI with the Memorychain status
    reporting system, allowing it to:
    - Report its current AI model
    - Update its operational status
    - Monitor other FEI instances in the network
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
               node_address: Optional[str] = None,
               node_id: Optional[str] = None):
        """
        Initialize the FEI assistant with status reporting
        
        Args:
            api_key: API key for the LLM (Claude API)
            model: Model to use
            node_address: Memorychain node address (ip:port)
            node_id: Optional node ID (defaults to random)
        """
        # Initialize FEI assistant
        self.api_key = api_key or load_api_key()
        self.model = model
        self.assistant = Agent(api_key=self.api_key, model=self.model)
        
        # Initialize Memorychain connector
        self.memorychain = get_connector(node_address)
        
        # Set initial status
        try:
            # Update node status with AI model and idle status
            self.memorychain.update_status(
                status=self.STATUS_IDLE,
                ai_model=self.model,
                load=0.0
            )
            print(f"Connected to Memorychain node and updated status")
        except Exception as e:
            print(f"Warning: Could not update status: {e}")
    
    def set_busy(self, task_description: Optional[str] = None):
        """
        Set status to busy
        
        Args:
            task_description: Optional description of what the FEI is busy with
        """
        try:
            self.memorychain.update_status(
                status=self.STATUS_BUSY,
                load=0.7,
                current_task_id=task_description or "Internal task"
            )
        except Exception as e:
            print(f"Warning: Could not update status: {e}")
    
    def set_idle(self):
        """Set status to idle"""
        try:
            self.memorychain.update_status(
                status=self.STATUS_IDLE,
                load=0.0,
                current_task_id=None
            )
        except Exception as e:
            print(f"Warning: Could not update status: {e}")
    
    def set_working_on_task(self, task_id: str, load: float = 0.5):
        """
        Set status to working on a specific task
        
        Args:
            task_id: ID of the task being worked on
            load: Load factor (default 0.5)
        """
        try:
            self.memorychain.update_status(
                status=self.STATUS_WORKING_ON_TASK,
                current_task_id=task_id,
                load=load
            )
        except Exception as e:
            print(f"Warning: Could not update status: {e}")
    
    def update_model(self, model: str):
        """
        Update the AI model information
        
        Args:
            model: New AI model being used
        """
        try:
            # Update both local model and report to network
            self.model = model
            self.memorychain.update_status(ai_model=model)
            print(f"Updated AI model to: {model}")
        except Exception as e:
            print(f"Warning: Could not update model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of this node
        
        Returns:
            Status information dictionary
        """
        try:
            return self.memorychain.get_node_status()
        except Exception as e:
            print(f"Warning: Could not get status: {e}")
            return {}
    
    def get_network_status(self) -> Dict[str, Any]:
        """
        Get status of all nodes in the network
        
        Returns:
            Network status information
        """
        try:
            return self.memorychain.get_network_status()
        except Exception as e:
            print(f"Warning: Could not get network status: {e}")
            return {}
    
    def display_network_status(self):
        """Display formatted network status information"""
        try:
            network_status = self.get_network_status()
            nodes = network_status.get("nodes", {})
            
            if not nodes:
                print("No nodes found in the network.")
                return
                
            print("\n=== Memorychain Network Status ===")
            print(f"Total nodes: {len(nodes)}")
            print(f"Network load: {network_status.get('network_load', 0.0):.2f}")
            print("\nNode Details:")
            
            for node_id, status in nodes.items():
                # Format status with colors if available
                status_str = status.get("status", "unknown")
                model = status.get("ai_model", "unknown")
                load = status.get("load", 0.0)
                task = status.get("current_task_id", "-")
                
                # Colorize status if running in a terminal
                if sys.stdout.isatty():
                    if status_str == self.STATUS_IDLE:
                        status_str = f"\033[92m{status_str}\033[0m"  # Green
                    elif status_str == self.STATUS_BUSY:
                        status_str = f"\033[91m{status_str}\033[0m"  # Red
                    elif status_str == self.STATUS_WORKING_ON_TASK:
                        status_str = f"\033[93m{status_str}\033[0m"  # Yellow
                
                print(f"- Node: {node_id}")
                print(f"  Status: {status_str}")
                print(f"  Model: {model}")
                print(f"  Load: {load:.2f}")
                print(f"  Task: {task}")
                print()
                
        except Exception as e:
            print(f"Error displaying network status: {e}")
    
    def chat(self, message: str) -> str:
        """
        Chat with the assistant, automatically updating status
        
        Args:
            message: User message
            
        Returns:
            Assistant response
        """
        try:
            # Set status to busy when processing a request
            self.set_busy(f"Processing user query: {message[:30]}...")
            
            # Process with assistant
            response = self.assistant.chat(message)
            
            # Return to idle
            self.set_idle()
            
            return response
            
        except Exception as e:
            # Make sure we set idle even on error
            self.set_idle()
            raise e
    
    def start_interactive(self):
        """Run an interactive chat session with status reporting"""
        print("FEI with Status Reporting - Interactive Chat")
        print("Type '/status' to see network status")
        print("Type '/model <name>' to change AI model")
        print("Type '/exit' to quit")
        print("=" * 50)
        
        # Initial status display
        self.display_network_status()
        
        while True:
            try:
                user_input = input("\nYou: ")
                
                if user_input.lower() in ["/exit", "/quit"]:
                    print("Goodbye!")
                    break
                
                # Special commands
                if user_input.startswith("/"):
                    parts = user_input[1:].split(" ", 1)
                    cmd = parts[0].lower()
                    args = parts[1] if len(parts) > 1 else ""
                    
                    if cmd == "status":
                        self.display_network_status()
                        continue
                        
                    elif cmd == "model" and args:
                        self.update_model(args)
                        continue
                    
                # Process regular chat input
                response = self.chat(user_input)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                
def simulate_network_activity():
    """
    Simulate multiple FEI nodes with different status values
    
    This function creates multiple nodes and updates their status
    values to demonstrate network status monitoring.
    """
    # Create a connector
    connector = get_connector()
    
    # Simulate different nodes with different models and status
    models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "gpt-4", "gemini-pro"]
    statuses = ["idle", "busy", "working_on_task", "solution_proposed"]
    
    # Create 5 simulated nodes
    for i in range(5):
        # Each simulated node needs its own connector to properly register
        node_connector = get_connector()
        
        # Simulate different node by assigning random values
        model = random.choice(models)
        status = random.choice(statuses)
        load = random.uniform(0.1, 0.9)
        
        task_id = None
        if status == "working_on_task" or status == "solution_proposed":
            task_id = f"task-{random.randint(1000, 9999)}"
        
        try:
            # Use the regular update_status endpoint with random values
            node_connector.update_status(
                status=status,
                ai_model=model,
                load=load,
                current_task_id=task_id
            )
            print(f"Simulated node with model {model} and status {status}")
        except Exception as e:
            print(f"Could not simulate node: {e} - skipping")

def main():
    """Main entry point"""
    # Load API key from environment or config
    api_key = load_api_key()
    
    # Get node address from environment or use default
    node_address = os.environ.get("MEMORYCHAIN_NODE", "localhost:6789")
    
    # Create the assistant
    assistant = StatusReportingFEI(api_key=api_key, node_address=node_address)
    
    # Optionally simulate network activity
    if "--simulate" in sys.argv:
        simulate_network_activity()
    
    # Start interactive session
    assistant.start_interactive()

if __name__ == "__main__":
    import requests
    main()