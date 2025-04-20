#!/usr/bin/env python3
"""
Memorychain CLI - Command line interface for the Memorychain distributed memory system

This script provides a convenient command-line interface for:
- Starting Memorychain nodes
- Proposing new memories and tasks to the chain
- Managing tasks and solutions
- Viewing FeiCoin balances and transactions
- Managing network connections
- Validating chain integrity
"""

import os
import sys
import argparse
import uuid
import time
import json
import logging
from datetime import datetime
import requests
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memorychain-cli')

# Import memorychain functionality
from memdir_tools.memorychain import (
    MemoryChain, 
    MemorychainNode, 
    DEFAULT_PORT,
    TASK_PROPOSED,
    TASK_ACCEPTED,
    TASK_IN_PROGRESS,
    TASK_SOLUTION_PROPOSED,
    TASK_COMPLETED,
    TASK_REJECTED,
    DIFFICULTY_LEVELS
)

def create_node_id_file(node_id: Optional[str] = None) -> str:
    """
    Create or retrieve a persistent node ID
    
    Args:
        node_id: Optional node ID to save
        
    Returns:
        The node ID
    """
    node_id_file = os.path.join(os.path.expanduser("~"), ".memdir", "node_id.txt")
    
    # Create .memdir directory if it doesn't exist
    os.makedirs(os.path.dirname(node_id_file), exist_ok=True)
    
    # If node_id is provided, save it
    if node_id:
        with open(node_id_file, 'w') as f:
            f.write(node_id)
        return node_id
        
    # If file exists, read from it
    if os.path.exists(node_id_file):
        with open(node_id_file, 'r') as f:
            return f.read().strip()
    
    # Otherwise, generate a new ID and save it
    new_id = str(uuid.uuid4())
    with open(node_id_file, 'w') as f:
        f.write(new_id)
    
    return new_id

def start_node(args):
    """Start a Memorychain node"""
    # Get or create a persistent node ID
    node_id = create_node_id_file()
    
    # Override with command line argument if provided
    if args.node_id:
        node_id = args.node_id
        create_node_id_file(node_id)
    
    logger.info(f"Starting node with ID: {node_id}")
    
    # Create and start the node
    try:
        # Create node with the specified port and difficulty
        node = MemorychainNode(port=args.port, difficulty=args.difficulty)
        
        # Connect to seed node if specified
        if args.seed:
            success = node.connect_to_network(args.seed)
            if success:
                logger.info(f"Successfully connected to seed node: {args.seed}")
            else:
                logger.warning(f"Failed to connect to seed node: {args.seed}")
        
        # Start the server
        logger.info(f"Starting Memorychain node on port {args.port}")
        node.start()
        
    except Exception as e:
        logger.error(f"Error starting node: {e}")
        sys.exit(1)

def propose_memory(args):
    """Propose a new memory to the chain"""
    # Create a memory from command line arguments
    headers = {}
    if args.subject:
        headers["Subject"] = args.subject
    if args.tags:
        headers["Tags"] = args.tags
    if args.priority:
        headers["Priority"] = args.priority
    if args.status:
        headers["Status"] = args.status
    
    # Add metadata
    metadata = {
        "unique_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "date": datetime.now().isoformat(),
        "flags": list(args.flags) if args.flags else []
    }
    
    # Get content
    content = ""
    if args.file:
        try:
            with open(args.file, "r") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)
    elif args.content:
        content = args.content
    else:
        print("Enter memory content (Ctrl+D to finish):")
        try:
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            content = "\n".join(lines)
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
    
    # Build the memory data
    memory_data = {
        "headers": headers,
        "metadata": metadata,
        "content": content
    }
    
    # Submit to the local node
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/propose", 
                               json={"memory": memory_data},
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                print(f"Memory proposal accepted: {result.get('message', '')}")
            else:
                print(f"Memory proposal rejected: {result.get('message', '')}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def list_chain(args):
    """List all blocks in the chain"""
    try:
        response = requests.get(f"http://localhost:{args.port}/memorychain/chain", timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            chain = result.get("chain", [])
            
            if not chain:
                print("Chain is empty.")
                return
            
            print(f"Memory Chain - {len(chain)} blocks:")
            print("=" * 80)
            
            for block in chain:
                # Skip genesis block if requested
                if args.skip_genesis and block["index"] == 0:
                    continue
                    
                # Extract memory data
                memory = block["memory_data"]
                headers = memory.get("headers", {})
                metadata = memory.get("metadata", {})
                
                # Format the output
                print(f"Block #{block['index']}")
                print(f"  Hash: {block['hash'][:16]}...")
                print(f"  Date: {datetime.fromtimestamp(block['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"  Memory ID: {metadata.get('unique_id', 'unknown')}")
                print(f"  Subject: {headers.get('Subject', 'No subject')}")
                print(f"  Tags: {headers.get('Tags', '')}")
                print(f"  Status: {headers.get('Status', '')}")
                print(f"  Responsible Node: {block['responsible_node']}")
                print(f"  Proposer Node: {block['proposer_node']}")
                
                # Show task info if relevant
                if memory.get("type") == "task":
                    task_state = block.get("task_state", TASK_PROPOSED)
                    print(f"  Task State: {task_state}")
                    print(f"  Difficulty: {block.get('difficulty', 'medium')} ({block.get('reward', 3)} FeiCoins)")
                    
                    if block.get("solver_node"):
                        print(f"  Solved by: {block['solver_node']}")
                
                if args.content:
                    print("\n  Content Preview:")
                    content = memory.get("content", "")
                    # Show just first few lines
                    preview = "\n    ".join(content.split("\n")[:5])
                    if len(content.split("\n")) > 5:
                        preview += "\n    ..."
                    print(f"    {preview}")
                
                print("-" * 80)
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def view_memory(args):
    """View a specific memory from the chain"""
    try:
        # First get the full chain
        response = requests.get(f"http://localhost:{args.port}/memorychain/chain", timeout=10)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return
            
        chain = response.json().get("chain", [])
        
        # Find the memory by ID
        memory_found = False
        for block in chain:
            memory = block["memory_data"]
            memory_id = memory.get("metadata", {}).get("unique_id", "")
            
            if memory_id == args.id:
                memory_found = True
                
                # Check if this is a task
                if memory.get("type") == "task":
                    # If it's a task, use the view-task function
                    view_task_args = argparse.Namespace()
                    view_task_args.id = args.id
                    view_task_args.content = True
                    view_task_args.port = args.port
                    view_task(view_task_args)
                    return
                
                # Display the memory
                headers = memory.get("headers", {})
                metadata = memory.get("metadata", {})
                content = memory.get("content", "")
                
                print(f"Memory: {memory_id}")
                print("=" * 80)
                print(f"Subject: {headers.get('Subject', 'No subject')}")
                print(f"Date: {datetime.fromtimestamp(metadata.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Tags: {headers.get('Tags', '')}")
                print(f"Status: {headers.get('Status', '')}")
                print(f"Priority: {headers.get('Priority', '')}")
                print(f"Flags: {''.join(metadata.get('flags', []))}")
                print(f"Block: #{block['index']} (Hash: {block['hash'][:10]}...)")
                print(f"Responsible Node: {block['responsible_node']}")
                print(f"Proposer Node: {block['proposer_node']}")
                
                print("\nContent:")
                print("-" * 80)
                print(content)
                break
        
        if not memory_found:
            print(f"Memory with ID {args.id} not found in the chain.")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def list_responsible_memories(args):
    """List memories that this node is responsible for"""
    try:
        # Get memories this node is responsible for
        response = requests.get(f"http://localhost:{args.port}/memorychain/responsible_memories", timeout=10)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return
            
        result = response.json()
        memories = result.get("memories", [])
        
        if not memories:
            print("This node is not responsible for any memories.")
            return
            
        print(f"This node is responsible for {len(memories)} memories:")
        print("=" * 80)
        
        for memory in memories:
            headers = memory.get("headers", {})
            metadata = memory.get("metadata", {})
            memory_id = metadata.get("unique_id", "unknown")
            
            print(f"Memory: {memory_id}")
            print(f"  Subject: {headers.get('Subject', 'No subject')}")
            print(f"  Date: {datetime.fromtimestamp(metadata.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Tags: {headers.get('Tags', '')}")
            print(f"  Status: {headers.get('Status', '')}")
            
            if args.content:
                print("\n  Content Preview:")
                content = memory.get("content", "")
                # Show just first few lines
                preview = "\n    ".join(content.split("\n")[:3])
                if len(content.split("\n")) > 3:
                    preview += "\n    ..."
                print(f"    {preview}")
            
            print("-" * 80)
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def connect_node(args):
    """Connect to another node in the network"""
    try:
        # Register the new node
        response = requests.post(f"http://localhost:{args.port}/memorychain/register", 
                               json={"node_address": args.seed},
                               timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                print(f"Successfully connected to node: {args.seed}")
                print(f"Total connected nodes: {result.get('total_nodes', 0)}")
            else:
                print(f"Failed to connect to node: {args.seed}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def node_status(args):
    """Check the status of a node"""
    try:
        # Get node health status
        response = requests.get(f"http://localhost:{args.port}/memorychain/health", timeout=5)
        
        if response.status_code == 200:
            status = response.json()
            
            print("Memorychain Node Status:")
            print("=" * 80)
            print(f"Node ID: {status.get('node_id', 'unknown')}")
            print(f"Status: {status.get('status', 'unknown')}")
            print(f"FEI Status: {status.get('fei_status', 'unknown')}")
            print(f"AI Model: {status.get('ai_model', 'unknown')}")
            print(f"Current Task: {status.get('current_task', 'None')}")
            print(f"Load: {status.get('load', 0)}")
            print(f"Chain Length: {status.get('chain_length', 0)} blocks")
            print(f"Connected Nodes: {status.get('connected_nodes', 0)}")
            print(f"FeiCoin Balance: {status.get('feicoin_balance', 0)}")
            
            # Get node ID from file for reference
            stored_id = create_node_id_file()
            if stored_id != status.get('node_id', ''):
                print(f"Warning: Node ID in config file ({stored_id}) doesn't match running node")
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def network_status(args):
    """Check the status of all nodes in the network"""
    try:
        # Get network status
        response = requests.get(f"http://localhost:{args.port}/memorychain/network_status", timeout=10)
        
        if response.status_code == 200:
            network = response.json()
            nodes = network.get("nodes", [])
            
            print("Memorychain Network Status:")
            print("=" * 80)
            print(f"Total Nodes: {network.get('total_nodes', 0)}")
            print(f"Online Nodes: {network.get('online_nodes', 0)}")
            print(f"Network Load: {network.get('network_load', 0):.2f}")
            print()
            
            # Format node status with colors
            for node in nodes:
                # Determine if this is the local node
                is_self = node.get("is_self", False)
                node_prefix = "LOCAL NODE: " if is_self else "REMOTE NODE: "
                
                # Color based on status
                status = node.get("status", "unknown")
                status_colors = {
                    "idle": "green",
                    "working_on_task": "blue",
                    "solution_proposed": "cyan",
                    "task_completed": "green",
                    "busy": "yellow"
                }
                status_color = status_colors.get(status, "reset")
                
                print(f"{node_prefix}{node.get('node_id', 'unknown')} @ {node.get('address', 'unknown')}")
                print(f"  Status: {colorize(status, status_color)}")
                print(f"  AI Model: {node.get('ai_model', 'unknown')}")
                print(f"  Current Task: {node.get('current_task', 'None')}")
                print(f"  Load: {node.get('load', 0):.2f}")
                print(f"  FeiCoin Balance: {node.get('feicoin_balance', 0)}")
                print(f"  Last Update: {datetime.fromtimestamp(node.get('last_update', 0)).strftime('%Y-%m-%d %H:%M:%S')}")
                print("-" * 80)
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)

def validate_chain(args):
    """Validate the integrity of the memory chain"""
    try:
        # Get the chain for validation
        chain_response = requests.get(f"http://localhost:{args.port}/memorychain/chain", timeout=10)
        
        if chain_response.status_code != 200:
            print(f"Error: {chain_response.status_code} - {chain_response.text}")
            return
            
        chain_data = chain_response.json().get("chain", [])
        
        # Create a temporary chain for validation
        temp_chain = MemoryChain("validator")
        
        # Replace the chain with the data we got
        from memdir_tools.memorychain import MemoryBlock
        temp_chain.chain = [MemoryBlock.from_dict(block) for block in chain_data]
        
        # Validate the chain
        is_valid = temp_chain.validate_chain()
        
        if is_valid:
            print(f"✅ Chain is valid! ({len(chain_data)} blocks)")
        else:
            print(f"❌ Chain validation FAILED! The chain may be corrupted or tampered with.")
            
    except requests.RequestException as e:
        print(f"Error connecting to local node: {e}")
        print("Is the Memorychain node running?")
        sys.exit(1)
    except Exception as e:
        print(f"Error validating chain: {e}")

def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text"""
    colors = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "red": "\033[31m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

def propose_task(args):
    """Propose a new task to the chain"""
    # Create task data
    task_data = {}
    
    # Add headers
    headers = {}
    if args.subject:
        headers["Subject"] = args.subject
    else:
        headers["Subject"] = "Task: " + args.description[:40] + ("..." if len(args.description) > 40 else "")
        
    if args.tags:
        headers["Tags"] = args.tags
    else:
        headers["Tags"] = "task"
        
    if args.priority:
        headers["Priority"] = args.priority
    
    if args.status:
        headers["Status"] = args.status
    
    # Add metadata
    metadata = {
        "unique_id": str(uuid.uuid4()),
        "timestamp": time.time(),
        "date": datetime.now().isoformat(),
        "flags": list(args.flags) if args.flags else []
    }
    
    # Get description
    description = args.description
    
    # Get detailed content
    if args.file:
        with open(args.file, "r") as f:
            content = f.read()
    elif args.content:
        content = args.content
    else:
        content = description
    
    # Build task data
    task_data = {
        "headers": headers,
        "metadata": metadata,
        "content": content,
        "description": description  # Add a shorter description for UI
    }
    
    # Set difficulty
    difficulty = args.difficulty or "medium"
    
    # Connect to local node and propose
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/propose_task", json={
            "task": task_data,
            "difficulty": difficulty
        })
        
        result = response.json()
        if result.get("success", False):
            print(f"Task proposal accepted: {result.get('message', '')}")
        else:
            print(f"Task proposal rejected: {result.get('message', '')}")
            
    except requests.RequestException as e:
        print(f"Error proposing task: {e}")

def list_tasks(args):
    """List all tasks in the chain"""
    try:
        query_params = {}
        if args.state:
            query_params["state"] = args.state
            
        response = requests.get(f"http://localhost:{args.port}/memorychain/tasks", params=query_params)
        
        if response.status_code == 200:
            data = response.json()
            tasks = data["tasks"]
            
            if not tasks:
                print("No tasks found.")
                return
            
            state_filter = args.state or "all"
            print(f"Tasks ({len(tasks)}) - State filter: {state_filter}")
            print("=" * 80)
            
            for task in tasks:
                # Format state with color
                state_colors = {
                    TASK_PROPOSED: "yellow",
                    TASK_ACCEPTED: "yellow",
                    TASK_IN_PROGRESS: "blue",
                    TASK_SOLUTION_PROPOSED: "cyan",
                    TASK_COMPLETED: "green",
                    TASK_REJECTED: "red"
                }
                state_color = state_colors.get(task["state"], "reset")
                state_display = colorize(task["state"], state_color)
                
                # Format difficulty with color
                difficulty_colors = {
                    "easy": "green",
                    "medium": "blue",
                    "hard": "magenta",
                    "very_hard": "red",
                    "extreme": "red"
                }
                difficulty_color = difficulty_colors.get(task["difficulty"], "reset")
                difficulty_display = colorize(f"{task['difficulty']} ({task['reward']} FeiCoins)", difficulty_color)
                
                print(f"ID: {task['id']}")
                print(f"Subject: {task['subject']}")
                print(f"State: {state_display}")
                print(f"Difficulty: {difficulty_display}")
                print(f"Working Nodes: {len(task['working_nodes'])}")
                print(f"Solutions: {task['solution_count']}")
                if task["solver"]:
                    print(f"Solved by: {task['solver']}")
                print("-" * 80)
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error listing tasks: {e}")

def view_task(args):
    """View a specific task"""
    try:
        response = requests.get(f"http://localhost:{args.port}/memorychain/tasks/{args.id}")
        
        if response.status_code == 200:
            task = response.json()
            
            # Format state with color
            state_colors = {
                TASK_PROPOSED: "yellow",
                TASK_ACCEPTED: "yellow",
                TASK_IN_PROGRESS: "blue",
                TASK_SOLUTION_PROPOSED: "cyan",
                TASK_COMPLETED: "green",
                TASK_REJECTED: "red"
            }
            state_color = state_colors.get(task["state"], "reset")
            state_display = colorize(task["state"], state_color)
            
            print(f"Task: {task['id']}")
            print("=" * 80)
            print(f"Subject: {task['subject']}")
            print(f"State: {state_display}")
            print(f"Difficulty: {task['difficulty']} ({task['reward']} FeiCoins)")
            print(f"Proposer: {task['proposer']}")
            print(f"Date: {datetime.fromtimestamp(task['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            if task["working_nodes"]:
                print(f"Working Nodes: {', '.join(task['working_nodes'])}")
                
            if task["solver"]:
                print(f"Solved by: {task['solver']}")
            
            # Show difficulty votes
            if task["difficulty_votes"]:
                print("\nDifficulty Votes:")
                for node, vote in task["difficulty_votes"].items():
                    print(f"  {node}: {vote}")
            
            # Show solutions
            if task["solutions"]:
                print("\nProposed Solutions:")
                for idx, solution in enumerate(task["solutions"]):
                    print(f"  Solution #{idx} by {solution['node_id']}:")
                    print(f"    Timestamp: {datetime.fromtimestamp(solution['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    # Show votes on this solution
                    if solution["votes"]:
                        yes_votes = sum(1 for v in solution["votes"].values() if v)
                        no_votes = sum(1 for v in solution["votes"].values() if not v)
                        print(f"    Votes: {yes_votes} yes, {no_votes} no")
            
            # Show task content
            if args.content:
                print("\nTask Description:")
                print("-" * 80)
                print(task["content"])
                
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        print(f"Error viewing task: {e}")

def claim_task(args):
    """Claim a task to work on"""
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/claim_task", json={
            "task_id": args.id
        })
        
        result = response.json()
        if result.get("success", False):
            print(result.get("message", ""))
        else:
            print(f"Failed to claim task: {result.get('message', '')}")
            
    except requests.RequestException as e:
        print(f"Error claiming task: {e}")

def submit_solution(args):
    """Submit a solution for a task"""
    # Get solution content
    if args.file:
        with open(args.file, "r") as f:
            solution_content = f.read()
    elif args.content:
        solution_content = args.content
    else:
        print("Enter solution content (Ctrl+D to finish):")
        content_lines = []
        try:
            while True:
                line = input()
                content_lines.append(line)
        except EOFError:
            solution_content = "\n".join(content_lines)
    
    # Create solution data
    solution_data = {
        "content": solution_content,
        "timestamp": time.time(),
        "summary": args.summary if args.summary else "Solution submission"
    }
    
    # Submit solution
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/submit_solution", json={
            "task_id": args.id,
            "solution": solution_data
        })
        
        result = response.json()
        if result.get("success", False):
            print(result.get("message", ""))
        else:
            print(f"Failed to submit solution: {result.get('message', '')}")
            
    except requests.RequestException as e:
        print(f"Error submitting solution: {e}")

def vote_solution(args):
    """Vote on a solution"""
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/vote_solution", json={
            "task_id": args.task_id,
            "solution_index": args.solution_index,
            "approve": args.approve
        })
        
        result = response.json()
        if result.get("success", False):
            print(result.get("message", ""))
        else:
            print(f"Failed to vote on solution: {result.get('message', '')}")
            
    except requests.RequestException as e:
        print(f"Error voting on solution: {e}")

def vote_difficulty(args):
    """Vote on task difficulty"""
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/vote_difficulty", json={
            "task_id": args.id,
            "difficulty": args.difficulty
        })
        
        result = response.json()
        if result.get("success", False):
            print(result.get("message", ""))
        else:
            print(f"Failed to vote on difficulty: {result.get('message', '')}")
            
    except requests.RequestException as e:
        print(f"Error voting on difficulty: {e}")

def show_wallet(args):
    """Show wallet balance and transactions"""
    try:
        # Get balance
        balance_response = requests.get(f"http://localhost:{args.port}/memorychain/wallet/balance", 
                                      params={"node_id": args.node_id})
        
        if balance_response.status_code != 200:
            print(f"Error: {balance_response.status_code} - {balance_response.text}")
            return
            
        balance_data = balance_response.json()
        
        # Get transactions
        params = {"limit": args.limit}
        if args.node_id:
            params["node_id"] = args.node_id
            
        tx_response = requests.get(f"http://localhost:{args.port}/memorychain/wallet/transactions", 
                                 params=params)
        
        if tx_response.status_code != 200:
            print(f"Error: {tx_response.status_code} - {tx_response.text}")
            return
            
        tx_data = tx_response.json()
        
        # Display wallet info
        print(f"Wallet for {balance_data['node_id']}")
        print("=" * 80)
        print(f"Balance: {balance_data['balance']} FeiCoins")
        
        # Display transactions
        if tx_data["transactions"]:
            print("\nRecent Transactions:")
            for tx in tx_data["transactions"]:
                timestamp = datetime.fromtimestamp(tx["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                
                if tx["type"] == "credit":
                    print(f"  {timestamp} - Received {tx['amount']} FeiCoins - {tx['reason']}")
                elif tx["type"] == "transfer":
                    if tx["from_node"] == balance_data["node_id"]:
                        print(f"  {timestamp} - Sent {tx['amount']} FeiCoins to {tx['to_node']} - {tx['reason']}")
                    else:
                        print(f"  {timestamp} - Received {tx['amount']} FeiCoins from {tx['from_node']} - {tx['reason']}")
        else:
            print("\nNo transactions found.")
            
    except requests.RequestException as e:
        print(f"Error accessing wallet: {e}")

def main():
    """Main entry point for the memorychain CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memorychain - Distributed memory ledger with FeiCoin rewards")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Node start command
    start_parser = subparsers.add_parser("start", help="Start a memory chain node")
    start_parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    start_parser.add_argument("-d", "--difficulty", type=int, default=2, help="Mining difficulty")
    start_parser.add_argument("-s", "--seed", help="Seed node to connect to")
    start_parser.add_argument("--node-id", help="Override node ID (default: generate or use existing)")
    start_parser.set_defaults(func=start_node)
    
    # Propose memory command
    propose_parser = subparsers.add_parser("propose", help="Propose a new memory to the chain")
    propose_parser.add_argument("-s", "--subject", help="Memory subject")
    propose_parser.add_argument("-t", "--tags", help="Memory tags (comma-separated)")
    propose_parser.add_argument("-p", "--priority", choices=["high", "medium", "low"], help="Memory priority")
    propose_parser.add_argument("--status", help="Memory status")
    propose_parser.add_argument("--flags", default="", help="Memory flags (e.g., 'FP' for Flagged+Priority)")
    propose_parser.add_argument("--file", help="Read content from file")
    propose_parser.add_argument("--content", help="Memory content")
    propose_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    propose_parser.set_defaults(func=propose_memory)
    
    # Propose task command
    task_parser = subparsers.add_parser("task", help="Propose a new task")
    task_parser.add_argument("description", help="Task description")
    task_parser.add_argument("-s", "--subject", help="Task subject (default: generated from description)")
    task_parser.add_argument("-t", "--tags", help="Task tags (comma-separated)")
    task_parser.add_argument("-d", "--difficulty", choices=list(DIFFICULTY_LEVELS.keys()), 
                           help="Initial difficulty estimate (default: medium)")
    task_parser.add_argument("-p", "--priority", choices=["high", "medium", "low"], help="Task priority")
    task_parser.add_argument("--status", help="Task status")
    task_parser.add_argument("--flags", default="", help="Task flags")
    task_parser.add_argument("--file", help="Read detailed task description from file")
    task_parser.add_argument("--content", help="Detailed task description")
    task_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    task_parser.set_defaults(func=propose_task)
    
    # List tasks command
    tasks_parser = subparsers.add_parser("tasks", help="List all tasks")
    tasks_parser.add_argument("--state", choices=[TASK_PROPOSED, TASK_ACCEPTED, TASK_IN_PROGRESS, 
                                              TASK_SOLUTION_PROPOSED, TASK_COMPLETED, TASK_REJECTED],
                            help="Filter by task state")
    tasks_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    tasks_parser.set_defaults(func=list_tasks)
    
    # View task command
    view_task_parser = subparsers.add_parser("view-task", help="View a specific task")
    view_task_parser.add_argument("id", help="Task ID")
    view_task_parser.add_argument("--content", action="store_true", help="Show task content")
    view_task_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    view_task_parser.set_defaults(func=view_task)
    
    # Claim task command
    claim_parser = subparsers.add_parser("claim", help="Claim a task to work on")
    claim_parser.add_argument("id", help="Task ID")
    claim_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    claim_parser.set_defaults(func=claim_task)
    
    # Submit solution command
    solve_parser = subparsers.add_parser("solve", help="Submit a solution for a task")
    solve_parser.add_argument("id", help="Task ID")
    solve_parser.add_argument("--summary", help="Short solution summary")
    solve_parser.add_argument("--file", help="Read solution from file")
    solve_parser.add_argument("--content", help="Solution content")
    solve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    solve_parser.set_defaults(func=submit_solution)
    
    # Vote on solution command
    vote_parser = subparsers.add_parser("vote", help="Vote on a task solution")
    vote_parser.add_argument("task_id", help="Task ID")
    vote_parser.add_argument("solution_index", type=int, help="Solution index (0-based)")
    vote_parser.add_argument("--approve", action="store_true", help="Approve the solution (default: reject)")
    vote_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    vote_parser.set_defaults(func=vote_solution)
    
    # Vote on difficulty command
    difficulty_parser = subparsers.add_parser("difficulty", help="Vote on task difficulty")
    difficulty_parser.add_argument("id", help="Task ID")
    difficulty_parser.add_argument("difficulty", choices=list(DIFFICULTY_LEVELS.keys()), help="Difficulty level")
    difficulty_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    difficulty_parser.set_defaults(func=vote_difficulty)
    
    # Wallet command
    wallet_parser = subparsers.add_parser("wallet", help="Show wallet balance and transactions")
    wallet_parser.add_argument("--node-id", help="Node ID (default: local node)")
    wallet_parser.add_argument("--limit", type=int, default=10, help="Maximum number of transactions to show")
    wallet_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    wallet_parser.set_defaults(func=show_wallet)
    
    # List chain command
    list_parser = subparsers.add_parser("list", help="List the memory chain")
    list_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    list_parser.add_argument("--content", action="store_true", help="Show content preview")
    list_parser.add_argument("--skip-genesis", action="store_true", help="Skip the genesis block")
    list_parser.set_defaults(func=list_chain)
    
    # List responsible memories command
    responsible_parser = subparsers.add_parser("responsible", help="List memories a node is responsible for")
    responsible_parser.add_argument("--node-id", help="Node ID (default: local node)")
    responsible_parser.add_argument("--with-content", action="store_true", help="Include memory content")
    responsible_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    responsible_parser.set_defaults(func=list_responsible_memories)
    
    # Connect command
    connect_parser = subparsers.add_parser("connect", help="Connect to a seed node")
    connect_parser.add_argument("seed", help="Seed node address (ip:port)")
    connect_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local node port")
    connect_parser.set_defaults(func=connect_node)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check node status")
    status_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    status_parser.set_defaults(func=node_status)
    
    # Validate chain command
    validate_parser = subparsers.add_parser("validate", help="Validate chain integrity")
    validate_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    validate_parser.set_defaults(func=validate_chain)
    
    # View memory command
    view_parser = subparsers.add_parser("view", help="View a specific memory")
    view_parser.add_argument("id", help="Memory ID")
    view_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    view_parser.set_defaults(func=view_memory)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command function if provided, otherwise show help
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()