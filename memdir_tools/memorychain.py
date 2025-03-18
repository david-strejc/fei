#!/usr/bin/env python3
"""
Memorychain - Distributed memory ledger system for Memdir

This module implements a distributed memory management system inspired by blockchain principles:
- Multiple nodes can propose new memories to a shared ledger
- Consensus mechanism ensures agreement on memory additions
- Each memory has a designated responsible node
- Tamper-proof chain of memory blocks with cryptographic verification
- Distributed operation with minimal centralization
- Task allocation and FeiCoin rewards for task completion
- Difficulty rating for tasks through consensus

Basic workflow:
1. Node creates a memory/task and proposes it to the network
2. Quorum of nodes validate and vote on the memory/task
3. Once accepted, memory/task is added to the chain
4. For tasks, nodes can claim to work on them
5. When a task is completed, nodes vote on the solution
6. Approved solutions are rewarded with FeiCoins based on difficulty
7. A responsible node is designated for managing memory/task
8. All nodes update their copy of the chain
"""

import os
import json
import time
import hashlib
import threading
import uuid
import logging
import socket
import random
import base64
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('memorychain')

# Import memdir utilities
from memdir_tools.utils import save_memory, list_memories, get_memdir_folders

# Constants
CHAIN_FILE = os.path.join(os.path.expanduser("~"), ".memdir", "memorychain.json")
TEMP_PROPOSAL_DIR = os.path.join(os.path.expanduser("~"), ".memdir", "proposals")
WALLET_FILE = os.path.join(os.path.expanduser("~"), ".memdir", "feicoin_wallet.json")
TASKS_DIR = os.path.join(os.path.expanduser("~"), ".memdir", "tasks")
DEFAULT_PORT = 6789
MIN_QUORUM_PERCENT = 51  # Minimum percentage of nodes required for consensus
INITIAL_FEICOINS = 100   # Initial FeiCoins for new nodes

# Task states
TASK_PROPOSED = "proposed"    # Newly proposed task
TASK_ACCEPTED = "accepted"    # Task accepted by consensus
TASK_IN_PROGRESS = "in_progress"  # Task is being worked on
TASK_SOLUTION_PROPOSED = "solution_proposed"  # Solution proposed but not approved
TASK_COMPLETED = "completed"  # Task completed and solution approved
TASK_REJECTED = "rejected"    # Task or solution rejected

# Default difficulty levels and rewards
DIFFICULTY_LEVELS = {
    "easy": 1,
    "medium": 3,
    "hard": 5,
    "very_hard": 10,
    "extreme": 20
}

class MemoryBlock:
    """Represents a single block in the memory chain"""
    
    def __init__(self, index: int, timestamp: float, memory_data: Dict[str, Any], 
                 previous_hash: str, responsible_node: str, proposer_node: str):
        """
        Initialize a memory block
        
        Args:
            index: Position in the chain
            timestamp: Creation time
            memory_data: The memory content and metadata
            previous_hash: Hash of the previous block
            responsible_node: Node ID responsible for this memory
            proposer_node: Node ID that proposed this memory
        """
        self.index = index
        self.timestamp = timestamp
        self.memory_data = memory_data
        self.previous_hash = previous_hash
        self.responsible_node = responsible_node
        self.proposer_node = proposer_node
        self.nonce = 0
        
        # Task-specific fields
        # These fields only apply if memory_data["type"] == "task"
        self.working_nodes = []  # Nodes working on this task
        self.solutions = []      # Proposed solutions
        self.difficulty = memory_data.get("task_difficulty", "medium")
        self.reward = DIFFICULTY_LEVELS.get(self.difficulty, 3)  # Default to medium if unknown
        self.task_state = memory_data.get("task_state", TASK_PROPOSED)
        self.solver_node = None  # Node that solved the task
        self.difficulty_votes = {}  # Node votes on difficulty
        
        self.hash = self.calculate_hash()
        
    def calculate_hash(self) -> str:
        """
        Calculate the cryptographic hash of this block
        
        Returns:
            SHA-256 hash of the block data
        """
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "memory_id": self.memory_data.get("metadata", {}).get("unique_id", ""),
            "previous_hash": self.previous_hash,
            "responsible_node": self.responsible_node,
            "proposer_node": self.proposer_node,
            "task_state": getattr(self, "task_state", None),
            "difficulty": getattr(self, "difficulty", None),
            "solver_node": getattr(self, "solver_node", None),
            "nonce": self.nonce
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 2) -> None:
        """
        Mine the block by finding a hash with leading zeros
        
        Args:
            difficulty: Number of leading zeros required
        """
        target = "0" * difficulty
        
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
    
    def is_task(self) -> bool:
        """
        Check if this block represents a task
        
        Returns:
            True if this is a task, False otherwise
        """
        return self.memory_data.get("type") == "task"
    
    def update_task_state(self, new_state: str) -> None:
        """
        Update the task state
        
        Args:
            new_state: New task state
        """
        if self.is_task():
            # Update both task_state attribute and in memory_data
            self.task_state = new_state
            self.memory_data["task_state"] = new_state
            # Recalculate hash after state change
            self.hash = self.calculate_hash()
    
    def add_working_node(self, node_id: str) -> bool:
        """
        Add a node to the list of nodes working on this task
        
        Args:
            node_id: ID of the node working on the task
            
        Returns:
            True if added, False if already working
        """
        if not self.is_task():
            return False
            
        if node_id not in self.working_nodes:
            self.working_nodes.append(node_id)
            # Update memory data to reflect change
            self.memory_data["working_nodes"] = self.working_nodes
            return True
        return False
    
    def add_solution(self, node_id: str, solution_data: Dict[str, Any]) -> bool:
        """
        Add a proposed solution for the task
        
        Args:
            node_id: ID of the node proposing the solution
            solution_data: Solution details
            
        Returns:
            True if added, False if invalid
        """
        if not self.is_task() or self.task_state in [TASK_COMPLETED, TASK_REJECTED]:
            return False
            
        # Create solution record
        solution = {
            "node_id": node_id,
            "timestamp": time.time(),
            "data": solution_data,
            "votes": {}  # For tracking votes on this solution
        }
        
        self.solutions.append(solution)
        # Update memory data to reflect change
        self.memory_data["solutions"] = self.solutions
        self.update_task_state(TASK_SOLUTION_PROPOSED)
        return True
    
    def vote_on_difficulty(self, node_id: str, difficulty: str) -> None:
        """
        Vote on the difficulty of this task
        
        Args:
            node_id: ID of the voting node
            difficulty: Proposed difficulty level
        """
        if not self.is_task():
            return
            
        # Record vote
        if difficulty in DIFFICULTY_LEVELS:
            self.difficulty_votes[node_id] = difficulty
            self.memory_data["difficulty_votes"] = self.difficulty_votes
            
            # Recalculate difficulty based on votes
            self._recalculate_difficulty()
    
    def _recalculate_difficulty(self) -> None:
        """Recalculate task difficulty based on votes"""
        if not self.difficulty_votes:
            return
            
        # Count votes for each difficulty level
        vote_counts = {}
        for vote in self.difficulty_votes.values():
            vote_counts[vote] = vote_counts.get(vote, 0) + 1
            
        # Find the most voted difficulty
        max_votes = 0
        top_difficulty = None
        
        for difficulty, count in vote_counts.items():
            if count > max_votes:
                max_votes = count
                top_difficulty = difficulty
                
        # Update difficulty and reward if changed
        if top_difficulty and top_difficulty != self.difficulty:
            self.difficulty = top_difficulty
            self.reward = DIFFICULTY_LEVELS.get(top_difficulty, 3)
            
            # Update memory data
            self.memory_data["task_difficulty"] = self.difficulty
            self.memory_data["task_reward"] = self.reward
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert block to dictionary for serialization
        
        Returns:
            Dictionary representation of the block
        """
        data = {
            "index": self.index,
            "timestamp": self.timestamp,
            "memory_data": self.memory_data,
            "previous_hash": self.previous_hash,
            "responsible_node": self.responsible_node,
            "proposer_node": self.proposer_node,
            "nonce": self.nonce,
            "hash": self.hash
        }
        
        # Add task-specific fields if this is a task
        if self.is_task():
            data.update({
                "working_nodes": self.working_nodes,
                "solutions": self.solutions,
                "difficulty": self.difficulty,
                "reward": self.reward,
                "task_state": self.task_state,
                "solver_node": self.solver_node,
                "difficulty_votes": self.difficulty_votes
            })
            
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryBlock':
        """
        Create a block from dictionary data
        
        Args:
            data: Dictionary representation of a block
            
        Returns:
            MemoryBlock instance
        """
        block = cls(
            data["index"],
            data["timestamp"],
            data["memory_data"],
            data["previous_hash"],
            data["responsible_node"],
            data["proposer_node"]
        )
        block.nonce = data["nonce"]
        block.hash = data["hash"]
        
        # Load task-specific fields if this is a task
        if block.is_task():
            block.working_nodes = data.get("working_nodes", [])
            block.solutions = data.get("solutions", [])
            block.difficulty = data.get("difficulty", "medium")
            block.reward = data.get("reward", DIFFICULTY_LEVELS.get(block.difficulty, 3))
            block.task_state = data.get("task_state", TASK_PROPOSED)
            block.solver_node = data.get("solver_node")
            block.difficulty_votes = data.get("difficulty_votes", {})
            
        return block


class FeiCoinWallet:
    """
    Manages FeiCoin balances and transactions for nodes
    """
    
    def __init__(self):
        """Initialize the wallet system"""
        self.balances = {}  # node_id -> balance
        self.transactions = []  # List of transaction records
        self.lock = threading.RLock()  # For thread safety
        
        # Create wallet directory if it doesn't exist
        os.makedirs(os.path.dirname(WALLET_FILE), exist_ok=True)
        
        # Load wallet data if it exists
        if os.path.exists(WALLET_FILE):
            self.load_wallet()
        
    def get_balance(self, node_id: str) -> float:
        """
        Get the balance for a node
        
        Args:
            node_id: ID of the node
            
        Returns:
            Current balance
        """
        with self.lock:
            return self.balances.get(node_id, 0)
    
    def add_funds(self, node_id: str, amount: float, reason: str) -> bool:
        """
        Add funds to a node's balance
        
        Args:
            node_id: ID of the receiving node
            amount: Amount to add
            reason: Reason for the transaction
            
        Returns:
            True if successful
        """
        if amount <= 0:
            return False
            
        with self.lock:
            # Initialize balance if new node
            if node_id not in self.balances:
                self.balances[node_id] = INITIAL_FEICOINS
                
            # Add funds
            self.balances[node_id] += amount
            
            # Record transaction
            transaction = {
                "type": "credit",
                "node_id": node_id,
                "amount": amount,
                "reason": reason,
                "timestamp": time.time()
            }
            self.transactions.append(transaction)
            
            # Save wallet
            self.save_wallet()
            return True
    
    def transfer(self, from_node: str, to_node: str, amount: float, reason: str) -> bool:
        """
        Transfer funds between nodes
        
        Args:
            from_node: ID of the sending node
            to_node: ID of the receiving node
            amount: Amount to transfer
            reason: Reason for the transaction
            
        Returns:
            True if successful, False if insufficient funds
        """
        if amount <= 0:
            return False
            
        with self.lock:
            # Check if sender has enough funds
            sender_balance = self.balances.get(from_node, 0)
            if sender_balance < amount:
                return False
                
            # Initialize receiver balance if new node
            if to_node not in self.balances:
                self.balances[to_node] = INITIAL_FEICOINS
                
            # Perform transfer
            self.balances[from_node] -= amount
            self.balances[to_node] += amount
            
            # Record transaction
            transaction = {
                "type": "transfer",
                "from_node": from_node,
                "to_node": to_node,
                "amount": amount,
                "reason": reason,
                "timestamp": time.time()
            }
            self.transactions.append(transaction)
            
            # Save wallet
            self.save_wallet()
            return True
    
    def get_transactions(self, node_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get recent transactions
        
        Args:
            node_id: Optional node ID to filter by
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction records
        """
        with self.lock:
            if node_id:
                # Filter transactions for this node
                node_transactions = [t for t in self.transactions 
                                   if (t.get("node_id") == node_id or 
                                       t.get("from_node") == node_id or 
                                       t.get("to_node") == node_id)]
                return sorted(node_transactions, key=lambda t: t["timestamp"], reverse=True)[:limit]
            else:
                # Return all transactions
                return sorted(self.transactions, key=lambda t: t["timestamp"], reverse=True)[:limit]
    
    def save_wallet(self) -> None:
        """Save wallet data to disk"""
        with self.lock:
            wallet_data = {
                "balances": self.balances,
                "transactions": self.transactions
            }
            
            with open(WALLET_FILE, 'w') as f:
                json.dump(wallet_data, f, indent=2)
    
    def load_wallet(self) -> bool:
        """
        Load wallet data from disk
        
        Returns:
            True if successful
        """
        try:
            with open(WALLET_FILE, 'r') as f:
                wallet_data = json.load(f)
                
            with self.lock:
                self.balances = wallet_data.get("balances", {})
                self.transactions = wallet_data.get("transactions", [])
                
            return True
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.error(f"Error loading wallet: {e}")
            return False


class MemoryChain:
    """Represents the entire chain of memory blocks"""
    
    def __init__(self, node_id: str, difficulty: int = 2):
        """
        Initialize the memory chain
        
        Args:
            node_id: Unique identifier for this node
            difficulty: Mining difficulty (number of leading zeros in hash)
        """
        self.chain: List[MemoryBlock] = []
        self.node_id = node_id
        self.difficulty = difficulty
        self.pending_memories: List[Dict[str, Any]] = []
        self.pending_tasks: List[Dict[str, Any]] = []
        self.nodes: Set[str] = set()  # Format: "ip:port"
        self.lock = threading.RLock()  # For thread safety
        
        # Initialize FeiCoin wallet
        self.wallet = FeiCoinWallet()
        
        # Create tasks directory
        os.makedirs(TASKS_DIR, exist_ok=True)
        
        # Create genesis block if chain is empty
        if not self.load_chain():
            self.create_genesis_block()
            self.save_chain()
    
    def create_genesis_block(self) -> None:
        """Create the initial block in the chain"""
        genesis_memory = {
            "metadata": {
                "unique_id": "genesis",
                "timestamp": time.time(),
                "date": datetime.now(),
                "flags": []
            },
            "headers": {
                "Subject": "Genesis Block",
                "Tags": "system,genesis,memorychain",
                "Status": "system"
            },
            "content": "Initial block of the Memory Chain. Created on " + 
                      datetime.now().isoformat()
        }
        
        genesis_block = MemoryBlock(0, time.time(), genesis_memory, "0", self.node_id, self.node_id)
        genesis_block.mine_block(self.difficulty)
        
        with self.lock:
            self.chain.append(genesis_block)
    
    def get_latest_block(self) -> MemoryBlock:
        """
        Get the most recent block in the chain
        
        Returns:
            The last block in the chain
        """
        with self.lock:
            return self.chain[-1]
    
    def add_memory(self, memory_data: Dict[str, Any], responsible_node: Optional[str] = None) -> str:
        """
        Add a new memory to the chain
        
        Args:
            memory_data: Memory content and metadata
            responsible_node: Node responsible for this memory (None = self)
            
        Returns:
            Hash of the newly created block
        """
        if responsible_node is None:
            responsible_node = self.node_id
            
        previous_block = self.get_latest_block()
        new_index = previous_block.index + 1
        
        new_block = MemoryBlock(
            new_index,
            time.time(),
            memory_data,
            previous_block.hash,
            responsible_node,
            self.node_id
        )
        
        new_block.mine_block(self.difficulty)
        
        with self.lock:
            self.chain.append(new_block)
            self.save_chain()
            
        return new_block.hash
    
    def validate_chain(self) -> bool:
        """
        Verify the integrity of the entire chain
        
        Returns:
            True if valid, False otherwise
        """
        with self.lock:
            for i in range(1, len(self.chain)):
                current_block = self.chain[i]
                previous_block = self.chain[i-1]
                
                # Check hash integrity
                if current_block.hash != current_block.calculate_hash():
                    logger.error(f"Block {i} has invalid hash")
                    return False
                
                # Check chain continuity
                if current_block.previous_hash != previous_block.hash:
                    logger.error(f"Block {i} has broken link to previous block")
                    return False
        
        return True
    
    def propose_memory(self, memory_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Propose a new memory to the network for consensus
        
        Args:
            memory_data: Memory content and metadata
            
        Returns:
            Tuple of (success, message)
        """
        # Create a proposal object
        proposal = {
            "memory_data": memory_data,
            "proposer_node": self.node_id,
            "timestamp": time.time(),
            "proposal_id": str(uuid.uuid4())
        }
        
        # Save proposal to temporary directory
        self._save_proposal(proposal)
        
        # Broadcast to all nodes for voting
        approval_count = 0
        total_nodes = len(self.nodes)
        
        if total_nodes == 0:
            # If this is the only node, approve automatically
            logger.info("No other nodes in network, adding memory without consensus")
            block_hash = self.add_memory(memory_data)
            return True, f"Memory added with block hash {block_hash}"
        
        # Request votes from other nodes
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._request_vote, node, proposal["proposal_id"], proposal) 
                      for node in self.nodes]
            
            for future in futures:
                if future.result():
                    approval_count += 1
        
        # Add self-vote
        approval_count += 1
        total_nodes += 1
        
        # Check if we have sufficient quorum
        required_votes = (total_nodes * MIN_QUORUM_PERCENT) // 100
        
        if approval_count >= required_votes:
            # Designate a responsible node (simple round-robin for now)
            all_nodes = list(self.nodes) + [f"{self._get_ip()}:{DEFAULT_PORT}"]
            responsible_idx = hash(proposal["proposal_id"]) % len(all_nodes)
            responsible_node = all_nodes[responsible_idx]
            
            # Add to chain
            block_hash = self.add_memory(memory_data, responsible_node)
            
            # Broadcast the updated chain
            self._broadcast_chain_update()
            
            # Clean up the proposal
            self._remove_proposal(proposal["proposal_id"])
            
            return True, f"Memory accepted with {approval_count}/{total_nodes} votes. Block hash: {block_hash}"
        else:
            self._remove_proposal(proposal["proposal_id"])
            return False, f"Memory rejected. Only received {approval_count}/{total_nodes} votes, needed {required_votes}"
    
    def propose_task(self, task_data: Dict[str, Any], difficulty: str = "medium") -> Tuple[bool, str]:
        """
        Propose a new task to the network for consensus
        
        Args:
            task_data: Task content and metadata
            difficulty: Initial difficulty estimate
            
        Returns:
            Tuple of (success, message)
        """
        # Mark this as a task
        task_data["type"] = "task"
        task_data["task_state"] = TASK_PROPOSED
        task_data["task_difficulty"] = difficulty
        task_data["task_reward"] = DIFFICULTY_LEVELS.get(difficulty, 3)
        task_data["working_nodes"] = []
        task_data["solutions"] = []
        task_data["difficulty_votes"] = {self.node_id: difficulty}  # Add proposer's vote
        
        # Use regular memory proposal mechanism
        return self.propose_memory(task_data)
    
    def claim_task(self, task_id: str) -> Tuple[bool, str]:
        """
        Claim a task to work on it
        
        Args:
            task_id: ID of the task to claim
            
        Returns:
            Tuple of (success, message)
        """
        # Find the task in the chain
        block = self._find_block_by_memory_id(task_id)
        if not block:
            return False, f"Task {task_id} not found"
            
        # Check if it's a task
        if not block.is_task():
            return False, f"Memory {task_id} is not a task"
            
        # Check if task can be claimed
        if block.task_state not in [TASK_PROPOSED, TASK_ACCEPTED]:
            return False, f"Task {task_id} cannot be claimed (state: {block.task_state})"
            
        # Add node to working nodes
        if block.add_working_node(self.node_id):
            # Update task state
            block.update_task_state(TASK_IN_PROGRESS)
            
            # Save chain
            self.save_chain()
            
            # Broadcast update
            self._broadcast_chain_update()
            
            return True, f"Task {task_id} claimed successfully"
        else:
            return False, f"Already working on task {task_id}"
    
    def submit_solution(self, task_id: str, solution_data: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Submit a solution for a task
        
        Args:
            task_id: ID of the task
            solution_data: Solution content
            
        Returns:
            Tuple of (success, message)
        """
        # Find the task in the chain
        block = self._find_block_by_memory_id(task_id)
        if not block:
            return False, f"Task {task_id} not found"
            
        # Check if it's a task
        if not block.is_task():
            return False, f"Memory {task_id} is not a task"
            
        # Check if solution can be submitted
        if block.task_state not in [TASK_IN_PROGRESS, TASK_SOLUTION_PROPOSED]:
            return False, f"Cannot submit solution for task {task_id} (state: {block.task_state})"
            
        # Check if node is working on this task
        if self.node_id not in block.working_nodes:
            return False, f"Not authorized to submit solution for task {task_id}"
            
        # Add solution
        solution_index = len(block.solutions)
        if block.add_solution(self.node_id, solution_data):
            # Save chain
            self.save_chain()
            
            # Broadcast update
            self._broadcast_chain_update()
            
            return True, f"Solution #{solution_index} submitted for task {task_id}"
        else:
            return False, f"Failed to submit solution for task {task_id}"
    
    def vote_on_solution(self, task_id: str, solution_index: int, approve: bool) -> Tuple[bool, str]:
        """
        Vote on a proposed solution
        
        Args:
            task_id: ID of the task
            solution_index: Index of the solution
            approve: Whether to approve the solution
            
        Returns:
            Tuple of (success, message)
        """
        # Find the task in the chain
        block = self._find_block_by_memory_id(task_id)
        if not block:
            return False, f"Task {task_id} not found"
            
        # Check if it's a task
        if not block.is_task():
            return False, f"Memory {task_id} is not a task"
            
        # Check if solution exists
        if solution_index < 0 or solution_index >= len(block.solutions):
            return False, f"Solution #{solution_index} not found for task {task_id}"
            
        # Record vote
        solution = block.solutions[solution_index]
        solution["votes"][self.node_id] = approve
        
        # Update memory data to reflect change
        block.memory_data["solutions"] = block.solutions
        
        # Check if we have enough votes to approve or reject
        total_nodes = len(self.nodes) + 1  # +1 for self
        yes_votes = sum(1 for vote in solution["votes"].values() if vote)
        no_votes = sum(1 for vote in solution["votes"].values() if not vote)
        
        # Required votes for consensus
        required_votes = (total_nodes * MIN_QUORUM_PERCENT) // 100
        
        # If we have enough yes votes, approve the solution
        if yes_votes >= required_votes:
            # Mark task as completed
            block.update_task_state(TASK_COMPLETED)
            block.solver_node = solution["node_id"]
            block.memory_data["solver_node"] = solution["node_id"]
            
            # Award FeiCoins to the solver
            reward = block.reward
            self.wallet.add_funds(
                solution["node_id"], 
                reward, 
                f"Reward for solving task {task_id}"
            )
            
            # Save chain
            self.save_chain()
            
            # Broadcast update
            self._broadcast_chain_update()
            
            return True, f"Solution approved for task {task_id}. {reward} FeiCoins awarded to {solution['node_id']}"
            
        # If we have enough no votes, reject the solution
        elif no_votes >= required_votes:
            # Remove the solution
            block.solutions.pop(solution_index)
            block.memory_data["solutions"] = block.solutions
            
            # If no solutions left, return to in progress
            if not block.solutions:
                block.update_task_state(TASK_IN_PROGRESS)
            
            # Save chain
            self.save_chain()
            
            # Broadcast update
            self._broadcast_chain_update()
            
            return True, f"Solution rejected for task {task_id}"
        
        # Not enough votes yet
        else:
            # Save chain
            self.save_chain()
            
            # Broadcast update
            self._broadcast_chain_update()
            
            return True, f"Vote recorded for solution #{solution_index} of task {task_id}"
    
    def vote_on_task_difficulty(self, task_id: str, difficulty: str) -> Tuple[bool, str]:
        """
        Vote on the difficulty of a task
        
        Args:
            task_id: ID of the task
            difficulty: Proposed difficulty level
            
        Returns:
            Tuple of (success, message)
        """
        # Check if difficulty level is valid
        if difficulty not in DIFFICULTY_LEVELS:
            return False, f"Invalid difficulty level: {difficulty}"
            
        # Find the task in the chain
        block = self._find_block_by_memory_id(task_id)
        if not block:
            return False, f"Task {task_id} not found"
            
        # Check if it's a task
        if not block.is_task():
            return False, f"Memory {task_id} is not a task"
            
        # Record vote and recalculate difficulty
        block.vote_on_difficulty(self.node_id, difficulty)
        
        # Save chain
        self.save_chain()
        
        # Broadcast update
        self._broadcast_chain_update()
        
        return True, f"Difficulty vote recorded for task {task_id} (current: {block.difficulty}, reward: {block.reward})"
    
    def _find_block_by_memory_id(self, memory_id: str) -> Optional[MemoryBlock]:
        """
        Find a block by memory ID
        
        Args:
            memory_id: Memory ID to search for
            
        Returns:
            MemoryBlock or None if not found
        """
        with self.lock:
            for block in self.chain:
                block_memory_id = block.memory_data.get("metadata", {}).get("unique_id", "")
                if block_memory_id == memory_id:
                    return block
        return None
    
    def vote_on_proposal(self, proposal_id: str, proposal_data: Dict[str, Any]) -> bool:
        """
        Vote on a memory proposal from another node
        
        Args:
            proposal_id: Unique ID of the proposal
            proposal_data: The proposed memory data
            
        Returns:
            True to approve, False to reject
        """
        # Simple validation for demonstration
        # In a real system, you would implement more sophisticated validation rules
        memory_data = proposal_data["memory_data"]
        
        # Check if it has required fields
        if not all(key in memory_data for key in ["metadata", "headers", "content"]):
            logger.warning(f"Rejecting proposal {proposal_id}: Missing required fields")
            return False
        
        # Check if it has valid metadata
        if not all(key in memory_data["metadata"] for key in ["unique_id", "timestamp"]):
            logger.warning(f"Rejecting proposal {proposal_id}: Invalid metadata")
            return False
        
        # Check if memory already exists in chain
        memory_id = memory_data["metadata"]["unique_id"]
        if self._memory_exists_in_chain(memory_id):
            logger.warning(f"Rejecting proposal {proposal_id}: Memory already exists")
            return False
        
        # For demonstration, we approve all valid proposals
        # In a real system, you might implement more complex rules
        return True
    
    def _memory_exists_in_chain(self, memory_id: str) -> bool:
        """Check if a memory already exists in the chain by ID"""
        with self.lock:
            for block in self.chain:
                if block.memory_data.get("metadata", {}).get("unique_id", "") == memory_id:
                    return True
        return False
    
    def _request_vote(self, node: str, proposal_id: str, proposal: Dict[str, Any]) -> bool:
        """
        Request a vote from another node
        
        Args:
            node: Node address in format "ip:port"
            proposal_id: Unique ID of the proposal
            proposal: The proposal data
            
        Returns:
            True if approved, False otherwise
        """
        try:
            url = f"http://{node}/memorychain/vote"
            response = requests.post(url, json={
                "proposal_id": proposal_id,
                "proposal": proposal
            }, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("approved", False)
            
        except requests.RequestException as e:
            logger.warning(f"Failed to request vote from {node}: {e}")
        
        return False
    
    def _broadcast_chain_update(self) -> None:
        """Broadcast updated chain to all nodes"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(self._send_chain_update, node) for node in self.nodes]
            
            # We don't need to wait for results, fire and forget
    
    def _send_chain_update(self, node: str) -> bool:
        """
        Send updated chain to a specific node
        
        Args:
            node: Node address in format "ip:port"
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"http://{node}/memorychain/update"
            
            # Serialize the chain
            chain_data = self.serialize_chain()
            
            response = requests.post(url, json={
                "chain": chain_data,
                "node_id": self.node_id
            }, timeout=10)
            
            return response.status_code == 200
            
        except requests.RequestException as e:
            logger.warning(f"Failed to send chain update to {node}: {e}")
            return False
    
    def receive_chain_update(self, chain_data: List[Dict[str, Any]]) -> bool:
        """
        Process a chain update from another node
        
        Args:
            chain_data: Serialized chain data
            
        Returns:
            True if accepted, False otherwise
        """
        # Parse the incoming chain
        new_chain = []
        for block_data in chain_data:
            block = MemoryBlock.from_dict(block_data)
            new_chain.append(block)
        
        # Validate the new chain
        if len(new_chain) <= len(self.chain):
            logger.info("Received chain is not longer than current chain, ignoring")
            return False
        
        # Check validity of the new chain
        for i in range(1, len(new_chain)):
            current_block = new_chain[i]
            previous_block = new_chain[i-1]
            
            # Check hash integrity
            if current_block.hash != current_block.calculate_hash():
                logger.warning(f"Rejecting chain update: Block {i} has invalid hash")
                return False
            
            # Check chain continuity
            if current_block.previous_hash != previous_block.hash:
                logger.warning(f"Rejecting chain update: Block {i} has broken link to previous block")
                return False
        
        # Check that our existing chain is a subset of the new chain
        with self.lock:
            for i in range(len(self.chain)):
                if self.chain[i].hash != new_chain[i].hash:
                    logger.warning("Rejecting chain update: Chains have diverged")
                    return False
            
            # Accept the new chain
            self.chain = new_chain
            self.save_chain()
            
        logger.info(f"Chain updated to {len(self.chain)} blocks")
        return True
    
    def register_node(self, node_address: str) -> bool:
        """
        Add a new node to the network
        
        Args:
            node_address: Node address in format "ip:port"
            
        Returns:
            True if added, False otherwise
        """
        if node_address not in self.nodes:
            with self.lock:
                self.nodes.add(node_address)
            logger.info(f"Registered new node: {node_address}")
            return True
        return False
    
    def get_memories_by_responsible_node(self, node_id: str) -> List[Dict[str, Any]]:
        """
        Get all memories that a specific node is responsible for
        
        Args:
            node_id: Node identifier
            
        Returns:
            List of memory blocks
        """
        memories = []
        with self.lock:
            for block in self.chain:
                if block.responsible_node == node_id:
                    memories.append(block.memory_data)
        return memories
    
    def get_my_responsible_memories(self) -> List[Dict[str, Any]]:
        """
        Get all memories that this node is responsible for
        
        Returns:
            List of memory data dictionaries
        """
        return self.get_memories_by_responsible_node(self.node_id)
    
    def serialize_chain(self) -> List[Dict[str, Any]]:
        """
        Convert the chain to a list of dictionaries for serialization
        
        Returns:
            List of block dictionaries
        """
        with self.lock:
            return [block.to_dict() for block in self.chain]
    
    def save_chain(self) -> None:
        """Save the chain to disk"""
        with self.lock:
            chain_data = self.serialize_chain()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(CHAIN_FILE), exist_ok=True)
            
            with open(CHAIN_FILE, 'w') as f:
                json.dump(chain_data, f, indent=2)
    
    def load_chain(self) -> bool:
        """
        Load the chain from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(CHAIN_FILE):
                return False
                
            with open(CHAIN_FILE, 'r') as f:
                chain_data = json.load(f)
            
            with self.lock:
                self.chain = [MemoryBlock.from_dict(block_data) for block_data in chain_data]
                
            return len(self.chain) > 0
                
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.error(f"Error loading chain: {e}")
            return False
    
    def _save_proposal(self, proposal: Dict[str, Any]) -> None:
        """Save a proposal to the temporary directory"""
        os.makedirs(TEMP_PROPOSAL_DIR, exist_ok=True)
        
        proposal_path = os.path.join(TEMP_PROPOSAL_DIR, f"{proposal['proposal_id']}.json")
        
        with open(proposal_path, 'w') as f:
            json.dump(proposal, f, indent=2)
    
    def _remove_proposal(self, proposal_id: str) -> None:
        """Remove a proposal from the temporary directory"""
        proposal_path = os.path.join(TEMP_PROPOSAL_DIR, f"{proposal_id}.json")
        
        if os.path.exists(proposal_path):
            os.remove(proposal_path)
    
    def _query_node_status(self, node_address: str) -> Optional[Dict[str, Any]]:
        """
        Query the status of another node in the network
        
        Args:
            node_address: Address of the node to query (ip:port)
            
        Returns:
            Node status information or None if unavailable
        """
        try:
            response = requests.get(f"http://{node_address}/memorychain/node_status", timeout=3)
            if response.status_code == 200:
                status = response.json()
                # Add the node address to the status
                status["address"] = node_address
                return status
            return None
        except Exception:
            return None
    
    def _get_ip(self) -> str:
        """Get the local IP address"""
        try:
            # This trick gets the IP address that would be used to connect to Google's DNS
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"


class MemorychainNode:
    """
    HTTP server node for the memory chain network
    
    This class implements an HTTP server that exposes the memory chain functionality
    over a RESTful API, allowing nodes to communicate and reach consensus.
    
    Supports both memory and task management with FeiCoin rewards.
    """
    
    def __init__(self, port: int = DEFAULT_PORT, difficulty: int = 2, 
                 ai_model: str = "default", initial_status: str = "idle"):
        """
        Initialize the node
        
        Args:
            port: Port to listen on
            difficulty: Mining difficulty
            ai_model: AI model used by this FEI node
            initial_status: Initial status of the node
        """
        self.port = port
        self.node_id = str(uuid.uuid4())
        self.chain = MemoryChain(self.node_id, difficulty)
        
        # FEI node status information
        self.ai_model = ai_model
        self.status = initial_status
        self.status_timestamp = time.time()
        self.current_task_id = None
        self.capabilities = []
        self.load = 0.0  # 0.0 to 1.0 load indicator
        
        # Initialize Flask app
        try:
            from flask import Flask, request, jsonify
            self.app = Flask(__name__)
            
            # Define routes
            @self.app.route('/memorychain/vote', methods=['POST'])
            def vote():
                data = request.json
                proposal_id = data['proposal_id']
                proposal = data['proposal']
                
                approved = self.chain.vote_on_proposal(proposal_id, proposal)
                
                return jsonify({
                    "approved": approved,
                    "node_id": self.node_id
                })
            
            @self.app.route('/memorychain/update', methods=['POST'])
            def update_chain():
                data = request.json
                chain_data = data['chain']
                
                success = self.chain.receive_chain_update(chain_data)
                
                return jsonify({
                    "success": success,
                    "chain_length": len(self.chain.chain)
                })
            
            @self.app.route('/memorychain/propose', methods=['POST'])
            def propose_memory():
                data = request.json
                memory_data = data['memory']
                
                success, message = self.chain.propose_memory(memory_data)
                
                return jsonify({
                    "success": success,
                    "message": message
                })
            
            @self.app.route('/memorychain/propose_task', methods=['POST'])
            def propose_task():
                data = request.json
                task_data = data['task']
                difficulty = data.get('difficulty', 'medium')
                
                success, message = self.chain.propose_task(task_data, difficulty)
                
                return jsonify({
                    "success": success,
                    "message": message
                })
            
            @self.app.route('/memorychain/claim_task', methods=['POST'])
            def claim_task():
                data = request.json
                task_id = data['task_id']
                
                # Get the AI model if provided (for status tracking)
                ai_model = data.get('ai_model', self.ai_model)
                
                success, message = self.chain.claim_task(task_id)
                
                # Update node status if successful
                if success:
                    self.update_status(
                        status="working_on_task",
                        current_task_id=task_id,
                        ai_model=ai_model,
                        load=0.5  # Assume medium load when starting a task
                    )
                
                return jsonify({
                    "success": success,
                    "message": message,
                    "node_status": self.status,
                    "current_task": self.current_task_id
                })
            
            @self.app.route('/memorychain/submit_solution', methods=['POST'])
            def submit_solution():
                data = request.json
                task_id = data['task_id']
                solution_data = data['solution']
                
                # Get processing stats if provided
                solution_load = data.get('solution_load', 0.8)  # Default to high load for solution generation
                
                success, message = self.chain.submit_solution(task_id, solution_data)
                
                # Update node status if successful - solution submitted but waiting for voting
                if success:
                    self.update_status(
                        status="solution_proposed",
                        current_task_id=task_id,
                        load=0.2  # Lower load after solution submission
                    )
                
                return jsonify({
                    "success": success,
                    "message": message,
                    "node_status": self.status
                })
            
            @self.app.route('/memorychain/vote_solution', methods=['POST'])
            def vote_solution():
                data = request.json
                task_id = data['task_id']
                solution_index = data['solution_index']
                approve = data['approve']
                
                success, message = self.chain.vote_on_solution(task_id, solution_index, approve)
                
                # Look for status of task after vote
                block = self.chain._find_block_by_memory_id(task_id)
                if block and block.task_state == TASK_COMPLETED:
                    # If this node was the solution submitter, update status
                    solution = block.solutions[solution_index]
                    if solution["node_id"] == self.node_id:
                        self.update_status(
                            status="task_completed", 
                            current_task_id=None,
                            load=0.1
                        )
                
                return jsonify({
                    "success": success,
                    "message": message
                })
            
            @self.app.route('/memorychain/vote_difficulty', methods=['POST'])
            def vote_difficulty():
                data = request.json
                task_id = data['task_id']
                difficulty = data['difficulty']
                
                success, message = self.chain.vote_on_task_difficulty(task_id, difficulty)
                
                return jsonify({
                    "success": success,
                    "message": message
                })
            
            @self.app.route('/memorychain/wallet/balance', methods=['GET'])
            def get_balance():
                node_id = request.args.get('node_id', self.node_id)
                
                balance = self.chain.wallet.get_balance(node_id)
                
                return jsonify({
                    "node_id": node_id,
                    "balance": balance
                })
            
            @self.app.route('/memorychain/wallet/transactions', methods=['GET'])
            def get_transactions():
                node_id = request.args.get('node_id', None)
                limit = request.args.get('limit', 20, type=int)
                
                transactions = self.chain.wallet.get_transactions(node_id, limit)
                
                return jsonify({
                    "transactions": transactions,
                    "count": len(transactions)
                })
            
            @self.app.route('/memorychain/register', methods=['POST'])
            def register_node():
                data = request.json
                node_address = data['node_address']
                
                success = self.chain.register_node(node_address)
                
                # If successful, register the requester in our node list
                if success:
                    # Send our node list to the new node
                    try:
                        requests.post(f"http://{node_address}/memorychain/sync_nodes", json={
                            "nodes": list(self.chain.nodes),
                            "chain": self.chain.serialize_chain()
                        })
                    except:
                        pass
                
                return jsonify({
                    "success": success,
                    "total_nodes": len(self.chain.nodes)
                })
            
            @self.app.route('/memorychain/sync_nodes', methods=['POST'])
            def sync_nodes():
                data = request.json
                nodes = data['nodes']
                chain_data = data.get('chain')
                
                # Add all nodes to our list
                for node in nodes:
                    if node != f"{self._get_ip()}:{self.port}":
                        self.chain.register_node(node)
                
                # Optionally update our chain if provided
                if chain_data:
                    self.chain.receive_chain_update(chain_data)
                
                return jsonify({
                    "success": True,
                    "node_count": len(self.chain.nodes)
                })
            
            @self.app.route('/memorychain/chain', methods=['GET'])
            def get_chain():
                return jsonify({
                    "chain": self.chain.serialize_chain(),
                    "length": len(self.chain.chain)
                })
            
            @self.app.route('/memorychain/tasks', methods=['GET'])
            def get_tasks():
                # Get task-specific blocks
                tasks = []
                for block in self.chain.chain:
                    if block.is_task():
                        tasks.append({
                            "id": block.memory_data.get("metadata", {}).get("unique_id", ""),
                            "subject": block.memory_data.get("headers", {}).get("Subject", "No subject"),
                            "state": block.task_state,
                            "difficulty": block.difficulty,
                            "reward": block.reward,
                            "working_nodes": block.working_nodes,
                            "solution_count": len(block.solutions),
                            "solver": block.solver_node,
                            "proposer": block.proposer_node,
                            "block_index": block.index
                        })
                
                # Filter by state if requested
                state = request.args.get('state', None)
                if state:
                    tasks = [t for t in tasks if t["state"] == state]
                
                return jsonify({
                    "tasks": tasks,
                    "count": len(tasks)
                })
            
            @self.app.route('/memorychain/tasks/<task_id>', methods=['GET'])
            def get_task(task_id):
                # Find the task
                block = self.chain._find_block_by_memory_id(task_id)
                
                if not block or not block.is_task():
                    return jsonify({
                        "success": False,
                        "message": f"Task {task_id} not found"
                    }), 404
                
                # Return detailed task information
                task = {
                    "id": block.memory_data.get("metadata", {}).get("unique_id", ""),
                    "subject": block.memory_data.get("headers", {}).get("Subject", "No subject"),
                    "content": block.memory_data.get("content", ""),
                    "state": block.task_state,
                    "difficulty": block.difficulty,
                    "reward": block.reward,
                    "working_nodes": block.working_nodes,
                    "solutions": block.solutions,
                    "solver": block.solver_node,
                    "proposer": block.proposer_node,
                    "block_index": block.index,
                    "timestamp": block.timestamp,
                    "difficulty_votes": block.difficulty_votes
                }
                
                return jsonify(task)
                
            @self.app.route('/memorychain/network_status', methods=['GET'])
            def network_status():
                """Get status of all nodes in the network"""
                node_statuses = []
                
                # Add our own status
                own_status = {
                    "node_id": self.node_id,
                    "address": f"{self._get_ip()}:{self.port}",
                    "ai_model": self.ai_model,
                    "status": self.status,
                    "current_task": self.current_task_id,
                    "load": self.load,
                    "last_update": self.status_timestamp,
                    "feicoin_balance": self.chain.wallet.get_balance(self.node_id),
                    "is_self": True
                }
                node_statuses.append(own_status)
                
                # Query status from all other nodes
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_node = {
                        executor.submit(self._query_node_status, node): node 
                        for node in self.chain.nodes
                    }
                    
                    for future in future_to_node:
                        node = future_to_node[future]
                        try:
                            status = future.result()
                            if status:
                                status["is_self"] = False
                                node_statuses.append(status)
                        except Exception as e:
                            logger.warning(f"Failed to get status from node {node}: {e}")
                
                return jsonify({
                    "timestamp": time.time(),
                    "nodes": node_statuses,
                    "total_nodes": len(node_statuses),
                    "online_nodes": len(node_statuses),
                    "network_load": sum(n.get("load", 0) for n in node_statuses) / max(1, len(node_statuses))
                })
                
                # Return detailed task information
                task = {
                    "id": block.memory_data.get("metadata", {}).get("unique_id", ""),
                    "subject": block.memory_data.get("headers", {}).get("Subject", "No subject"),
                    "content": block.memory_data.get("content", ""),
                    "state": block.task_state,
                    "difficulty": block.difficulty,
                    "reward": block.reward,
                    "working_nodes": block.working_nodes,
                    "solutions": block.solutions,
                    "solver": block.solver_node,
                    "proposer": block.proposer_node,
                    "block_index": block.index,
                    "timestamp": block.timestamp,
                    "difficulty_votes": block.difficulty_votes
                }
                
                return jsonify(task)
            
            @self.app.route('/memorychain/responsible_memories', methods=['GET'])
            def get_responsible_memories():
                node_id = request.args.get('node_id', self.node_id)
                
                memories = self.chain.get_memories_by_responsible_node(node_id)
                
                return jsonify({
                    "memories": memories,
                    "count": len(memories)
                })
                
            @self.app.route('/memorychain/health', methods=['GET'])
            def health_check():
                return jsonify({
                    "status": "healthy",
                    "node_id": self.node_id,
                    "chain_length": len(self.chain.chain),
                    "connected_nodes": len(self.chain.nodes),
                    "feicoin_balance": self.chain.wallet.get_balance(self.node_id),
                    "ai_model": self.ai_model,
                    "fei_status": self.status,
                    "current_task": self.current_task_id,
                    "load": self.load,
                    "last_status_update": self.status_timestamp
                })
                
            @self.app.route('/memorychain/node_status', methods=['GET'])
            def node_status():
                """Return detailed FEI node status information"""
                active_tasks = []
                
                # Gather information about tasks this node is working on
                for block in self.chain.chain:
                    if block.is_task() and self.node_id in block.working_nodes:
                        if block.task_state in [TASK_IN_PROGRESS, TASK_SOLUTION_PROPOSED]:
                            task_id = block.memory_data.get("metadata", {}).get("unique_id", "")
                            subject = block.memory_data.get("headers", {}).get("Subject", "No subject")
                            active_tasks.append({
                                "task_id": task_id,
                                "subject": subject,
                                "state": block.task_state,
                                "difficulty": block.difficulty
                            })
                
                return jsonify({
                    "node_id": self.node_id,
                    "ai_model": self.ai_model,
                    "status": self.status,
                    "status_updated": self.status_timestamp,
                    "current_task_id": self.current_task_id,
                    "load": self.load,
                    "capabilities": self.capabilities,
                    "active_tasks": active_tasks,
                    "feicoin_balance": self.chain.wallet.get_balance(self.node_id),
                    "connected_nodes": len(self.chain.nodes),
                    "ip_address": self._get_ip(),
                    "port": self.port
                })
                
            @self.app.route('/memorychain/update_status', methods=['POST'])
            def update_status():
                """Update the node's status information"""
                data = request.json
                
                # Update status fields if provided
                if "status" in data:
                    self.status = data["status"]
                    self.status_timestamp = time.time()
                    
                if "ai_model" in data:
                    self.ai_model = data["ai_model"]
                    
                if "current_task_id" in data:
                    self.current_task_id = data["current_task_id"]
                    
                if "load" in data:
                    self.load = float(data["load"])
                    
                if "capabilities" in data:
                    self.capabilities = list(data["capabilities"])
                
                return jsonify({
                    "success": True,
                    "node_id": self.node_id,
                    "status": self.status,
                    "ai_model": self.ai_model,
                    "updated": self.status_timestamp
                })
                
            self.has_flask = True
            
        except ImportError:
            logger.error("Flask not installed. HTTP server functionality disabled.")
            self.app = None
            self.has_flask = False
    
    def start(self) -> None:
        """Start the HTTP server"""
        if not self.has_flask:
            logger.error("Cannot start server: Flask not installed")
            return
            
        logger.info(f"Starting Memorychain node on port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port)
    
    def update_status(self, status: str, current_task_id: Optional[str] = None, 
                   ai_model: Optional[str] = None, load: Optional[float] = None) -> None:
        """
        Update the node's status information
        
        Args:
            status: Current status (e.g., "idle", "busy", "processing_task")
            current_task_id: ID of task being worked on (if applicable)
            ai_model: AI model being used (if changed)
            load: Current load factor (0.0-1.0)
        """
        self.status = status
        self.status_timestamp = time.time()
        
        if current_task_id is not None:
            self.current_task_id = current_task_id
            
        if ai_model is not None:
            self.ai_model = ai_model
            
        if load is not None:
            self.load = load
    
    def connect_to_network(self, seed_node: str) -> bool:
        """
        Connect to the memory chain network via a seed node
        
        Args:
            seed_node: Address of a node in the network
            
        Returns:
            True if successful, False otherwise
        """
        if not self.has_flask:
            logger.error("Cannot connect to network: Flask not installed")
            return False
            
        try:
            # Register with the seed node
            response = requests.post(f"http://{seed_node}/memorychain/register", json={
                "node_address": f"{self._get_ip()}:{self.port}"
            })
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    logger.info(f"Successfully connected to network via {seed_node}")
                    
                    # Get the chain from the seed node
                    chain_response = requests.get(f"http://{seed_node}/memorychain/chain")
                    
                    if chain_response.status_code == 200:
                        chain_data = chain_response.json()
                        self.chain.receive_chain_update(chain_data["chain"])
                    
                    return True
            
            logger.error(f"Failed to connect to network: {response.text}")
            return False
            
        except requests.RequestException as e:
            logger.error(f"Error connecting to network: {e}")
            return False
    
    def _get_ip(self) -> str:
        """Get the local IP address"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"


# CLI command implementations

def start_node(args):
    """Start a memory chain node"""
    port = args.port or DEFAULT_PORT
    node = MemorychainNode(port=port, difficulty=args.difficulty)
    
    if args.seed:
        success = node.connect_to_network(args.seed)
        if not success:
            logger.warning(f"Failed to connect to seed node {args.seed}")
    
    node.start()

def propose_memory(args):
    """Propose a new memory to the chain"""
    # Create memory data from input
    memory_data = {}
    
    # Add headers
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
        "date": datetime.now(),
        "flags": list(args.flags) if args.flags else []
    }
    
    # Get content
    if args.file:
        with open(args.file, "r") as f:
            content = f.read()
    elif args.content:
        content = args.content
    else:
        logger.error("No content provided. Use --file or --content")
        return
    
    # Build full memory data
    memory_data = {
        "headers": headers,
        "metadata": metadata,
        "content": content
    }
    
    # Connect to local node and propose
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/propose", json={
            "memory": memory_data
        })
        
        result = response.json()
        if result.get("success", False):
            logger.info(f"Memory proposal accepted: {result.get('message', '')}")
        else:
            logger.error(f"Memory proposal rejected: {result.get('message', '')}")
            
    except requests.RequestException as e:
        logger.error(f"Error proposing memory: {e}")

def list_chain(args):
    """List the memory chain blocks"""
    try:
        response = requests.get(f"http://localhost:{args.port}/memorychain/chain")
        
        if response.status_code == 200:
            data = response.json()
            chain = data["chain"]
            
            print(f"Memory Chain ({len(chain)} blocks):")
            print("-" * 80)
            
            for block in chain:
                memory = block["memory_data"]
                subject = memory.get("headers", {}).get("Subject", "No subject")
                memory_id = memory.get("metadata", {}).get("unique_id", "")
                timestamp = datetime.fromtimestamp(block["timestamp"]).isoformat()
                
                print(f"Block #{block['index']} - {timestamp}")
                print(f"  Hash: {block['hash'][:10]}...")
                print(f"  Memory ID: {memory_id}")
                print(f"  Subject: {subject}")
                print(f"  Responsible Node: {block['responsible_node']}")
                print(f"  Proposer Node: {block['proposer_node']}")
                print("-" * 80)
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        logger.error(f"Error listing chain: {e}")

def list_responsible_memories(args):
    """List memories that a node is responsible for"""
    try:
        params = {}
        if args.node_id:
            params["node_id"] = args.node_id
            
        response = requests.get(f"http://localhost:{args.port}/memorychain/responsible_memories", params=params)
        
        if response.status_code == 200:
            data = response.json()
            memories = data["memories"]
            
            print(f"Responsible Memories ({len(memories)}):")
            print("-" * 80)
            
            for memory in memories:
                subject = memory.get("headers", {}).get("Subject", "No subject")
                memory_id = memory.get("metadata", {}).get("unique_id", "")
                timestamp = datetime.fromtimestamp(memory.get("metadata", {}).get("timestamp", 0)).isoformat()
                
                print(f"Memory {memory_id}")
                print(f"  Subject: {subject}")
                print(f"  Created: {timestamp}")
                print(f"  Tags: {memory.get('headers', {}).get('Tags', '')}")
                
                if args.with_content:
                    print("\nContent:")
                    print(memory.get("content", ""))
                    
                print("-" * 80)
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        logger.error(f"Error listing memories: {e}")

def connect_node(args):
    """Connect to the network via a seed node"""
    try:
        response = requests.post(f"http://localhost:{args.port}/memorychain/register", json={
            "node_address": args.seed
        })
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success", False):
                logger.info(f"Successfully connected to {args.seed}")
                logger.info(f"Total connected nodes: {data.get('total_nodes', 0)}")
            else:
                logger.error("Failed to connect to seed node")
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        logger.error(f"Error connecting to seed node: {e}")

def validate_chain(args):
    """Validate the integrity of the memory chain"""
    try:
        response = requests.get(f"http://localhost:{args.port}/memorychain/chain")
        
        if response.status_code == 200:
            data = response.json()
            chain_data = data["chain"]
            
            # Create a temporary chain to validate
            temp_chain = MemoryChain("validator")
            
            # Manually rebuild the chain from the received data
            temp_chain.chain = [MemoryBlock.from_dict(block) for block in chain_data]
            
            # Validate
            valid = temp_chain.validate_chain()
            
            if valid:
                logger.info(f"Chain is valid with {len(chain_data)} blocks")
            else:
                logger.error("Chain validation failed!")
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            
    except requests.RequestException as e:
        logger.error(f"Error validating chain: {e}")

def main():
    """Main entry point for the memorychain CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memorychain - Distributed memory ledger")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Node start command
    start_parser = subparsers.add_parser("start", help="Start a memory chain node")
    start_parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    start_parser.add_argument("-d", "--difficulty", type=int, default=2, help="Mining difficulty")
    start_parser.add_argument("-s", "--seed", help="Seed node to connect to")
    start_parser.set_defaults(func=start_node)
    
    # Propose memory command
    propose_parser = subparsers.add_parser("propose", help="Propose a new memory")
    propose_parser.add_argument("-s", "--subject", help="Memory subject")
    propose_parser.add_argument("-t", "--tags", help="Memory tags (comma-separated)")
    propose_parser.add_argument("-p", "--priority", choices=["high", "medium", "low"], help="Memory priority")
    propose_parser.add_argument("--status", help="Memory status")
    propose_parser.add_argument("--flags", default="", help="Memory flags (e.g., 'FP' for Flagged+Priority)")
    propose_parser.add_argument("--file", help="Read content from file")
    propose_parser.add_argument("--content", help="Memory content")
    propose_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    propose_parser.set_defaults(func=propose_memory)
    
    # List chain command
    list_parser = subparsers.add_parser("list", help="List the memory chain")
    list_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
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
    
    # Validate chain command
    validate_parser = subparsers.add_parser("validate", help="Validate chain integrity")
    validate_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Node port")
    validate_parser.set_defaults(func=validate_chain)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()