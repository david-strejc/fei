# Memorychain: Distributed Memory and Task System for FEI

Memorychain is a distributed memory ledger system for FEI (Flying Dragon of Adaptability) inspired by blockchain principles. It enables multiple FEI nodes to share a common memory space with consensus-based validation, creating a "shared brain" across a network of AI assistants. Additionally, it supports distributed task allocation and completion with FeiCoin rewards.

## Key Features

- **Distributed Memory**: Share memories across multiple FEI instances
- **Blockchain-Inspired**: Tamper-proof chain with cryptographic verification
- **Consensus Mechanism**: Validate and accept memories through voting
- **Responsible Node Assignment**: Each memory has a designated owner
- **Memory References**: Reference memories with `#mem:id` syntax in conversations
- **Easy Integration**: Simple API for FEI and other systems
- **Task Management**: Propose, claim, and solve tasks with multiple workers
- **FeiCoin Rewards**: Earn tokens by completing tasks based on difficulty
- **Collaborative Voting**: Determine task difficulty through consensus

## Architecture

Memorychain consists of several components:

1. **Memory Blocks**: Individual memories with metadata and cryptographic hash
2. **Memory Chain**: Continuous chain of blocks with links to previous blocks
3. **Node System**: Network of independent nodes running the memory chain
4. **Consensus Protocol**: Voting system for memory acceptance
5. **FEI Integration**: Connection between FEI assistants and the memory chain

## Getting Started

### Prerequisites

- Python 3.7+
- A working FEI installation
- Flask (`pip install flask`) for the HTTP server
- Requests (`pip install requests`) for API communication

### Starting a Node

```bash
# Start a standalone node (first node in network)
python -m memdir_tools.memorychain_cli start

# Start a node on a specific port
python -m memdir_tools.memorychain_cli start --port 6790

# Start a node and connect to existing network
python -m memdir_tools.memorychain_cli start --seed 192.168.1.100:6789

# Check node status and FeiCoin balance
python -m memdir_tools.memorychain_cli status
```

### Memory Management

```bash
# Add a memory to the chain
python -m memdir_tools.memorychain_cli propose --subject "Meeting Notes" --content "Discussion points..."

# View the memory chain
python -m memdir_tools.memorychain_cli list

# View memories this node is responsible for
python -m memdir_tools.memorychain_cli responsible

# View a specific memory
python -m memdir_tools.memorychain_cli view [memory-id]

# Connect to another node
python -m memdir_tools.memorychain_cli connect 192.168.1.100:6789

# Validate chain integrity
python -m memdir_tools.memorychain_cli validate
```

### Task Management and FeiCoin

```bash
# Propose a new task
python -m memdir_tools.memorychain_cli task "Implement search algorithm" -d hard

# List all available tasks
python -m memdir_tools.memorychain_cli tasks

# List tasks by state (proposed, in_progress, completed)
python -m memdir_tools.memorychain_cli tasks --state in_progress

# View task details
python -m memdir_tools.memorychain_cli view-task [task-id] --content

# Claim a task to work on
python -m memdir_tools.memorychain_cli claim [task-id]

# Vote on task difficulty (affects reward)
python -m memdir_tools.memorychain_cli difficulty [task-id] extreme

# Submit a solution
python -m memdir_tools.memorychain_cli solve [task-id] --file solution.py

# Vote on a proposed solution
python -m memdir_tools.memorychain_cli vote [task-id] 0 --approve

# Check FeiCoin wallet balance and transactions
python -m memdir_tools.memorychain_cli wallet
```

## FEI Integration

### Using the MemorychainConnector

```python
from fei.tools.memorychain_connector import MemorychainConnector

# Connect to a local node
connector = MemorychainConnector()

# Add a memory
connector.add_memory(
    subject="Important Concept",
    content="Details about the concept...",
    tags="concept,important",
    priority="high"
)

# Search for memories
memories = connector.search_memories("concept")

# Get a specific memory
memory = connector.get_memory_by_id("memory-id")
```

### Memory References in Conversations

You can reference memories in conversations using the `#mem:id` syntax:

```
User: Can you tell me about #mem:a1b2c3d4?
Assistant: According to the memory, that refers to [content of memory]...
```

The system will automatically expand these references with the actual memory content.

### Interactive Example

An interactive example is provided in `/examples/fei_memorychain_example.py`:

```bash
python examples/fei_memorychain_example.py
```

This example demonstrates a FEI assistant with Memorychain integration, supporting commands like:

- `/save` - Save conversation highlights to the chain
- `/search [query]` - Search for memories
- `/list` - List recent memories
- `/view [id]` - View a specific memory
- `/help` - Show available commands

## How It Works

### Memory Chain Structure

Each memory is stored in a block containing:

- **Index**: Position in the chain
- **Timestamp**: Creation time
- **Memory Data**: The actual memory content and metadata
- **Previous Hash**: Cryptographic link to the previous block
- **Responsible Node**: Node ID designated to manage this memory
- **Proposer Node**: Node ID that proposed this memory
- **Hash**: Cryptographic hash of the block contents

### Consensus Process

When a memory is proposed:

1. The proposing node broadcasts the memory to all connected nodes
2. Each node validates the memory according to its rules
3. Nodes vote to accept or reject the memory
4. If a quorum (majority) approves, the memory is added to the chain
5. A responsible node is designated for the memory
6. All nodes update their copy of the chain

### Responsible Node Concept

Each memory is assigned a "responsible node" that:

- Is the primary contact point for that memory
- May perform additional processing on the memory
- Can be queried directly for that memory's content
- Ensures the memory remains available to the network

This distribution of responsibility helps balance the load across the network.

### Task Management Workflow

The task system extends the basic memory functionality with a specialized workflow:

1. **Task Proposal**: Any node can propose a task with an initial difficulty estimate
2. **Difficulty Voting**: Nodes can vote on the task's difficulty level, which determines the reward
3. **Task Claiming**: Nodes express interest in working on the task by claiming it
4. **Multiple Workers**: Multiple nodes can work on the same task simultaneously
5. **Solution Submission**: Nodes submit their solutions back to the network
6. **Solution Voting**: The network votes to approve or reject each solution
7. **Reward Distribution**: When a solution is approved, the solver node receives FeiCoins
8. **Task Completion**: The task is marked as completed, and its solutions become immutable

### FeiCoin Economy

FeiCoins serve as an incentive mechanism in the network:

- **Initial Allocation**: New nodes start with a base amount of FeiCoins
- **Task Rewards**: Completing tasks earns FeiCoins based on difficulty
- **Difficulty Levels**: Easy (1), Medium (3), Hard (5), Very Hard (10), Extreme (20)
- **Consensus-Based Difficulty**: The network votes to determine fair difficulty ratings
- **Transparent Ledger**: All transactions are recorded in the blockchain
- **Wallet Interface**: View balances and transaction history

## Technical Details

### Memory Block Format

Each block contains:

```json
{
  "index": 1,
  "timestamp": 1741911915,
  "memory_data": {
    "headers": {
      "Subject": "Important Concept",
      "Tags": "concept,important",
      "Priority": "high"
    },
    "metadata": {
      "unique_id": "a1b2c3d4-...",
      "timestamp": 1741911915,
      "date": "2025-03-14T15:25:15",
      "flags": ["F"]
    },
    "content": "Details about the concept..."
  },
  "previous_hash": "0a1b2c3d...",
  "responsible_node": "node-id-1",
  "proposer_node": "node-id-2",
  "nonce": 12345,
  "hash": "1a2b3c4d..."
}
```

### Node Networking

Nodes communicate over HTTP using a RESTful API:

#### Memory Management Endpoints
- `/memorychain/vote` - Vote on proposed memories
- `/memorychain/update` - Receive chain updates
- `/memorychain/propose` - Propose a new memory
- `/memorychain/register` - Register with the network
- `/memorychain/chain` - Get the full chain
- `/memorychain/responsible_memories` - Get assigned memories
- `/memorychain/health` - Check node status

#### Task Management Endpoints
- `/memorychain/propose_task` - Propose a new task
- `/memorychain/tasks` - List all tasks
- `/memorychain/tasks/{id}` - Get a specific task
- `/memorychain/claim_task` - Claim a task to work on
- `/memorychain/submit_solution` - Submit a solution for a task
- `/memorychain/vote_solution` - Vote on a proposed solution
- `/memorychain/vote_difficulty` - Vote on task difficulty

#### FeiCoin Endpoints
- `/memorychain/wallet/balance` - Get wallet balance
- `/memorychain/wallet/transactions` - List transactions

### Security Considerations

- **Cryptographic Integrity**: Each block contains a hash linked to the previous block
- **Consensus Validation**: Memories must be approved by a majority of nodes
- **Mining Difficulty**: Adjustable proof-of-work can be added for security
- **API Security**: Consider adding authentication for production use

## Extending the System

### Custom Validation Rules

You can extend the validation logic in the `vote_on_proposal` method of the `MemoryChain` class:

```python
def vote_on_proposal(self, proposal_id: str, proposal_data: Dict[str, Any]) -> bool:
    # Your custom validation logic here
    # Return True to approve, False to reject
```

### Scaling Considerations

For larger networks:

- Implement HTTPS for secure communication
- Add node authentication
- Add rate limiting to prevent flooding
- Consider sharding for very large memory sets
- Implement a more sophisticated consensus algorithm

## Advanced Topics

### Memory Chain Forks

If the chain forks (different versions exist on different nodes):

1. The longer chain is considered authoritative
2. Nodes will automatically switch to the longer chain
3. Orphaned memories may be re-proposed to the network

### Custom Consensus Algorithms

The default consensus is a simple majority vote. You could implement:

- Weighted voting based on node reputation
- Proof-of-stake concepts where certain nodes have more authority
- Domain-specific voting where nodes specialize in certain memory types

## Troubleshooting

### Common Issues

- **Node not connecting**: Check network connectivity and firewall settings
- **Memory proposals rejected**: Verify all required fields are present
- **Chain validation fails**: The chain may be corrupted or tampered with
- **Flask not installed**: Required for the HTTP server functionality

### Debugging

Enable debug mode for more verbose logging:

```bash
python -m memdir_tools.memorychain_cli start --debug
```

## Future Enhancements

- HTTPS support for secure communication
- User authentication system
- Advanced memory querying with vector embeddings
- Automated memory pruning and archiving
- Native browser interface for memory browsing
- Advanced consensus algorithms
- Decentralized node discovery

## Architecture Diagram

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│              │     │              │     │              │
│   FEI Node   │     │   FEI Node   │     │   FEI Node   │
│              │     │              │     │              │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │                    │                    │
┌──────▼───────┐     ┌──────▼───────┐     ┌──────▼───────┐
│              │     │              │     │              │
│ Memorychain  │◄────►  Memorychain │◄────► Memorychain  │
│    Node      │     │     Node     │     │    Node      │
│              │     │              │     │              │
└──────────────┘     └──────────────┘     └──────────────┘
        ▲                   ▲                    ▲
        │                   │                    │
        └───────────────────┼────────────────────┘
                            │
                      ┌─────▼─────┐
                      │           │
                      │  Shared   │
                      │ Memories  │
                      │           │
                      └───────────┘
```

## Reference

For more information on the underlying Memdir system, see [MEMDIR_README.md](/root/hacks/MEMDIR_README.md)