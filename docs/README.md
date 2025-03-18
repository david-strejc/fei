# Fei - Advanced Code Assistant

Fei (named after the Chinese flying dragon of adaptability) is a powerful code assistant that combines AI capabilities with advanced code manipulation tools.

## Features

### LLM Integration
- **Multiple LLM Providers**: Support for various AI providers through LiteLLM (Anthropic, OpenAI, Groq, etc.)
- **Configurable Models**: Easy selection of different LLM models
- **Provider Management**: Seamless switching between providers

### File Manipulation Tools
- **GlobTool**: Fast file pattern matching using glob patterns
- **GrepTool**: Content searching using regular expressions
- **View**: File viewing with line limiting and offset
- **Edit**: Precise code editing with context preservation
- **Replace**: Complete file content replacement
- **LS**: Directory listing with pattern filtering

### Token-Efficient Search Tools
- **BatchGlob**: Search for multiple file patterns in a single operation
- **FindInFiles**: Search for patterns across specific files
- **SmartSearch**: Context-aware code search for definitions and usage
- **RegexEdit**: Edit files using regex patterns for batch changes

*See [SEARCH_TOOLS.md](SEARCH_TOOLS.md) for detailed documentation on token-efficient tools*

### Repository Mapping Tools
- **RepoMap**: Generate a concise map of the repository structure and key components
- **RepoSummary**: Create a high-level summary of modules and dependencies
- **RepoDependencies**: Extract and visualize dependencies between files and modules

*See [REPO_MAP.md](REPO_MAP.md) for detailed documentation on repository mapping*

### MCP (Model Context Protocol) Services
- **Brave Search**: Web search integration for real-time information
- **Memory Service**: Knowledge graph for persistent memory
- **Fetch Service**: URL fetching for internet access
- **GitHub Service**: GitHub integration for repository management

*See [BRAVE_SEARCH_TROUBLESHOOTING.md](BRAVE_SEARCH_TROUBLESHOOTING.md) for troubleshooting Brave Search integration*

### Memdir Memory Management System
- **Maildir-compatible Structure**: Hierarchical memory organization with `cur/new/tmp` folders
- **Header-based Metadata**: Tags, priorities, statuses and custom fields
- **Flag-based Status Tracking**: Maildir-compatible filename flags (Seen, Replied, Flagged, Priority)
- **Filtering System**: Automatic organization of memories based on content and metadata
- **Archiving and Maintenance**: Lifecycle management with retention policies
- **CLI Tools**: Complete command-line interface for memory manipulation
- **HTTP API**: RESTful API for remote memory access and sharing between FEI instances
- **Memory Integration**: Seamless integration between FEI and memory system

### Memorychain - Distributed Memory Ledger
- **Blockchain Principles**: Tamper-proof chain with cryptographic verification
- **Consensus Mechanism**: Validate and accept memories through node voting
- **Distributed Processing**: Multiple nodes sharing memory responsibility
- **Memory References**: Reference memories in conversations using `#mem:id` syntax
- **Node Network**: Peer-to-peer communication between FEI instances
- **Shared Brain**: Create a collective intelligence across multiple agents
- **Status Reporting**: FEI nodes report AI model and operational status
- **Network Monitoring**: View status of all nodes in the network
- **Task Tracking**: Monitor which tasks nodes are working on

*See [MEMDIR_README.md](MEMDIR_README.md) for the memory system documentation*
*See [MEMORYCHAIN_README.md](MEMORYCHAIN_README.md) for the distributed memory system documentation*
*See [memdir_tools/HTTP_API_README.md](memdir_tools/HTTP_API_README.md) for HTTP API documentation*

All tools are seamlessly integrated with LLM providers to provide intelligent assistance for coding tasks.

## Installation

```bash
# Install from the current directory
pip install -e .

# Or install from GitHub
pip install git+https://github.com/yourusername/fei.git
```

## Usage

### Fei Code Assistant

```bash
# Start interactive chat (traditional CLI)
fei

# Start modern Textual-based chat interface
fei --textual

# Send a single message and exit
fei --message "Find all Python files in the current directory"

# Use a specific model
fei --model claude-3-7-sonnet-20250219

# Use a specific provider
fei --provider openai --model gpt-4o

# Test different providers
python test_litellm_integration.py --provider groq
python test_litellm_integration.py --all

# Enable debug logging
fei --debug

# Run the Textual interface example
python examples/textual_chat_example.py
```

### Memdir Memory Management

```bash
# Create memory directory structure
python -m memdir_tools

# Create a new memory
python -m memdir_tools create --subject "Meeting Notes" --tags "notes,meeting" --content "Discussion points..."

# List memories in folder
python -m memdir_tools list --folder ".Projects/Python"

# Search memories
python -m memdir_tools search "python"

# Advanced search with complex query
python -m memdir_tools search "tags:python,important date>now-7d Status=active sort:date" --format compact

# Run automatic filters
python -m memdir_tools run-filters

# Archive old memories
python -m memdir_tools maintenance

# Create sample memories
python -m memdir_tools init-samples

# Start the HTTP API server
python -m memdir_tools.run_server --generate-key

# Use the HTTP client
python examples/memdir_http_client.py list

# Run FEI with memory integration
python examples/fei_memdir_integration.py
```

# Memorychain Distributed Memory and Task System

```bash
# Start a Memorychain node
python -m memdir_tools.memorychain_cli start

# Start a second node on a different port
python -m memdir_tools.memorychain_cli start --port 6790

# Connect nodes together
python -m memdir_tools.memorychain_cli connect 127.0.0.1:6789 --port 6790

# Check node status and FeiCoin balance
python -m memdir_tools.memorychain_cli status

# Memory Management
python -m memdir_tools.memorychain_cli propose --subject "Shared Knowledge" --content "This memory will be distributed"
python -m memdir_tools.memorychain_cli list

# Task Management with FeiCoin Rewards
python -m memdir_tools.memorychain_cli task "Implement new algorithm" -d hard
python -m memdir_tools.memorychain_cli tasks
python -m memdir_tools.memorychain_cli claim taskid123
python -m memdir_tools.memorychain_cli solve taskid123 --file solution.py
python -m memdir_tools.memorychain_cli vote taskid123 0 --approve
python -m memdir_tools.memorychain_cli wallet

# Status Reporting and Monitoring
python -m memdir_tools.memorychain_cli network_status
python -m memdir_tools.memorychain_cli update_status --status busy --model "claude-3-opus" --load 0.7
python -m memdir_tools.memorychain_cli node_status

# Run FEI with Memorychain integration
python examples/fei_memorychain_example.py
# Use /status command in the chat to see network status
# Use /model command to change AI model (e.g., /model claude-3-sonnet)

# Run FEI with Status Reporting Example
python examples/fei_status_reporting_example.py
```

### Environment Variables

#### API Keys
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OPENAI_API_KEY`: Your OpenAI API key
- `GROQ_API_KEY`: Your Groq API key
- `BRAVE_API_KEY`: Your Brave Search API key
- `LLM_API_KEY`: Generic API key (fallback for LLM providers)

#### Configuration
- `FEI_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `FEI_LOG_FILE`: Path to log file
- `MEMDIR_API_KEY`: API key for Memdir HTTP server
- `MEMDIR_SERVER_URL`: URL of the Memdir HTTP server
- `MEMDIR_PORT`: Port for the Memdir HTTP server
- `MEMORYCHAIN_NODE`: Address of Memorychain node to connect to (default: localhost:6789)
- `MEMORYCHAIN_PORT`: Port for the Memorychain node to listen on (default: 6789)
- `MEMORYCHAIN_DIFFICULTY`: Mining difficulty for the Memorychain (default: 2)
- `MEMORYCHAIN_NODE_ID`: Override the node's ID (default: auto-generated UUID)
- `MEMORYCHAIN_AI_MODEL`: Default AI model for status reporting
- `MEMORYCHAIN_STATUS`: Default status (idle, busy, etc.)

## Project Structure

```
/
├── config/           # Configuration files and API keys
├── examples/         # Example usage scripts
├── fei/              # Main package
│   ├── core/         # Core assistant implementation
│   ├── tools/        # Code manipulation tools
│   ├── ui/           # User interfaces
│   ├── utils/        # Utility modules
│   └── tests/        # Test modules
├── requirements.txt  # Project dependencies
└── setup.py         # Installation script
```

## License

MIT