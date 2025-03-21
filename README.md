# Fei - Advanced Code Assistant 🐉

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Version](https://img.shields.io/badge/version-0.1.0-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-early%20development-orange.svg)

> Fei (named after the Chinese flying dragon of adaptability) is a powerful code assistant that combines AI capabilities with advanced code manipulation tools and a distributed memory system.

<div align="center">
  <img src="https://github.com/stepanvalic/fei-david_streajc/blob/main/tmp-logo.jpg?raw=true" alt="Fei Logo" width="200"/>
</div>

## 📑 Table of Contents

- [Overview](#-overview)
- [Project Vision](#-project-vision)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [FEI Network](#-fei-network)
- [Architecture](#-architecture)
- [Documentation](#-documentation)
- [Known Issues & Bugs](#-known-issues--bugs)
- [Project Roadmap](#-project-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🔍 Overview

Fei is an advanced AI-powered code assistant built to enhance software development workflows. It integrates with multiple LLM providers, offers powerful code manipulation tools, and features a distributed memory system for persistent knowledge across sessions.

By leveraging the capabilities of large language models like Claude and GPT, Fei provides intelligent assistance for coding tasks, code search, refactoring, and documentation.

## 🌈 Project Vision

The Fei project represents more than just a code assistant; it's part of a broader vision for a democratized AI ecosystem called the FEI Network (Flying Dragon of Adaptability Network).

The FEI Network aims to be a truly democratic, distributed system of artificial intelligence that serves the collective good through:

1. **Distributed Processing**: Harnessing collective computational power across millions of individual nodes
2. **Specialized Intelligence Federation**: Creating a network of specialized intelligences that collaborate through open protocols
3. **Task-Oriented Contribution**: Participants contribute according to their capability, transforming computation from wasteful competition to purposeful collaboration
4. **Global Inclusion**: Active design for participation across economic, geographic, linguistic, and cultural boundaries
5. **Public Benefit Orientation**: Serving humanity's collective interests rather than narrow priorities

The project stands as an alternative to centralized AI approaches, focusing on human agency, democratization of artificial intelligence, and equitable distribution of computational power.

## ✨ Features

### LLM Integration

- **Multi-provider Support**: Seamless integration with Anthropic, OpenAI, Groq, and more through LiteLLM
- **Model Selection**: Easily switch between various LLM models
- **API Key Management**: Secure handling of API keys with proper precedence

### Code Manipulation Tools

- **Intelligent Search**:
  - `GlobTool`: Fast file pattern matching using glob patterns
  - `GrepTool`: Content searching using regular expressions
  - `SmartSearch`: Context-aware code search for definitions and usage

- **Code Editing**:
  - `View`: File viewing with line limiting and offset
  - `Edit`: Precise code editing with context preservation
  - `Replace`: Complete file content replacement
  - `RegexEdit`: Edit files using regex patterns for batch changes

- **Code Organization**:
  - `LS`: Directory listing with pattern filtering
  - `BatchGlob`: Search for multiple patterns in a single operation
  - `FindInFiles`: Search for patterns across specific files

### Memory Management System

- **Memdir**: Maildir-compatible memory organization
  - Hierarchical memory with cur/new/tmp folders
  - Header-based metadata and tags
  - Flag-based status tracking
  - Advanced filtering system
  - Memory lifecycle management

- **Memorychain**: Distributed memory ledger
  - Blockchain-inspired tamper-proof chain
  - Consensus-based memory validation
  - Peer-to-peer node communication
  - Shared brain across multiple agents
  - Node health monitoring and task tracking

### External Services (MCP)

- **Brave Search**: Web search integration for real-time information
- **Memory Service**: Knowledge graph for persistent memory
- **Fetch Service**: URL fetching for internet access
- **GitHub Service**: GitHub integration for repository management
- **Sequential Thinking**: Multi-step reasoning service

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/david-strejc/fei.git
cd fei

# Install from the current directory
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/david-strejc/fei.git
```

### Requirements

- Python 3.8 or higher
- Required API keys (at least one):
  - `ANTHROPIC_API_KEY`: Anthropic Claude API key
  - `OPENAI_API_KEY`: OpenAI API key
  - `GROQ_API_KEY`: Groq API key
  - `BRAVE_API_KEY`: Brave Search API key (for web search)

## 🚀 Usage

### Basic Usage

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

# Enable debug logging
fei --debug
```

### Python API

```python
from fei.core.assistant import Assistant

# Create an assistant
assistant = Assistant()

# Simple interaction
response = assistant.ask("What files contain the function 'process_data'?")
print(response)

# Interactive session
assistant.start_interactive_session()
```

### Environment Variables

Configure Fei through environment variables:

```bash
# API Keys
export ANTHROPIC_API_KEY=your_anthropic_api_key
export OPENAI_API_KEY=your_openai_api_key
export GROQ_API_KEY=your_groq_api_key
export BRAVE_API_KEY=your_brave_api_key

# Configuration
export FEI_LOG_LEVEL=DEBUG
export FEI_LOG_FILE=/path/to/logfile.log
```

### Memory System Usage

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
```

## 🌐 FEI Network

Fei is part of the broader FEI Network vision - a distributed, democratic system for collective intelligence. The network functions as a living, adaptive neural network composed of thousands of individual nodes, each with specialized capabilities.

### Core Network Principles

1. **Radical Openness**: Anyone with computing resources can participate
2. **Emergent Specialization**: Nodes naturally specialize based on their capabilities
3. **Autonomous Organization**: The network self-organizes through quorum-based decision making
4. **Value Reciprocity**: Contributions are fairly rewarded with FeiCoin
5. **Distributed Resilience**: No single point of failure or control

### Node Specializations

The FEI Network consists of specialized node types:

- **Mathematical Nodes**: Solving complex computational problems and formal reasoning
- **Creative Nodes**: Generation of text, images, music, and creative works
- **Analytical Nodes**: Pattern recognition, data analysis, and insight extraction
- **Knowledge Nodes**: Information retrieval, verification, and contextualization
- **Coordination Nodes**: Supporting collaboration between humans and AI systems

### Technical Implementation

The network is implemented through several layers:

- **Computation Layer**: Harnessing diverse hardware
- **Memory Layer**: Distributed storage of models and knowledge
- **Communication Layer**: Efficient routing of tasks and results
- **Verification Layer**: Ensuring quality and alignment with human values
- **Governance Layer**: Enabling collective decision-making

## 🏗️ Architecture

Fei's architecture is designed for extensibility and performance:

```
/
├── fei/                  # Main package
│   ├── core/             # Core assistant implementation
│   │   ├── assistant.py  # Main assistant class
│   │   ├── mcp.py        # MCP service integration
│   │   └── task_executor.py # Task execution logic
│   ├── tools/            # Code manipulation tools
│   │   ├── code.py       # File and code manipulation
│   │   ├── registry.py   # Tool registration
│   │   └── definitions.py # Tool definitions
│   ├── ui/               # User interfaces
│   │   ├── cli.py        # Command-line interface
│   │   └── textual_chat.py # TUI with Textual
│   └── utils/            # Utility modules
│       ├── config.py     # Configuration management
│       └── logging.py    # Logging setup
├── memdir_tools/         # Memory system
│   ├── server.py         # HTTP API server
│   ├── memorychain.py    # Distributed memory system
│   └── filter.py         # Memory filtering engine
└── examples/             # Example usage scripts
```

## 📚 Documentation

The Fei project includes comprehensive documentation in the `docs/` directory:

### Core Documents

- [FEI Manifesto](docs/FEI_MANIFESTO.md): Declaration of digital independence and collective intelligence
- [How FEI Network Works](docs/HOW_FEI_NETWORK_WORKS.md): Detailed explanation of the distributed network
- [Project Status](docs/PROJECT_STATUS.md): Current development status and roadmap
- [RepoMap](docs/REPO_MAP.md): Tools for understanding codebase structure
- [MemDir README](docs/MEMDIR_README.md): Documentation for the MemDir memory system
- [MemoryChain README](docs/MEMORYCHAIN_README.md): Documentation for the distributed memory ledger

### Feature Documentation

- [Brave Search Troubleshooting](docs/BRAVE_SEARCH_TROUBLESHOOTING.md): Resolving issues with web search
- [Search Tools](docs/SEARCH_TOOLS.md): Guide to code search capabilities
- [Textual README](docs/TEXTUAL_README.md): Documentation for the TUI interface

## ⚠️ Known Issues & Bugs

### Core Issues

1. **Error Handling**
   - Generic exception handling in `/root/fei/fei/core/assistant.py` masks specific errors
   - Swallowed exceptions in various components hide underlying problems
   - Missing proper checks before accessing nested attributes

2. **Race Conditions**
   - Process termination lacks proper synchronization
   - Background process management has potential race conditions
   - Complex asyncio event loop handling needs improvement

3. **Performance Issues**
   - Inefficient glob pattern matching with large codebases
   - Binary file detection is slow for large files
   - Memory usage can be high when processing many files

### Tool Limitations

1. **Edit Tool**
   - Requires unique context for find/replace operations
   - No support for multi-file refactoring
   - Limited validation capabilities

2. **Shell Command Execution**
   - Interactive commands are not fully supported
   - Command allowlisting is restrictive
   - Potential for zombie processes

3. **MCP Integration**
   - Limited error handling for network issues
   - No automatic reconnection for failed services
   - Response size limitations

### Memory System Issues

1. **Memdir**
   - No cleanup mechanism for old memories
   - Memory copy functionality not implemented
   - Folder creation is not recursive

2. **Memorychain**
   - Consensus mechanism is oversimplified
   - Limited protection against malicious nodes
   - No real blockchain implementation yet

## 🗺️ Project Roadmap

### v0.1 (Current)
- Basic assistant functionality
- File manipulation tools
- MCP service integration
- Command-line interface

### v0.2 (Next)
- Enhanced error handling
- Improved test coverage
- Performance optimization
- Documentation improvement

### v0.3 (Planned)
- Basic web UI
- Plugin system foundations
- Advanced code generation
- Multi-file operations

### v1.0 (Future)
- Stable API
- IDE integrations
- Comprehensive documentation
- Production-ready memory system

## 👥 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>
    Built with ❤️ by the Fei team
  </p>
  <p>
    <a href="https://github.com/david-strejc">GitHub</a> •
    <a href="https://github.com/david-strejc/fei/issues">Issues</a> •
    <a href="https://github.com/david-strejc/fei/blob/main/README.md">Documentation</a>
  </p>
  <p>
    <i>The FEI Network: Intelligence of the people, by the people, for the people.</i>
  </p>
</div>
