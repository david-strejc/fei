# FEI Web Search Integration

This feature enhances FEI (Flying Electronic Intelligence) with web search capabilities using the Brave Search API. With this integration, FEI can now provide up-to-date information by searching the web and combining the search results with AI-powered responses.

## Features Added

### 1. Brave Search Integration

- Direct API integration with Brave Search
- Fallback mechanism for when MCP server is unavailable
- Proper handling of API keys via environment variables
- Support for multiple result formats

### 2. Environment Variable Management

- Added proper .env file support
- Configured dotenv for loading environment variables
- Implemented fallback mechanisms for API keys

### 3. CLI Commands

- Added `search` command to FEI CLI for direct web searches
- Created standalone `ask_with_search.py` script for web-powered Q&A

## Usage Examples

### Search Command

Search the web directly from the FEI CLI:

```bash
# Basic search
python -m fei.ui.cli search "What are the latest features in Python 3.12?"

# Control number of results
python -m fei.ui.cli search --count 10 "Latest AI developments"
```

### Ask With Search Script

Ask questions with web search capabilities:

```bash
# Using default provider (Anthropic)
python examples/ask_with_search.py "What are the latest features in Python 3.12?"

# Using a specific provider
python examples/ask_with_search.py --provider openai "What are the latest features in Python 3.12?"

# Using a specific model
python examples/ask_with_search.py --provider groq --model "groq/llama3-70b-8192" "What are the latest features in Python 3.12?"
```

## Configuration

### .env File

Create a `.env` file in the project root with the following settings:

```
# API Keys for LLM Providers
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key

# Brave Search API Key
BRAVE_API_KEY=your_brave_api_key

# FEI Configuration
FEI_LOG_LEVEL=INFO
# FEI_LOG_FILE=/path/to/log/file.log

# Default provider configuration
FEI_DEFAULT_PROVIDER=anthropic
FEI_DEFAULT_MODEL=claude-3-7-sonnet-20250219
```

## Implementation Details

### Compatibility Issues Fixed

- Fixed incompatibilities between different LLM providers in LiteLLM
- Added proper handling of tool definitions for Anthropic and OpenAI
- Implemented direct search capabilities to bypass MCP issues

### Architecture Improvements

- Added abstraction for search functionality
- Improved configuration management
- Separated concerns between CLI and core functionality

## Future Improvements

- Extend support for more search providers
- Implement better MCP server discovery
- Add caching for search results
- Improve error handling and fallback mechanisms
- Add support for more specialized search types (academic, news, etc.)