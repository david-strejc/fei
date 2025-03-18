# FEI Developer Guide

## Commands

- `export ANTHROPIC_API_KEY=$(cat config/keys | grep ANTHROPIC_API_KEY | cut -d'=' -f2)` - Set API key from config
- `source venv/bin/activate` - Activate virtual environment
- `python -m fei ask "query"` - Run FEI with a query
- `python -m examples.mcp_brave_search` - Test Brave Search directly
- `python -m examples.mcp_brave_search --assistant` - Test Brave Search with assistant

## Code Style Guidelines

### Python
- 4 spaces indentation
- snake_case for methods/variables, PascalCase for classes
- Comprehensive docstrings in Google style
- Type hints with Optional/List/Dict types
- Asyncio for async operations

### Architecture 
- Tool registry pattern for LLM tool management
- MCP (Model Control Protocol) for external services
- Service layer pattern
- Event-based execution flow

### Error Handling
- Proper asyncio error handling
- ThreadPoolExecutor for blocking operations
- Exception handling with detailed logging
- Fallback mechanisms for external services