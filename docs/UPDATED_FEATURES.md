# Enhanced .env File Handling in Fei

## Overview
The Fei configuration system has been updated to provide more robust support for loading API keys from `.env` files. This update ensures:
1. Proper loading of API keys from `.env` files in multiple locations
2. Correct precedence order for key sources
3. Support for fallback to generic LLM_API_KEY for LLM providers
4. Preservation of manually set environment variables

## Enhancements

### Multiple .env File Locations
The system now looks for `.env` files in multiple locations:
- The specified env_file path (if provided)
- Current working directory (./env)
- User's home directory (~/.env)
- Fei project root directory

### Precedence Order for API Keys
Keys are sourced with the following precedence (highest to lowest):
1. Direct environment variables (set manually or by the system)
2. `.env` file environment variables
3. Config file values (e.g., ~/.fei.ini)
4. Default values

### LLM API Key Fallback
If a specific provider API key is not found (e.g., ANTHROPIC_API_KEY), the system will fall back to using a generic LLM_API_KEY for LLM providers.

### Environment Variable Preservation
When loading `.env` files, manually set environment variables are preserved and take precedence over values from the `.env` file.

## Testing
Comprehensive tests have been created to validate the implementation:
- Basic `.env` file reading
- Custom `.env` file locations
- Key precedence rules
- LLM_API_KEY fallback functionality
- Environment variable preservation
- Real application scenarios

## Usage Example
```python
# Example: Using .env file with Fei
from fei.utils.config import get_config

# Load the configuration (automatically loads from .env files)
config = get_config()

# Get API keys for different providers
anthropic_key = config.get('anthropic.api_key')
openai_key = config.get('openai.api_key')
groq_key = config.get('groq.api_key')
brave_key = config.get('brave.api_key')
```

## .env File Format Example
```
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
BRAVE_API_KEY=your_brave_api_key
LLM_API_KEY=your_generic_llm_api_key
FEI_CUSTOM_API_KEY=your_custom_api_key
```