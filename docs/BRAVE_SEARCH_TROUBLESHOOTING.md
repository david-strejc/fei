# Brave Search Integration Troubleshooting

This document provides guidance for troubleshooting issues with the Brave Search integration in Fei.

## Implementation Notes

The Brave Search integration in Fei has been updated to bypass the MCP server completely and use the direct Brave Search API. This change was made after discovering that the MCP server doesn't properly implement the expected methods.

Previously, you might have seen errors like:

```
MCP service error: Method not found
```

The current implementation bypasses these issues by:

1. Skipping the MCP server entirely
2. Making direct API calls to the Brave Search API using the requests library
3. Using the API key from environment variables or a default key

### 2. API Key Issues

If you see errors related to authentication or API key issues:

1. Make sure you have a valid Brave Search API key
2. Set the API key in your `.env` file:
   ```
   BRAVE_API_KEY=your_api_key_here
   ```
3. The system will automatically use the API key from your environment variables

## Checking Installation

To ensure the MCP server package is installed correctly:

1. Check if the package is installed globally:
   ```bash
   npm list -g | grep modelcontextprotocol
   ```

2. If not installed, install it:
   ```bash
   npm install -g @modelcontextprotocol/server-brave-search
   ```

## Testing the Integration

You can test the Brave Search integration using the provided test script:

```bash
python test_brave_search.py
```

This script will:
1. Create an assistant that uses Brave Search
2. Ask a question that requires current information
3. Use the fallback mechanisms if needed to get results

## How Fallback Works

The integration is designed to be resilient:

1. First, it tries to use the MCP server's methods directly
2. If that fails, it falls back to direct API calls using the requests library
3. The API key is read from environment variables with a default fallback value

## Debugging

For more detailed debugging:

1. Set the logging level to DEBUG in your environment:
   ```bash
   export FEI_LOG_LEVEL=DEBUG
   ```

2. Run your script with this environment variable set
3. Check the log output for detailed information about what's happening

If you still encounter issues, please report them with the full debug logs.