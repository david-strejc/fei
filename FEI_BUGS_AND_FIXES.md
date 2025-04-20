# FEI Bugs and Fixes

This document outlines bugs identified during the development process and plans for fixing them.

## Identified Bugs

### 1. LiteLLMProvider Provider Reset Issue

**Description:**
The LiteLLMProvider class has a bug where it resets the provider to 'anthropic' during setup, which causes issues when trying to use other providers like 'gemini'.

**Location:**
`fei/providers/litellm_provider.py`

**Fix Plan:**
- Modify the `setup` method to preserve the provider value set by the caller
- Add a check to ensure the provider isn't reset to a default value
- Add tests to verify the fix works with different providers

### 2. CLI Argument Handling

**Description:**
The main CLI doesn't properly pass all arguments to the underlying UI implementations, causing some options to be ignored.

**Location:**
`fei/__main__.py`

**Fix Plan:**
- Update the argument parsing to include all relevant options
- Ensure arguments are properly passed to the UI implementations
- Add tests to verify argument handling

### 3. Model Configuration from Environment Variables

**Description:**
The model configuration from environment variables doesn't always take precedence over default values.

**Location:**
`fei/utils/config.py`

**Fix Plan:**
- Update the configuration loading logic to ensure environment variables take precedence
- Add tests to verify the configuration loading order

### 4. Gemini Model Handling

**Description:**
The Gemini model handling in the LiteLLM provider has issues with model name formatting and API key handling.

**Location:**
`fei/providers/litellm_provider.py`

**Fix Plan:**
- Update the model name formatting to ensure it works with Gemini models
- Fix the API key handling to properly use the GOOGLE_API_KEY environment variable
- Add tests to verify Gemini model handling

## Implementation Priority

1. Fix the LiteLLMProvider provider reset issue (highest priority)
2. Fix the Gemini model handling
3. Fix the model configuration from environment variables
4. Fix the CLI argument handling

## Testing Plan

For each bug fix:
1. Create a test case that reproduces the issue
2. Implement the fix
3. Verify the test case passes
4. Add additional tests to ensure the fix is robust

## Additional Improvements

1. Add better error handling for API key issues
2. Improve documentation for model configuration
3. Add more comprehensive tests for different providers
4. Enhance the CLI interface with better help messages and examples
