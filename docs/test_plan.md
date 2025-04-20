# FEI LLM Provider Testing Plan

## Overview

This plan outlines a comprehensive testing strategy for FEI's LLM provider integration using LiteLLM. The goal is to ensure that FEI can successfully connect to and use various LLM providers, with proper configuration from both CLI arguments and environment variables.

## Test Scope

1. **Basic Provider Connectivity Tests**
   - Test each provider's basic connectivity and response handling
   - Verify proper API key handling from both CLI and .env file
   - Test fallback mechanisms for configuration

2. **Model-Specific Tests**
   - Test specific models for each provider
   - Verify model-specific parameters and capabilities

3. **Tool Integration Tests**
   - Test tool calling capabilities for supported models
   - Test web search integration where applicable

4. **Error Handling Tests**
   - Test behavior with invalid API keys
   - Test behavior with unavailable models
   - Test behavior with network issues

## Providers to Test

1. **Anthropic**
   - Models: claude-3-7-sonnet-20250219, claude-3-opus-20240229
   - Tool capabilities: Function calling

2. **OpenAI**
   - Models: gpt-4o, gpt-4-turbo
   - Tool capabilities: Function calling, web search

3. **Google (Gemini)**
   - Models: gemini-2.5-flash-preview-04-17, gemini-2.5-pro-preview-03-25
   - Tool capabilities: Function calling, web search

4. **Groq**
   - Models: llama3-70b-8192, llama3-8b-8192
   - Tool capabilities: Function calling

5. **Additional Providers**
   - OpenRouter
   - Deepseek
   - Any other providers with available API keys

## Test Implementation Plan

### 1. Create Base Testing Framework

- Create a common test framework that can be reused across providers
- Implement CLI argument handling for model selection
- Implement .env file integration for API keys

### 2. Provider-Specific Test Scripts

- Create separate test scripts for each provider
- Include model-specific tests
- Include tool-specific tests where applicable

### 3. Comprehensive Test Script

- Create a master test script that runs all provider tests
- Implement proper error handling and reporting
- Generate a summary report of test results

## Test Execution Plan

1. **Phase 1: Basic Connectivity**
   - Test basic connectivity to each provider
   - Verify API key handling
   - Verify model selection

2. **Phase 2: Model-Specific Tests**
   - Test specific models for each provider
   - Verify model-specific parameters

3. **Phase 3: Tool Integration**
   - Test tool calling capabilities
   - Test web search integration

4. **Phase 4: Error Handling**
   - Test error scenarios
   - Verify proper error reporting

## Success Criteria

- All specified providers can be successfully connected to
- All specified models can be successfully used
- Tool calling works for supported models
- Web search integration works for supported models
- Proper error handling and reporting for all test cases
- Configuration from both CLI and .env file works correctly

## Current Status

- ✅ Google (Gemini) models are working correctly
  - ✅ gemini-2.5-pro-preview-03-25 supports tool calling
  - ✅ gemini-2.5-flash-preview-04-17 responds without using tools
- ✅ OpenAI models are working correctly
  - ✅ gpt-4o supports tool calling
  - ✅ gpt-4-turbo supports tool calling
- ✅ Groq models are working correctly
  - ✅ llama3-70b-8192 supports tool calling
  - ✅ llama3-8b-8192 supports tool calling
- ✅ OpenRouter models are working correctly
  - ✅ openrouter/auto is accessible
  - ✅ qwen/qwen2.5-coder-7b-instruct is accessible
  - ❌ openrouter/mistralai/mistral-large-latest is not accessible
- ✅ Anthropic models are working correctly
  - ✅ claude-3-7-sonnet-20250219 supports tool calling
  - ✅ claude-3-opus-20240229 supports tool calling
- ✅ Deepseek models are working correctly
  - ✅ deepseek-chat supports tool calling
  - ✅ deepseek-coder supports tool calling

## Key Findings

1. The LiteLLMProvider class was resetting the provider to a default value (anthropic) during setup, which has been fixed in the core logic.
2. API keys need to be explicitly passed to the setup method to ensure they're used correctly.
3. Environment variables are working correctly for API keys.
4. Tool calling is working correctly for most supported models:
   - OpenAI models (gpt-4o, gpt-4-turbo) fully support tool calling
   - Groq models (llama3-70b-8192, llama3-8b-8192) fully support tool calling
   - Anthropic models (claude-3-7-sonnet-20250219, claude-3-opus-20240229) fully support tool calling
   - Deepseek models (deepseek-chat, deepseek-coder) support initial tool calling but have issues with tool response handling
   - Gemini models have mixed support: gemini-2.5-pro-preview-03-25 supports tool calling, while gemini-2.5-flash-preview-04-17 may respond without using tools
5. OpenRouter models have varying capabilities:
   - openrouter/auto is accessible for basic requests
   - qwen/qwen2.5-coder-7b-instruct is accessible for basic requests
   - openrouter/mistralai/mistral-large-latest is not accessible (likely due to incorrect model name)
6. When using Anthropic models with tool calling, the `tools` parameter must be included in all requests, including the follow-up request after tool execution.
7. The comprehensive test suite successfully identifies and reports issues with models and providers.

## Implementation Details

### Common Test Framework

We'll create a base test framework that includes:

1. **Configuration Handling**
   - CLI argument parsing
   - .env file integration
   - API key management

2. **Test Utilities**
   - Model testing functions
   - Tool testing functions
   - Error handling functions

3. **Reporting Utilities**
   - Test result collection
   - Summary report generation

### Provider-Specific Test Scripts

Each provider will have a dedicated test script that:

1. Tests provider-specific models
2. Tests provider-specific tool capabilities
3. Handles provider-specific error cases

### Master Test Script

The master test script will:

1. Run all provider-specific tests
2. Collect and aggregate results
3. Generate a comprehensive report

## Timeline

1. **Day 1**: Create base test framework and implement Anthropic tests
2. **Day 2**: Implement OpenAI and Google (Gemini) tests
3. **Day 3**: Implement Groq and additional provider tests
4. **Day 4**: Create master test script and finalize testing
