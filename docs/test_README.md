# FEI LLM Provider Test Suite

This directory contains a comprehensive test suite for FEI's LLM provider integration using LiteLLM.

## Overview

The test suite includes:

1. **Individual Provider Tests**
   - `test_anthropic_models.py`: Tests Anthropic Claude models
   - `test_openai_models.py`: Tests OpenAI GPT models
   - `test_gemini_models.py`: Tests Google Gemini models
   - `test_groq_models.py`: Tests Groq models

2. **Comprehensive Tests**
   - `test_llm_providers.py`: Tests any specific provider or model
   - `test_all_providers.py`: Tests all configured providers and generates a report
   - `test_cli_model_selection.py`: Tests CLI argument handling for model selection

## Requirements

- Python 3.8+
- FEI codebase
- Provider API keys set in environment variables or .env file

## Usage

### Testing a Specific Provider

```bash
python test_anthropic_models.py
python test_openai_models.py
python test_gemini_models.py
python test_groq_models.py
```

### Testing a Specific Model

```bash
python test_anthropic_models.py --model claude-3-opus-20240229
python test_openai_models.py --model gpt-4o
python test_gemini_models.py --model gemini-2.5-pro-preview-03-25
python test_groq_models.py --model llama3-70b-8192
```

### Testing Tool Calling Capabilities

```bash
python test_anthropic_models.py --test-tools
python test_openai_models.py --test-tools
python test_gemini_models.py --test-tools
python test_groq_models.py --test-tools
```

### Testing All Providers

```bash
python test_all_providers.py
```

### Generating a Test Report

```bash
python test_all_providers.py --report provider_test_report.json
```

### Testing CLI Model Selection

```bash
python test_cli_model_selection.py
```

## Test Configuration

The test suite is configured to test the following providers and models:

1. **Anthropic**
   - claude-3-7-sonnet-20250219
   - claude-3-opus-20240229

2. **OpenAI**
   - gpt-4o
   - gpt-4-turbo

3. **Google (Gemini)**
   - gemini-2.5-flash-preview-04-17
   - gemini-2.5-pro-preview-03-25

4. **Groq**
   - llama3-70b-8192
   - llama3-8b-8192

5. **OpenRouter**
   - openrouter/auto
   - openrouter/mistralai/mistral-large-latest

6. **Deepseek**
   - deepseek-chat
   - deepseek-coder

## API Keys

The test suite requires API keys for each provider to be set in environment variables:

- Anthropic: `ANTHROPIC_API_KEY`
- OpenAI: `OPENAI_API_KEY`
- Google: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- Groq: `GROQ_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- Deepseek: `DEEPSEEK_API_KEY`

These can be set in the environment or in a `.env` file in the project root.

## Test Reports

The `test_all_providers.py` script can generate a JSON report with the results of all tests. The report includes:

- Timestamp
- Whether tool calling was tested
- Results for each provider and model
- Summary statistics

Example:

```json
{
  "timestamp": "2025-04-18T10:00:02.235146",
  "test_tools": false,
  "results": {
    "anthropic": {
      "claude-3-7-sonnet-20250219": false,
      "claude-3-opus-20240229": false
    },
    "openai": {
      "gpt-4o": false,
      "gpt-4-turbo": false
    },
    "gemini": {
      "gemini-2.5-flash-preview-04-17": true,
      "gemini-2.5-pro-preview-03-25": true
    }
  },
  "summary": {
    "providers_tested": 3,
    "models_tested": 6,
    "successful_models": 2,
    "failed_models": 4,
    "provider_status": {
      "anthropic": false,
      "openai": false,
      "gemini": true
    }
  }
}
```
