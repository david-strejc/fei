# FEI Tests

This directory contains all the tests for the FEI project. The tests are organized into subdirectories based on the component they test.

## Directory Structure

- `cli/`: Tests for the command-line interface
- `core/`: Tests for the core functionality
- `evolution/`: Tests for the evolution system
- `examples/`: Tests for the example code
- `generation/`: Tests for code generation features
- `litellm/`: Tests for LiteLLM integration
- `mcp/`: Tests for the MCP (Model Control Protocol)
- `mcp_consent/`: Tests for MCP consent features
- `memdir/`: Tests for the memory directory
- `memdir_tools/`: Tests for memory directory tools
- `memory/`: Tests for the memory system
- `misc/`: Miscellaneous tests
- `models/`: Tests for model functionality
  - `providers/`: Tests for specific model providers
- `providers/`: Tests for provider functionality
- `shell/`: Tests for shell command execution
- `textual/`: Tests for the Textual UI
- `tools/`: Tests for the tools system
- `ui/`: Tests for the user interface
- `utils/`: Tests for utility functions

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run tests for a specific component:

```bash
python -m pytest tests/models/
```

To run a specific test file:

```bash
python -m pytest tests/models/test_base.py
```

To run a specific test:

```bash
python -m pytest tests/models/test_base.py::TestModelBase::test_init
```
