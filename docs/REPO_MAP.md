# Repository Mapping for FEI

Repository mapping is a powerful feature that helps FEI understand the structure of your codebase more efficiently. This document explains how the repository mapping tools work and how to use them.

## Overview

The repository mapping tools in FEI are designed to:

1. Create a concise map of your entire codebase
2. Identify key components, classes, and functions
3. Detect dependencies between files and modules
4. Provide token-efficient context for the LLM

By using repository mapping, FEI can better understand your codebase without needing to see all the code, which would consume too many tokens. This approach is inspired by the repository mapping technique from [aider.chat](https://aider.chat/2023/10/22/repomap.html).

## Available Tools

### RepoMap

Generates a detailed map of the repository showing important classes, functions, and their relationships.

```python
result = invoke_tool("RepoMap", {
    "path": "/path/to/repo",
    "token_budget": 1000,
    "exclude_patterns": ["**/*.log", "node_modules/**"]
})
```

Parameters:
- `path`: Repository path (default: current directory)
- `token_budget`: Maximum tokens for the map (default: 1000)
- `exclude_patterns`: Patterns to exclude (optional)

### RepoSummary

Creates a concise summary of the repository focused on key modules and dependencies. This uses fewer tokens than the complete map.

```python
result = invoke_tool("RepoSummary", {
    "path": "/path/to/repo",
    "max_tokens": 500,
    "exclude_patterns": ["**/*.log", "node_modules/**"]
})
```

Parameters:
- `path`: Repository path (default: current directory)
- `max_tokens`: Maximum tokens for the summary (default: 500)
- `exclude_patterns`: Patterns to exclude (optional)

### RepoDependencies

Extracts and visualizes dependencies between files and modules in the codebase.

```python
result = invoke_tool("RepoDependencies", {
    "path": "/path/to/repo",
    "module": "fei/tools",  # Optional: focus on specific module
    "depth": 1
})
```

Parameters:
- `path`: Repository path (default: current directory)
- `module`: Optional module to focus on
- `depth`: Dependency depth to analyze (default: 1)

## Implementation Details

### Token-Efficient Code Understanding

The repository mapping tools provide a way for FEI to understand your codebase while using a significantly smaller number of tokens compared to loading entire files. This is achieved by:

1. Extracting only the most important symbols (classes, functions, methods)
2. Showing only function signatures and not entire implementations
3. Ranking files by importance using a PageRank-like algorithm
4. Focusing on module-level dependencies rather than line-by-line details

### Symbol Extraction

FEI uses two methods for extracting symbols:

1. **Tree-sitter Parsing (Preferred)**: Uses the `tree-sitter-languages` library to build a precise Abstract Syntax Tree (AST) for each source file. This provides accurate symbol extraction with proper parsing of the code.

2. **Regex-based Fallback**: When tree-sitter is not available, falls back to regex-based pattern matching for common code structures. This is less accurate but still provides useful information.

### Dependency Analysis

The repository mapping tools perform dependency analysis to understand how different parts of your codebase relate to each other:

1. **File-level dependencies**: Identifies which files reference symbols defined in other files
2. **Module-level dependencies**: Aggregates file dependencies to understand module relationships
3. **Importance ranking**: Uses a simplified PageRank algorithm to determine which files are most central to the codebase

## Usage Workflow

For the most effective code-understanding experience, follow this workflow:

1. Start with a repository summary to get a high-level overview:
   ```python
   repo_summary = invoke_tool("RepoSummary", {"path": "/path/to/repo"})
   ```

2. Generate a more detailed repository map with a higher token budget if needed:
   ```python
   repo_map = invoke_tool("RepoMap", {"path": "/path/to/repo", "token_budget": 2000})
   ```

3. Explore specific dependencies when focusing on modifying or understanding relationships:
   ```python
   deps = invoke_tool("RepoDependencies", {"path": "/path/to/repo", "module": "specific/module"})
   ```

4. Use the understanding from the repository map to make more targeted searches with the `GlobTool`, `GrepTool`, or `SmartSearch` tools.

## Example

You can see a complete demonstration of these tools in action by running:

```bash
python examples/repo_map_example.py
```

Or to analyze a specific repository:

```bash
python examples/repo_map_example.py --path /path/to/repo --tokens 2000
```

## Dependencies

To get the best results from repository mapping, install the `tree-sitter-languages` package:

```bash
pip install tree-sitter-languages
```

This package is included in the FEI requirements.txt file.