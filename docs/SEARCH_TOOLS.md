# Token-Efficient Search Tools for Fei

This document describes the token-efficient search tools for Fei. These tools are designed to minimize token usage while maintaining full functionality, making them more efficient for working with large codebases.

## Table of Contents
- [Overview](#overview)
- [Basic Tools](#basic-tools)
  - [GlobTool](#globtool)
  - [GrepTool](#greptool)
  - [View](#view)
  - [LS](#ls)
- [Advanced Tools](#advanced-tools)
  - [BatchGlob](#batchglob)
  - [FindInFiles](#findinfiles)
  - [SmartSearch](#smartsearch)
  - [RegexEdit](#regexedit)
- [Usage Examples](#usage-examples)

## Overview

The token-efficient search tools in Fei are designed to:

1. Use concise descriptions to minimize token usage
2. Support batch operations to reduce multiple tool calls
3. Provide targeted search capabilities for specific use cases
4. Return focused, relevant results

## Basic Tools

### GlobTool

Finds files by name patterns using glob syntax.

```python
# Example usage
result = invoke_tool("GlobTool", {
    "pattern": "**/*.py",
    "path": "/path/to/search"
})
```

### GrepTool

Searches file contents using regular expressions.

```python
# Example usage
result = invoke_tool("GrepTool", {
    "pattern": "function\\s+\\w+",
    "include": "*.js",
    "path": "/path/to/search"
})
```

### View

Reads file contents with support for line limiting and offset.

```python
# Example usage
result = invoke_tool("View", {
    "file_path": "/path/to/file.py",
    "limit": 100,  # Optional
    "offset": 50   # Optional
})
```

### LS

Lists files and directories in a given path.

```python
# Example usage
result = invoke_tool("LS", {
    "path": "/path/to/directory",
    "ignore": ["*.log", "node_modules"]  # Optional
})
```

## Advanced Tools

### BatchGlob

Searches for multiple file patterns in a single operation. More efficient than making multiple GlobTool calls.

```python
# Example usage
result = invoke_tool("BatchGlob", {
    "patterns": ["**/*.py", "**/*.js", "**/*.ts"],
    "path": "/path/to/search",
    "limit_per_pattern": 20  # Optional
})
```

### FindInFiles

Searches for code patterns across specific files. More efficient than GrepTool when you already know which files to search.

```python
# Example usage
result = invoke_tool("FindInFiles", {
    "files": ["/path/to/file1.py", "/path/to/file2.py"],
    "pattern": "def\\s+\\w+",
    "case_sensitive": False  # Optional
})
```

### SmartSearch

Context-aware code search that finds relevant definitions, usages, and related code.

```python
# Example usage
result = invoke_tool("SmartSearch", {
    "query": "class User",
    "language": "python",  # Optional
    "context": "authentication"  # Optional
})
```

### RegexEdit

Edits files using regex patterns. Better than Edit when making multiple similar changes.

```python
# Example usage
result = invoke_tool("RegexEdit", {
    "file_path": "/path/to/file.py",
    "pattern": "old_function_name\\(",
    "replacement": "new_function_name(",
    "validate": True  # Optional
})
```

## Usage Examples

### Combining Multiple Search Operations

When searching for multiple file types and then searching within those files, you can use BatchGlob followed by FindInFiles:

```python
# Step 1: Find all relevant files
files_result = invoke_tool("BatchGlob", {
    "patterns": ["**/*.py", "**/*.js"],
    "path": "/path/to/project"
})

# Step 2: Search within those files
all_files = []
for pattern, files in files_result["results"].items():
    if isinstance(files, list):
        all_files.extend(files)

search_result = invoke_tool("FindInFiles", {
    "files": all_files,
    "pattern": "function\\s+process"
})
```

### Finding Function Definitions

To find all function definitions in Python files:

```python
# Using SmartSearch
result = invoke_tool("SmartSearch", {
    "query": "function get_data",
    "language": "python"
})
```

### Batch Editing Multiple Files

To rename a function across multiple files:

```python
# First find all files with the function
files_with_function = invoke_tool("GrepTool", {
    "pattern": "old_function_name",
    "include": "**/*.py"
})

# Then edit each file
for file_path in files_with_function["results"].keys():
    edit_result = invoke_tool("RegexEdit", {
        "file_path": file_path,
        "pattern": "old_function_name\\(",
        "replacement": "new_function_name("
    })
```

You can see a complete demonstration of these tools in action by running:

```bash
python examples/efficient_search.py
```