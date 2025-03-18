#!/usr/bin/env python3
"""
Tool definitions for Fei code assistant

This module provides definitions for Claude Universal Assistant tools.
"""

from typing import Dict, List, Any

# GlobTool definition
GLOB_TOOL = {
    "name": "GlobTool",
    "description": "Find files by name patterns using glob syntax (e.g., '**/*.js', 'src/**/*.ts'). Returns matching files sorted by modification time.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g., '**/*.py', 'src/**/*.ts')"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            }
        },
        "required": ["pattern"]
    }
}

# GrepTool definition
GREP_TOOL = {
    "name": "GrepTool",
    "description": "Search file contents using regex. Use include parameter to filter file types (e.g., '*.js'). Returns line numbers and content of matches.",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern (e.g., 'function\\s+\\w+', 'log.*Error')"
            },
            "include": {
                "type": "string",
                "description": "File types to search (e.g., '*.js', '*.{ts,tsx}')"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            }
        },
        "required": ["pattern"]
    }
}

# View tool definition
VIEW_TOOL = {
    "name": "View",
    "description": "Read file contents. Use absolute paths only. For large files, use limit and offset parameters.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to file"
            },
            "limit": {
                "type": "number",
                "description": "Max lines to read (for large files)"
            },
            "offset": {
                "type": "number",
                "description": "Starting line number (0-indexed)"
            }
        },
        "required": ["file_path"]
    }
}

# Edit tool definition
EDIT_TOOL = {
    "name": "Edit",
    "description": """Replace exact string in a file. For multiple edits, use RegexEdit instead.

REQUIREMENTS:
- old_string must be UNIQUE in the file
- Include 3-5 lines of context before/after edit point
- Include exact whitespace and indentation
- Use absolute file paths

To create new file:
- Provide the file_path
- Set old_string to empty
- Put content in new_string""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to file"
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to replace (with context)"
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text"
            }
        },
        "required": ["file_path", "old_string", "new_string"]
    }
}

# Replace tool definition
REPLACE_TOOL = {
    "name": "Replace",
    "description": "Overwrite file with new content or create a new file. Use absolute paths only.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to file"
            },
            "content": {
                "type": "string",
                "description": "Content to write"
            }
        },
        "required": ["file_path", "content"]
    }
}

# LS tool definition
LS_TOOL = {
    "name": "LS",
    "description": "List directory contents. Use absolute paths only. For targeted file searching, use GlobTool instead.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path to directory"
            },
            "ignore": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Patterns to ignore (e.g., ['*.log', 'node_modules'])"
            }
        },
        "required": ["path"]
    }
}

# Brave Search tool definition for Anthropic compatibility
BRAVE_SEARCH_TOOL = {
    "name": "brave_web_search",
    "description": "Search the web for current information using Brave Search.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": "Search query"
            },
            "count": {
                "type": "number",
                "description": "Results count (1-20, default: 10)"
            },
            "offset": {
                "type": "number",
                "description": "Pagination offset (default: 0)"
            }
        },
        "required": ["query"]
    }
}

# RegexEdit tool definition
REGEX_EDIT_TOOL = {
    "name": "RegexEdit",
    "description": """Edit files using regex patterns. Better than Edit when making multiple similar changes.

Examples:
- Change function names: pattern="def old_name\\(" replacement="def new_name("
- Update variables: pattern="var = (\\d+)" replacement="var = \\1 * 2"

Use capture groups (\\1, \\2) in replacement to reference matched groups.
Set validate=true to ensure code syntax remains valid.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute path to file"
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern with re.MULTILINE support"
            },
            "replacement": {
                "type": "string",
                "description": "Replacement text (can use \\1, \\2 for groups)"
            },
            "validate": {
                "type": "boolean",
                "description": "Validate syntax after changes (default: true)"
            },
            "validators": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Validators to use (e.g., ['ast'] for Python)"
            }
        },
        "required": ["file_path", "pattern", "replacement"]
    }
}

# BatchGlob tool for more efficient file searches
BATCH_GLOB_TOOL = {
    "name": "BatchGlob",
    "description": "Search for multiple file patterns at once. More efficient than multiple GlobTool calls.",
    "input_schema": {
        "type": "object",
        "properties": {
            "patterns": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of glob patterns to search"
            },
            "path": {
                "type": "string",
                "description": "Directory to search in (default: current directory)"
            },
            "limit_per_pattern": {
                "type": "number",
                "description": "Maximum files per pattern (default: 20)"
            }
        },
        "required": ["patterns"]
    }
}

# FindInFiles tool for more efficient content searching
FIND_IN_FILES_TOOL = {
    "name": "FindInFiles",
    "description": "Search for code patterns across specific files. More efficient than GrepTool for targeted searches.",
    "input_schema": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of file paths to search in"
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for"
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether search is case sensitive (default: false)"
            }
        },
        "required": ["files", "pattern"]
    }
}

# SmartSearch tool for context-aware searching
SMART_SEARCH_TOOL = {
    "name": "SmartSearch",
    "description": "Context-aware code search that finds relevant definitions, usages, and related code.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for (e.g., 'function process_data', 'class User')"
            },
            "context": {
                "type": "string",
                "description": "Additional context to narrow results (optional)"
            },
            "language": {
                "type": "string",
                "description": "Programming language to focus on (e.g., 'python', 'javascript')"
            }
        },
        "required": ["query"]
    }
}

# Repository Map tool definition
REPO_MAP_TOOL = {
    "name": "RepoMap",
    "description": "Generate a concise map of the code repository to understand structure and key components. Shows important classes, functions, and their relationships.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "token_budget": {
                "type": "number",
                "description": "Maximum tokens for the map (default: 1000)"
            },
            "exclude_patterns": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Patterns to exclude (e.g., ['**/*.log', 'node_modules/**'])"
            }
        }
    }
}

# Repository Summary tool definition
REPO_SUMMARY_TOOL = {
    "name": "RepoSummary",
    "description": "Generate a concise summary of the repository focused on key modules and dependencies. More token-efficient than full repo map.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "max_tokens": {
                "type": "number",
                "description": "Maximum tokens for summary (default: 500)"
            },
            "exclude_patterns": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Patterns to exclude (e.g., ['**/*.log', 'node_modules/**'])"
            }
        }
    }
}

# Repository Dependencies tool definition
REPO_DEPS_TOOL = {
    "name": "RepoDependencies",
    "description": "Extract and visualize dependencies between files and modules in the codebase. Helps understand code relationships.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Repository path (default: current directory)"
            },
            "module": {
                "type": "string",
                "description": "Optional module to focus on (e.g., 'fei/tools')"
            },
            "depth": {
                "type": "number",
                "description": "Dependency depth to analyze (default: 1)"
            }
        }
    }
}

# Shell tool definition
SHELL_TOOL = {
    "name": "Shell",
    "description": """Execute shell commands. Use with caution as this can modify your system.
    
Interactive commands (like games or GUI applications) will automatically run in background mode with a timeout.
Use the background parameter to force running in background or foreground mode.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute"
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default: 60)"
            },
            "current_dir": {
                "type": "string",
                "description": "Directory to run the command in (default: current directory)"
            },
            "background": {
                "type": "boolean",
                "description": "Force running in background mode (default: auto-detect for interactive commands)"
            }
        },
        "required": ["command"]
    }
}

# List of all tool definitions
TOOL_DEFINITIONS = [
    GLOB_TOOL,
    GREP_TOOL,
    VIEW_TOOL,
    EDIT_TOOL,
    REPLACE_TOOL,
    LS_TOOL,
    REGEX_EDIT_TOOL,
    BATCH_GLOB_TOOL,
    FIND_IN_FILES_TOOL,
    SMART_SEARCH_TOOL,
    REPO_MAP_TOOL,
    REPO_SUMMARY_TOOL,
    REPO_DEPS_TOOL,
    SHELL_TOOL
]

# Anthropic-specific tool definitions - Anthropic requires using 'custom' type for tool definitions
ANTHROPIC_TOOL_DEFINITIONS = [
    GLOB_TOOL,
    GREP_TOOL,
    VIEW_TOOL,
    EDIT_TOOL,
    REPLACE_TOOL,
    LS_TOOL,
    REGEX_EDIT_TOOL,
    BATCH_GLOB_TOOL,
    FIND_IN_FILES_TOOL,
    SMART_SEARCH_TOOL,
    REPO_MAP_TOOL,
    REPO_SUMMARY_TOOL,
    REPO_DEPS_TOOL,
    SHELL_TOOL,
    BRAVE_SEARCH_TOOL
]