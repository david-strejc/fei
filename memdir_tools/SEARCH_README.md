# Memdir Advanced Search Guide

The Memdir system provides a powerful search engine that allows you to find, filter, and organize your memories with precision. This comprehensive guide explains all search capabilities from basic to advanced.

## Table of Contents
- [Quick Start](#quick-start)
- [Search Syntax](#search-syntax)
- [Field Operators](#field-operators)
- [Special Fields and Shortcuts](#special-fields-and-shortcuts)
- [Date Filtering](#date-filtering)
- [Sorting and Pagination](#sorting-and-pagination)
- [Output Formats](#output-formats)
- [Command Line Options](#command-line-options)
- [Examples](#examples)
- [Programmatic API](#programmatic-api)
- [Tips and Best Practices](#tips-and-best-practices)

## Quick Start

To get started quickly with memory search:

```bash
# Basic search for keywords
python -m memdir_tools search "python learning"

# Search by tags (two equivalent methods)
python -m memdir_tools search "tags:python,learning"
python -m memdir_tools search "#python #learning"

# Search by status and priority
python -m memdir_tools search "Status=active priority:high"

# Search with date range (last 7 days)
python -m memdir_tools search "date>now-7d"

# Find flagged memories
python -m memdir_tools search "+F"
```

## Search Syntax

The search query syntax is flexible and supports multiple ways to find your memories:

### Basic Keyword Search

When you provide simple text, Memdir searches both the subject and content:

```bash
python -m memdir_tools search "machine learning"
```

This finds memories that contain both "machine" and "learning" in either the subject or content.

### Field-Specific Search

To search in specific fields, use the `field:value` syntax:

```bash
python -m memdir_tools search "subject:python tags:learning"
```

This finds memories with "python" in the subject and the "learning" tag.

### Combining Multiple Conditions

You can combine multiple search conditions in a single query:

```bash
python -m memdir_tools search "subject:python tags:learning priority:high date>now-30d"
```

This finds high-priority memories from the last 30 days with "python" in the subject and the "learning" tag.

## Field Operators

Memdir supports various operators for field comparisons:

| Operator | Description | Example |
|----------|-------------|---------|
| `:` | Contains (case-insensitive) | `subject:python` |
| `=/regex/` | Matches regex pattern | `subject:/^Project.*$/` |
| `=` | Equals exactly | `priority=high` |
| `!=` | Not equals | `Status!=completed` |
| `>` | Greater than | `date>2023-01-01` |
| `<` | Less than | `date<2023-12-31` |
| `>=` | Greater than or equal | `priority>=medium` |
| `<=` | Less than or equal | `priority<=medium` |

### Regex Patterns

You can use regular expressions for pattern matching:

```bash
python -m memdir_tools search "subject:/^Project.*$/"
python -m memdir_tools search "content:/function\s+\w+/"
```

## Special Fields and Shortcuts

### Tags

Search for tags using either of these methods:

```bash
python -m memdir_tools search "tags:python,learning"  # Multiple tags with comma
python -m memdir_tools search "#python #learning"     # Using # shorthand
```

### Flags

Search for flags using either of these methods:

```bash
python -m memdir_tools search "flags:F"  # Flagged memories
python -m memdir_tools search "+F"       # Using + shorthand
```

Flag meanings:
- `F` - Flagged (important)
- `S` - Seen (read)
- `R` - Replied (responded to)
- `P` - Priority (high priority)

You can combine flags:

```bash
python -m memdir_tools search "+FP"  # Both Flagged and Priority
```

### Status

There are two different status fields:

1. `Status` - The memory Status header (active, pending, completed, etc.)
   ```bash
   python -m memdir_tools search "Status=active"
   ```

2. `status` - The Maildir status folder (cur, new, tmp)
   ```bash
   python -m memdir_tools search "status=new"
   ```

You can also use these aliases for the Status header:
- `status_value=active`
- `state=active`

## Date Filtering

### Absolute Dates

Search with absolute dates:

```bash
python -m memdir_tools search "date>2023-01-01 date<2023-12-31"
```

### Relative Dates

Search with relative dates:

```bash
python -m memdir_tools search "date>now-7d"  # Last 7 days
python -m memdir_tools search "date>now-1m"  # Last month
python -m memdir_tools search "date<now+1w"  # Next week
```

Supported units:
- `d` - Days
- `w` - Weeks
- `m` - Months (approximated as 30 days)
- `y` - Years (approximated as 365 days)

### Date Fields

You can search in various date fields:

```bash
python -m memdir_tools search "date>now-7d"     # Creation date
python -m memdir_tools search "Due>now Due<now+1w"  # Due date
```

## Sorting and Pagination

### Sorting

Sort results by any field with the `sort:` keyword:

```bash
python -m memdir_tools search "python sort:date"       # Sort by date (newest first)
python -m memdir_tools search "python sort:-date"      # Sort by date (oldest first)
python -m memdir_tools search "python sort:priority"   # Sort by priority
```

### Pagination

Limit results with the `limit:` keyword:

```bash
python -m memdir_tools search "python limit:10"        # Show only 10 results
python -m memdir_tools search "python limit:20 sort:date"  # Show 20 newest memories
```

## Output Formats

Control the output format with the `--format` option:

```bash
python -m memdir_tools search "python" --format text      # Default detailed format
python -m memdir_tools search "python" --format json      # JSON output
python -m memdir_tools search "python" --format csv       # CSV output
python -m memdir_tools search "python" --format compact   # One-line per memory
```

### Including Content

By default, search results don't include the memory content. To include it:

```bash
python -m memdir_tools search "python" --with-content
```

You can also use the `with_content` keyword in the query:

```bash
python -m memdir_tools search "python with_content"
```

## Command Line Options

The search command has many additional options:

```bash
python -m memdir_tools search --help
```

Key options:

| Option | Description |
|--------|-------------|
| `--folder` | Search only in a specific folder |
| `--status` | Search only in a specific status folder |
| `--format` | Output format (text, json, csv, compact) |
| `--with-content` | Include memory content in results |
| `--headers-only` | Search only in headers (not content) |
| `--sort` | Sort results by field |
| `--reverse` | Reverse sort order |
| `--limit` | Maximum number of results |
| `--offset` | Skip this many results (for pagination) |

## Examples

### Basic Searches

```bash
# Find memories about Python
python -m memdir_tools search "python"

# Find memories with specific tags
python -m memdir_tools search "#python #learning"

# Find memories with high priority
python -m memdir_tools search "priority:high"

# Find flagged memories
python -m memdir_tools search "+F"
```

### Advanced Queries

```bash
# Find active high-priority items created in the last week
python -m memdir_tools search "Status=active priority:high date>now-7d"

# Find memories about projects with regex
python -m memdir_tools search "subject:/^Project.*$/"

# Find incomplete tasks with due dates in the next week
python -m memdir_tools search "Status!=completed Due>now Due<now+1w"

# Find memories with specific content pattern
python -m memdir_tools search "content:/function\s+\w+/"
```

### Output Control

```bash
# Export search results to CSV
python -m memdir_tools search "python" --format csv > python_memories.csv

# Get compact results sorted by date
python -m memdir_tools search "python sort:date" --format compact

# Get JSON output with content
python -m memdir_tools search "python" --format json --with-content
```

### Folder-Specific Search

```bash
# Search in a specific folder
python -m memdir_tools search "python" --folder ".Projects/Python"

# Search in a specific status folder
python -m memdir_tools search "python" --status new
```

## Programmatic API

You can use the search functionality from Python:

```python
from memdir_tools.search import SearchQuery, search_memories, print_search_results

# Create a query programmatically
query = SearchQuery()
query.add_condition("Subject", "contains", "python")
query.add_condition("Tags", "has_tag", "learning")
query.set_sort("date", reverse=True)
query.set_pagination(limit=10, offset=0)
query.with_content(True)

# Execute search
results = search_memories(query)

# Process results
for memory in results:
    print(memory["headers"]["Subject"])
    print(memory["content"])

# Or use the pretty printer
print_search_results(results, output_format="text")
```

You can also parse a search string:

```python
from memdir_tools.search import parse_search_args, search_memories

# Parse a search string into a query
query = parse_search_args("subject:python tags:learning sort:date")

# Execute search
results = search_memories(query)
```

## Tips and Best Practices

### Efficient Searching

- Use field-specific searches for better precision
- Use tags consistently for easy organization
- Use flags (F, P, S, R) for quick status filtering
- Add a Status header to all memories for workflow tracking

### Tag Organization

Create a consistent tagging system:
- Use plural forms for categories: #projects, #ideas, #tasks
- Use singular forms for specific topics: #python, #javascript, #design
- Combine tags to create more specific filters: #projects #python

### Working with Dates

- Use relative dates for recurring searches
- Create saved search commands for common date ranges
- Use consistent date formats in custom date fields

### Power Features

- Combine field operators with flags: `+F priority:high`
- Use regex for complex pattern matching
- Create shell aliases for common searches
- Use the JSON output with `jq` for custom processing

### Troubleshooting

If search isn't returning expected results, try:
- Check case sensitivity (searches are case-insensitive by default)
- Verify field names match exactly
- Use `--debug` flag to see search internals
- Check if you're using the correct operator (contains vs. equals)

For a complete overview of the Memdir system, see [MEMDIR_README.md](/root/hacks/MEMDIR_README.md).