# Memdir: Memory Management System

Memdir is a sophisticated memory management system inspired by the Unix Maildir format. It provides a robust, hierarchical approach to storing, organizing, and retrieving knowledge and notes.

## Key Features

- **Maildir-compatible Structure**: Hierarchical organization with `cur/new/tmp` folders
- **Rich Metadata**: Headers for tags, priorities, statuses, and references
- **Flag-based Status Tracking**: Seen (S), Replied (R), Flagged (F), Priority (P)
- **Powerful Search**: Complex queries with tag, content, date, and flag filtering
- **Folder Management**: Create, move, copy, and organize memory folders
- **Memory Lifecycle**: Archiving, cleanup, and retention policies
- **Filtering System**: Automatic organization based on content and metadata

## Getting Started

1. Install dependencies:
   ```
   pip install python-dateutil
   ```

2. Generate sample memories:
   ```
   python -m memdir_tools init-samples
   ```

3. List your memories:
   ```
   python -m memdir_tools list
   ```

4. Search for memories:
   ```
   python -m memdir_tools search "#python"
   ```

## Command Reference

- `python -m memdir_tools create`: Create a new memory
- `python -m memdir_tools list`: List memories in a folder
- `python -m memdir_tools view`: View a specific memory
- `python -m memdir_tools search`: Search memories with advanced queries
- `python -m memdir_tools flag`: Add or remove flags on memories
- `python -m memdir_tools mkdir`: Create a new memory folder
- `python -m memdir_tools run-filters`: Apply organization filters
- `python -m memdir_tools maintenance`: Run archiving and cleanup

## Advanced Search System

Memdir includes a comprehensive search engine with powerful query capabilities:

```bash
# Basic search for keywords
python -m memdir_tools search "python learning"

# Search by tag with shorthand syntax
python -m memdir_tools search "#python #learning"

# Search with flags and status
python -m memdir_tools search "+F Status=active"

# Date-based filtering with relative dates
python -m memdir_tools search "date>now-7d sort:date"

# Complex regex patterns
python -m memdir_tools search "subject:/Project.*/ content:/function\s+\w+/"

# Output formats
python -m memdir_tools search "python" --format json
python -m memdir_tools search "python" --format csv
python -m memdir_tools search "python" --format compact
```

### Search Documentation

For detailed information about the search system, see:

- **Quick Help**: Run `python -m memdir_tools search --help` for command-line options
- **Complete Syntax**: Run `python -m memdir_tools.search --help` for comprehensive syntax reference
- **Detailed Guide**: Read the [Search Guide](memdir_tools/SEARCH_README.md) for full documentation including:
  - Field operators and comparison types
  - Special shortcuts for tags and flags
  - Date filtering and relative dates
  - Regex pattern matching
  - Sorting and pagination
  - Programmatic API usage
  - Best practices and tips

## Folder Structure

```
Memdir/
├── cur/       # Current (read) memories
├── new/       # New (unread) memories
├── tmp/       # Temporary files during creation
├── .Projects/ # Project-related memories
├── .Archive/  # Archived memories
├── .Trash/    # Deleted memories
└── ...        # User-created folders
```

## Memory Format

Each memory is stored as a plain text file with:

1. A header section containing metadata
2. A content section with the actual memory text
3. Separation by a "---" line

Example:
```
Subject: Python Learning Notes
Tags: python,learning,programming
Priority: high
Status: active
Date: 2025-03-14T01:25:15
---
# Python Learning Notes

## Key Concepts
- Everything in Python is an object
- Functions are first-class citizens
- Dynamic typing with strong type enforcement
```

## Filename Convention

Memdir uses the Maildir filename convention:
```
timestamp.unique_id.hostname:2,flags
```

Example: `1741911915.fcf18e11.Debian12:2,F`

## Future Enhancements

- Full-text search indexing
- Web interface for memory browsing
- Attachment support
- IMAP/SMTP gateway for email integration
- Encryption for sensitive memories