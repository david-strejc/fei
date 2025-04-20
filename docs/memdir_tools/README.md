# Memdir - Memory Management System

A Maildir-inspired hierarchical memory storage system for organizing and managing knowledge, notes, and information.

## Features

- **Maildir-compatible Structure**: Uses `cur/new/tmp` directories for each memory folder
- **Hierarchical Organization**: Support for nested folders (e.g., `.Projects/Python`)
- **Metadata Headers**: Each memory includes headers with tags, priority, subject, etc.
- **Flag System**: Status tracking using flags in filenames (Seen, Replied, Flagged, Priority)
- **Automatic Filtering**: Rule-based organization of memories based on content and headers
- **Cross-References**: Memories can reference other memories
- **Command-line Interface**: Full suite of tools for memory management

## Directory Structure

```
Memdir/
├── cur/               # Current/active memories
├── new/               # Newly created memories waiting for processing
├── tmp/               # Temporary storage during memory creation
├── .Trash/            # Deleted memories
│   ├── cur/
│   ├── new/
│   └── tmp/
├── .ToDoLater/        # Postponed memories
│   ├── cur/ 
│   ├── new/
│   └── tmp/
├── .Projects/         # Project-related memories
│   ├── Python/        # Subfolder for Python projects
│   │   ├── cur/
│   │   ├── new/
│   │   └── tmp/
│   └── AI/            # Subfolder for AI projects
│       ├── cur/
│       ├── new/
│       └── tmp/
└── .Archive/          # Long-term storage
    ├── cur/
    ├── new/
    └── tmp/
```

## Memory Format

Each memory is a plain text file with:

1. **Filename**: `timestamp.unique_id.hostname:2,flags`
   - Example: `1710429600.a1b2c3d4.server01:2,FR` (Flagged and Replied)

2. **Content Format**:
   ```
   Subject: Memory Title
   Date: 2025-03-14T14:30:00Z
   Tags: tag1,tag2,tag3
   Priority: high
   Status: active
   References: <memory_id>
   ---
   Memory content goes here...
   ```

## Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/username/memdir.git
cd memdir

# Install in development mode
pip install -e .
```

### Basic Commands

```bash
# Create a new memory
python -m memdir_tools create --subject "My Memory" --tags "important,notes" --content "This is a test memory"

# List memories in the inbox
python -m memdir_tools list

# Search memories
python -m memdir_tools search "python"

# View a specific memory
python -m memdir_tools view <memory_id>

# Create a new folder
python -m memdir_tools mkdir .Projects/NewProject

# Move a memory to another folder
python -m memdir_tools move <memory_id> "" ".Projects/NewProject"

# Flag a memory
python -m memdir_tools flag <memory_id> --add "FP"
```

### Folder Management

```bash
# Create a new memory folder
python -m memdir_tools create-folder .Projects/Work/ClientA

# List all memory folders
python -m memdir_tools list-folders

# List subfolders under a specific folder
python -m memdir_tools list-folders --parent .Projects --recursive

# Get folder statistics
python -m memdir_tools folder-stats .Projects/Work --include-subfolders

# Move a folder
python -m memdir_tools move-folder .Projects/OldName .Projects/NewName

# Copy a folder with all contents
python -m memdir_tools copy-folder .Projects/Template .Projects/NewProject

# Bulk tag all memories in a folder
python -m memdir_tools bulk-tag .Projects/Work --tags "work,project,important" --operation add
```

### Sample Data and Filters

```bash
# Create sample memories
python -m memdir_tools init-samples

# Run filters on new memories
python -m memdir_tools run-filters

# Run filters in dry-run mode (simulation)
python -m memdir_tools run-filters --dry-run

# Run filters on all memories
python -m memdir_tools run-filters --all
```

## Integration 

This system is designed to be compatible with standard Maildir tools, allowing for:

- Viewing with mail clients that support Maildir format
- Indexing with tools like notmuch or mu
- Synchronization with tools like mbsync or offlineimap
- Scripting with standard Unix tools

## Memory Maintenance and Archiving

Memdir includes a robust system for managing the lifecycle of memories:

### Archiving

Memories can be automatically archived based on:
- Age (e.g., memories older than 90 days)
- Tags (e.g., all memories with "completed" tag)
- Status (e.g., all memories marked as "done")

```bash
# Archive memories older than 90 days
python -m memdir_tools archive --age 90

# Archive memories from a specific folder
python -m memdir_tools archive --folder ".Projects/Python" --target ".Archive/Python"

# Simulate archiving without making changes
python -m memdir_tools archive --dry-run
```

### Cleanup

Obsolete or completed memories can be moved to trash or deleted:

```bash
# Move memories with status "completed" to trash
python -m memdir_tools cleanup --status "completed"

# Delete memories with specific tags
python -m memdir_tools cleanup --tags "obsolete,deprecated" --action delete

# Simulation mode
python -m memdir_tools cleanup --dry-run
```

### Retention Policies

Control how many memories are kept in each folder:

```bash
# Keep only the 10 newest memories in a folder
python -m memdir_tools retention --folder ".Projects/AI" --max 10 --mode age

# Keep only the 5 most important memories
python -m memdir_tools retention --folder "Inbox" --max 5 --mode importance
```

### Trash Management

```bash
# Empty trash (delete memories older than 30 days)
python -m memdir_tools empty-trash

# Change the age threshold for trash deletion
python -m memdir_tools empty-trash --age 7
```

### Status Updates

Automatically update memory statuses based on content and age:

```bash
# Update memory statuses
python -m memdir_tools update-status
```

### Full Maintenance

Run all maintenance operations at once:

```bash
# Full maintenance
python -m memdir_tools maintenance

# Simulation mode
python -m memdir_tools maintenance --dry-run
```

## Future Enhancements

- Web interface for browsing memories
- Full-text search indexing
- Attachment support
- IMAP/SMTP gateway for email integration
- Encryption for sensitive memories
- Mobile app integration