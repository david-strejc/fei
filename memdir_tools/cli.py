#!/usr/bin/env python3
"""
Command-line interface for Memdir memory management
"""

import os
import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from memdir_tools.utils import (
    ensure_memdir_structure,
    get_memdir_folders,
    save_memory,
    list_memories,
    move_memory,
    search_memories,
    update_memory_flags,
    FLAGS,
    STANDARD_FOLDERS
)

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}

def colorize(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

def print_memory(memory: Dict[str, Any], show_content: bool = True) -> None:
    """
    Print memory information in a formatted way
    
    Args:
        memory: Memory info dictionary
        show_content: Whether to show the memory content
    """
    print(colorize(f"Memory ID: {memory['metadata']['unique_id']}", "bold"))
    print(f"File: {memory['filename']}")
    print(f"Date: {memory['metadata']['date'].isoformat()}")
    print(f"Folder: {memory['folder'] or 'Inbox'}")
    print(f"Status: {memory['status']}")
    print(f"Flags: {', '.join([FLAGS.get(f, f) for f in memory['metadata']['flags']])}" if memory['metadata']['flags'] else "Flags: None")
    
    print(colorize("\nHeaders:", "yellow"))
    for key, value in memory["headers"].items():
        print(f"  {key}: {value}")
    
    if show_content and "content" in memory:
        print(colorize("\nContent:", "green"))
        print(memory["content"])
    elif show_content and "content_preview" in memory:
        print(colorize("\nContent Preview:", "green"))
        print(memory["content_preview"])
    
    print("")

def create_memory(args: argparse.Namespace) -> None:
    """Create a new memory"""
    # Parse headers from args
    headers = {}
    if args.subject:
        headers["Subject"] = args.subject
    if args.tags:
        headers["Tags"] = args.tags
    if args.priority:
        headers["Priority"] = args.priority
    
    # Get content from file or stdin
    if args.file:
        with open(args.file, "r") as f:
            content = f.read()
    elif args.content:
        content = args.content
    else:
        print(colorize("Enter memory content (Ctrl+D to finish):", "yellow"))
        content_lines = []
        try:
            while True:
                line = input()
                content_lines.append(line)
        except EOFError:
            content = "\n".join(content_lines)
    
    # Create the memory
    folder = args.folder if args.folder else ""
    filename = save_memory(folder, content, headers, args.flags)
    
    print(colorize(f"Memory created: {filename}", "green"))
    print(f"In folder: {folder or 'Inbox'}")

def list_memories_cmd(args: argparse.Namespace) -> None:
    """List memories in a folder"""
    folder = args.folder if args.folder else ""
    status = args.status if args.status else "cur"
    
    memories = list_memories(folder, status, include_content=args.content)
    
    if not memories:
        print(colorize(f"No memories found in {folder or 'Inbox'}/{status}", "yellow"))
        return
    
    print(colorize(f"Found {len(memories)} memories in {folder or 'Inbox'}/{status}", "bold"))
    
    # Output format
    if args.json:
        # Convert datetime objects to strings for JSON serialization
        for memory in memories:
            memory["metadata"]["date"] = memory["metadata"]["date"].isoformat()
        print(json.dumps(memories, indent=2))
    else:
        for memory in memories:
            print_memory(memory, show_content=args.content)

def view_memory(args: argparse.Namespace) -> None:
    """View a specific memory"""
    folder = args.folder if args.folder else ""
    status = args.status if args.status else "cur"
    
    # First try to find by unique ID
    all_memories = []
    for s in STANDARD_FOLDERS:
        all_memories.extend(list_memories(folder, s, include_content=True))
    
    found = False
    for memory in all_memories:
        if args.id in (memory["filename"], memory["metadata"]["unique_id"]):
            print_memory(memory, show_content=True)
            found = True
            break
    
    if not found:
        print(colorize(f"Memory not found: {args.id}", "red"))

def move_memory_cmd(args: argparse.Namespace) -> None:
    """Move a memory from one folder to another"""
    # Find the memory first
    all_memories = []
    for s in STANDARD_FOLDERS:
        all_memories.extend(list_memories(args.source_folder, s))
    
    found = False
    for memory in all_memories:
        if args.id in (memory["filename"], memory["metadata"]["unique_id"]):
            # Found the memory, now move it
            source_status = memory["status"]
            result = move_memory(
                memory["filename"],
                args.source_folder,
                args.target_folder,
                source_status,
                args.target_status,
                args.flags
            )
            
            if result:
                print(colorize(f"Memory {args.id} moved from {args.source_folder or 'Inbox'}/{source_status} to {args.target_folder or 'Inbox'}/{args.target_status}", "green"))
            else:
                print(colorize(f"Failed to move memory {args.id}", "red"))
                
            found = True
            break
    
    if not found:
        print(colorize(f"Memory not found: {args.id}", "red"))

def search_memories_cmd(args: argparse.Namespace) -> None:
    """Search memories for a query"""
    folders = [args.folder] if args.folder else None
    statuses = [args.status] if args.status else None
    
    # Parse query string into a SearchQuery object
    from memdir_tools.search import parse_search_args, SearchQuery, search_memories, print_search_results
    
    # Create a SearchQuery with arguments from command line
    query = parse_search_args(args.query)
    
    # Apply additional CLI arguments
    if hasattr(args, 'sort') and args.sort:
        query.set_sort(args.sort, getattr(args, 'reverse', False))
    
    if hasattr(args, 'limit') and args.limit:
        query.set_pagination(limit=args.limit, offset=getattr(args, 'offset', 0))
    
    if hasattr(args, 'with_content') and args.with_content:
        query.with_content(True)
    
    # Determine output format
    output_format = "text"
    if hasattr(args, 'format'):
        output_format = args.format
    elif hasattr(args, 'json') and args.json:
        output_format = "json"
    
    # Execute search
    results = search_memories(query, folders, statuses, getattr(args, 'debug', False))
    
    if not results:
        print(colorize(f"No memories found matching query: {args.query}", "yellow"))
        return
    
    print(colorize(f"Found {len(results)} memories matching query: {args.query}", "bold"))
    
    # Print results with the correct formatter
    if output_format == "text":
        for memory in results:
            print_memory(memory, show_content=not args.headers_only)
            print("----------------------------------")
    else:
        print_search_results(results, output_format)

def flag_memory(args: argparse.Namespace) -> None:
    """Add or remove flags on a memory"""
    # Find the memory first
    all_memories = []
    folder = args.folder if args.folder else ""
    for s in STANDARD_FOLDERS:
        all_memories.extend(list_memories(folder, s))
    
    found = False
    for memory in all_memories:
        if args.id in (memory["filename"], memory["metadata"]["unique_id"]):
            # Found the memory, now update flags
            status = memory["status"]
            current_flags = "".join(memory["metadata"]["flags"])
            
            # Process flag changes
            if args.add:
                new_flags = current_flags + args.add
                # Ensure no duplicate flags
                new_flags = "".join(sorted(set(new_flags)))
            elif args.remove:
                new_flags = "".join([f for f in current_flags if f not in args.remove])
            elif args.set:
                new_flags = args.set
            else:
                print(colorize(f"Current flags: {current_flags}", "green"))
                found = True
                break
            
            result = update_memory_flags(
                memory["filename"],
                folder,
                status,
                new_flags
            )
            
            if result:
                print(colorize(f"Memory {args.id} flags updated from '{current_flags}' to '{new_flags}'", "green"))
            else:
                print(colorize(f"Failed to update flags for memory {args.id}", "red"))
                
            found = True
            break
    
    if not found:
        print(colorize(f"Memory not found: {args.id}", "red"))

def mkdir_cmd(args: argparse.Namespace) -> None:
    """Create a new memory directory"""
    from memdir_tools.utils import MEMDIR_BASE
    
    folder_path = os.path.join(MEMDIR_BASE, args.folder)
    
    try:
        # Create the directory structure
        for status in STANDARD_FOLDERS:
            os.makedirs(os.path.join(folder_path, status), exist_ok=True)
        
        print(colorize(f"Created memory directory: {args.folder}", "green"))
    except Exception as e:
        print(colorize(f"Error creating directory {args.folder}: {e}", "red"))

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Memdir - Memory Management Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # memdir-create command
    create_parser = subparsers.add_parser("create", help="Create a new memory")
    create_parser.add_argument("-s", "--subject", help="Memory subject")
    create_parser.add_argument("-t", "--tags", help="Memory tags (comma-separated)")
    create_parser.add_argument("-p", "--priority", choices=["high", "medium", "low"], help="Memory priority")
    create_parser.add_argument("-f", "--folder", help="Target folder (default: Inbox)")
    create_parser.add_argument("--flags", default="", help="Memory flags (e.g., 'FP' for Flagged+Priority)")
    create_parser.add_argument("--file", help="Read content from file")
    create_parser.add_argument("--content", help="Memory content")
    
    # memdir-list command
    list_parser = subparsers.add_parser("list", help="List memories in a folder")
    list_parser.add_argument("-f", "--folder", help="Folder to list (default: Inbox)")
    list_parser.add_argument("-s", "--status", choices=STANDARD_FOLDERS, default="cur", help="Status folder")
    list_parser.add_argument("-c", "--content", action="store_true", help="Show memory content")
    list_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    
    # memdir-view command
    view_parser = subparsers.add_parser("view", help="View a specific memory")
    view_parser.add_argument("id", help="Memory ID or filename")
    view_parser.add_argument("-f", "--folder", help="Folder (default: Inbox)")
    view_parser.add_argument("-s", "--status", choices=STANDARD_FOLDERS, help="Status folder")
    
    # memdir-move command
    move_parser = subparsers.add_parser("move", help="Move a memory to another folder")
    move_parser.add_argument("id", help="Memory ID or filename")
    move_parser.add_argument("source_folder", help="Source folder (use '' for Inbox)")
    move_parser.add_argument("target_folder", help="Target folder (use '' for Inbox)")
    move_parser.add_argument("-s", "--target-status", choices=STANDARD_FOLDERS, default="cur", help="Target status folder")
    move_parser.add_argument("--flags", help="New flags for the memory")
    
    # memdir-search command
    search_parser = subparsers.add_parser(
        "search", 
        help="Search memories with advanced query syntax", 
        description="Search memories using powerful query syntax and filtering options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUERY SYNTAX:
  Simple keywords:             python learning
  Tag search:                  #python #learning
  Field search:                subject:python tags:learning
  Regex search:                subject:/^Project.*/ content:/function\s+\w+/
  Equality match:              priority=high Status=active
  Comparison:                  date>2023-01-01 date<2023-12-31
  Date ranges:                 date>now-7d date<now
  Flag search:                 +F (Flagged) +S (Seen) +P (Priority) +R (Replied)

EXAMPLES:
  python -m memdir_tools search "#python #learning"
  python -m memdir_tools search "priority:high date>now-7d"
  python -m memdir_tools search "Status=active subject:/Project.*/ sort:date"
  python -m memdir_tools search "content:/import\s+os/" --format json
  python -m memdir_tools search "python" --folder ".Projects" --with-content

For complete documentation:
  python -m memdir_tools.search --help
"""
    )
    search_parser.add_argument("query", help="Search query with field operators and shortcuts")
    search_parser.add_argument("-f", "--folder", help="Folder to search (default: all folders)")
    search_parser.add_argument("-s", "--status", choices=STANDARD_FOLDERS, help="Status folder (default: all statuses)")
    search_parser.add_argument("--headers-only", action="store_true", help="Search only in headers")
    search_parser.add_argument("--format", choices=["text", "json", "csv", "compact"], default="text", 
                               help="Output format (text=full details, json=structured data, compact=one line per result)")
    search_parser.add_argument("--with-content", action="store_true", help="Include memory content in results")
    search_parser.add_argument("--sort", help="Sort results by field (e.g., date, subject, priority)")
    search_parser.add_argument("--reverse", action="store_true", help="Reverse sort order")
    search_parser.add_argument("--limit", type=int, help="Limit number of results")
    search_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    search_parser.add_argument("--json", action="store_true", help="Legacy option: equivalent to --format json")
    
    # memdir-flag command
    flag_parser = subparsers.add_parser("flag", help="Add or remove flags on a memory")
    flag_parser.add_argument("id", help="Memory ID or filename")
    flag_parser.add_argument("-f", "--folder", help="Folder (default: Inbox)")
    flag_parser.add_argument("--add", help="Flags to add")
    flag_parser.add_argument("--remove", help="Flags to remove")
    flag_parser.add_argument("--set", help="Set flags (replaces existing flags)")
    
    # memdir-mkdir command
    mkdir_parser = subparsers.add_parser("mkdir", help="Create a new memory directory")
    mkdir_parser.add_argument("folder", help="Folder name")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    return args

def main() -> None:
    """Main entry point"""
    # Ensure memdir structure exists
    ensure_memdir_structure()
    
    # Parse arguments
    args = parse_args()
    
    # Execute command
    if args.command == "create":
        create_memory(args)
    elif args.command == "list":
        list_memories_cmd(args)
    elif args.command == "view":
        view_memory(args)
    elif args.command == "move":
        move_memory_cmd(args)
    elif args.command == "search":
        search_memories_cmd(args)
    elif args.command == "flag":
        flag_memory(args)
    elif args.command == "mkdir":
        mkdir_cmd(args)

if __name__ == "__main__":
    main()