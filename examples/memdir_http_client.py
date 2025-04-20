#!/usr/bin/env python3
"""
Example client for the Memdir HTTP API.
Demonstrates how to interact with the Memdir server from another application.
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime

# Default server URL and API key
DEFAULT_SERVER_URL = "http://localhost:5000"
DEFAULT_API_KEY = "YOUR_API_KEY_HERE"

# Get server URL and API key from environment variables or use defaults
SERVER_URL = os.environ.get("MEMDIR_SERVER_URL", DEFAULT_SERVER_URL)
API_KEY = os.environ.get("MEMDIR_API_KEY", DEFAULT_API_KEY)

def setup_headers():
    """Set up headers with API key"""
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}

def check_server_health():
    """Check if the server is running and healthy"""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def list_memories(args):
    """List memories with optional filtering"""
    params = {}
    if args.folder:
        params["folder"] = args.folder
    if args.status:
        params["status"] = args.status
    if args.with_content:
        params["with_content"] = "true"
    
    response = requests.get(
        f"{SERVER_URL}/memories",
        headers=setup_headers(),
        params=params
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['count']} memories in {result['folder']}/{result['status']}")
        
        for memory in result["memories"]:
            print(f"\nID: {memory['metadata']['unique_id']}")
            print(f"Subject: {memory['headers'].get('Subject', 'No subject')}")
            print(f"Date: {memory['metadata']['date']}")
            print(f"Flags: {''.join(memory['metadata']['flags'])}")
            
            if args.with_content and "content" in memory:
                print(f"\nContent:\n{memory['content']}")
            
            print("-" * 40)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def create_memory(args):
    """Create a new memory"""
    # Build headers dictionary
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
    else:
        print("Enter memory content (Ctrl+D to finish):")
        content_lines = []
        try:
            while True:
                line = input()
                content_lines.append(line)
        except EOFError:
            content = "\n".join(content_lines)
    
    # Build request data
    data = {
        "content": content,
        "headers": headers,
        "folder": args.folder if args.folder else "",
        "flags": args.flags if args.flags else ""
    }
    
    # Send request
    response = requests.post(
        f"{SERVER_URL}/memories",
        headers=setup_headers(),
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Memory created: {result['filename']}")
        print(f"In folder: {result['folder']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def view_memory(args):
    """View a specific memory"""
    params = {}
    if args.folder:
        params["folder"] = args.folder
    
    response = requests.get(
        f"{SERVER_URL}/memories/{args.id}",
        headers=setup_headers(),
        params=params
    )
    
    if response.status_code == 200:
        memory = response.json()
        print(f"ID: {memory['metadata']['unique_id']}")
        print(f"File: {memory['filename']}")
        print(f"Date: {memory['metadata']['date']}")
        print(f"Folder: {memory['folder'] or 'Inbox'}")
        print(f"Status: {memory['status']}")
        print(f"Flags: {''.join(memory['metadata']['flags'])}")
        
        print("\nHeaders:")
        for key, value in memory["headers"].items():
            print(f"  {key}: {value}")
        
        if "content" in memory:
            print("\nContent:")
            print(memory["content"])
    else:
        print(f"Error: {response.status_code} - {response.text}")

def move_memory(args):
    """Move a memory from one folder to another"""
    data = {
        "source_folder": args.source_folder,
        "target_folder": args.target_folder,
        "target_status": args.target_status
    }
    
    if args.flags:
        data["flags"] = args.flags
    
    response = requests.put(
        f"{SERVER_URL}/memories/{args.id}",
        headers=setup_headers(),
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Memory moved: {args.id}")
        print(f"From: {result['source']}")
        print(f"To: {result['destination']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def update_flags(args):
    """Update the flags of a memory"""
    data = {
        "source_folder": args.folder,
        "flags": args.flags
    }
    
    response = requests.put(
        f"{SERVER_URL}/memories/{args.id}",
        headers=setup_headers(),
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Memory flags updated: {args.id}")
        print(f"New flags: {result['new_flags']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def delete_memory(args):
    """Move a memory to the trash folder"""
    params = {}
    if args.folder:
        params["folder"] = args.folder
    
    response = requests.delete(
        f"{SERVER_URL}/memories/{args.id}",
        headers=setup_headers(),
        params=params
    )
    
    if response.status_code == 200:
        print(f"Memory moved to trash: {args.id}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def search_memories(args):
    """Search memories with query parameters"""
    params = {"q": args.query}
    
    if args.folder:
        params["folder"] = args.folder
    if args.status:
        params["status"] = args.status
    if args.limit:
        params["limit"] = args.limit
    if args.offset:
        params["offset"] = args.offset
    if args.with_content:
        params["with_content"] = "true"
    if args.debug:
        params["debug"] = "true"
    
    response = requests.get(
        f"{SERVER_URL}/search",
        headers=setup_headers(),
        params=params
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Found {result['count']} matching memories")
        
        for memory in result["results"]:
            print(f"\nID: {memory['metadata']['unique_id']}")
            print(f"Subject: {memory['headers'].get('Subject', 'No subject')}")
            print(f"Date: {memory['metadata']['date']}")
            print(f"Folder: {memory['folder'] or 'Inbox'}/{memory['status']}")
            print(f"Tags: {memory['headers'].get('Tags', '')}")
            print(f"Flags: {''.join(memory['metadata']['flags'])}")
            
            if args.with_content and "content" in memory:
                print(f"\nContent:\n{memory['content']}")
            
            print("-" * 40)
    else:
        print(f"Error: {response.status_code} - {response.text}")

def list_folders(args):
    """List all folders in the Memdir structure"""
    response = requests.get(
        f"{SERVER_URL}/folders",
        headers=setup_headers()
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Memdir Folders:")
        for folder in result["folders"]:
            print(f"  {folder or 'Inbox'}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def create_folder(args):
    """Create a new folder"""
    data = {"folder": args.folder}
    
    response = requests.post(
        f"{SERVER_URL}/folders",
        headers=setup_headers(),
        json=data
    )
    
    if response.status_code == 200:
        print(f"Folder created: {args.folder}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def delete_folder(args):
    """Delete a folder"""
    response = requests.delete(
        f"{SERVER_URL}/folders/{args.folder}",
        headers=setup_headers()
    )
    
    if response.status_code == 200:
        print(f"Folder deleted: {args.folder}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def rename_folder(args):
    """Rename a folder"""
    data = {"new_name": args.new_name}
    
    response = requests.put(
        f"{SERVER_URL}/folders/{args.folder}",
        headers=setup_headers(),
        json=data
    )
    
    if response.status_code == 200:
        print(f"Folder renamed from {args.folder} to {args.new_name}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def run_filters(args):
    """Run all filters to organize memories"""
    data = {"dry_run": args.dry_run}
    
    response = requests.post(
        f"{SERVER_URL}/filters/run",
        headers=setup_headers(),
        json=data
    )
    
    if response.status_code == 200:
        result = response.json()
        print("Filters executed successfully")
        
        if result.get("actions"):
            print("\nActions performed:")
            for action in result["actions"]:
                print(f"  {action}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Memdir HTTP API Client")
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List memories")
    list_parser.add_argument("-f", "--folder", help="Folder to list (default: Inbox)")
    list_parser.add_argument("-s", "--status", choices=["cur", "new", "tmp"], default="cur", help="Status folder")
    list_parser.add_argument("-c", "--with-content", action="store_true", help="Include content in results")
    list_parser.set_defaults(func=list_memories)
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new memory")
    create_parser.add_argument("-s", "--subject", help="Memory subject")
    create_parser.add_argument("-t", "--tags", help="Memory tags (comma-separated)")
    create_parser.add_argument("-p", "--priority", choices=["high", "medium", "low"], help="Memory priority")
    create_parser.add_argument("-f", "--folder", help="Target folder (default: Inbox)")
    create_parser.add_argument("--flags", help="Memory flags (e.g., 'FP' for Flagged+Priority)")
    create_parser.add_argument("--file", help="Read content from file")
    create_parser.set_defaults(func=create_memory)
    
    # View command
    view_parser = subparsers.add_parser("view", help="View a specific memory")
    view_parser.add_argument("id", help="Memory ID or filename")
    view_parser.add_argument("-f", "--folder", help="Folder (default: all folders)")
    view_parser.set_defaults(func=view_memory)
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move a memory to another folder")
    move_parser.add_argument("id", help="Memory ID or filename")
    move_parser.add_argument("source_folder", help="Source folder (use '' for Inbox)")
    move_parser.add_argument("target_folder", help="Target folder (use '' for Inbox)")
    move_parser.add_argument("-s", "--target-status", choices=["cur", "new", "tmp"], default="cur", help="Target status folder")
    move_parser.add_argument("--flags", help="New flags for the memory")
    move_parser.set_defaults(func=move_memory)
    
    # Update flags command
    flag_parser = subparsers.add_parser("flag", help="Update memory flags")
    flag_parser.add_argument("id", help="Memory ID or filename")
    flag_parser.add_argument("-f", "--folder", help="Folder (default: Inbox)")
    flag_parser.add_argument("flags", help="New flags for the memory")
    flag_parser.set_defaults(func=update_flags)
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Move a memory to trash")
    delete_parser.add_argument("id", help="Memory ID or filename")
    delete_parser.add_argument("-f", "--folder", help="Folder (default: all folders)")
    delete_parser.set_defaults(func=delete_memory)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search memories")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-f", "--folder", help="Folder to search (default: all folders)")
    search_parser.add_argument("-s", "--status", choices=["cur", "new", "tmp"], help="Status folder (default: all statuses)")
    search_parser.add_argument("--limit", help="Maximum number of results")
    search_parser.add_argument("--offset", help="Offset for pagination")
    search_parser.add_argument("-c", "--with-content", action="store_true", help="Include content in results")
    search_parser.add_argument("--debug", action="store_true", help="Show debug information")
    search_parser.set_defaults(func=search_memories)
    
    # Folder commands
    folders_parser = subparsers.add_parser("folders", help="List all folders")
    folders_parser.set_defaults(func=list_folders)
    
    mkdir_parser = subparsers.add_parser("mkdir", help="Create a new folder")
    mkdir_parser.add_argument("folder", help="Folder name")
    mkdir_parser.set_defaults(func=create_folder)
    
    rmdir_parser = subparsers.add_parser("rmdir", help="Delete a folder")
    rmdir_parser.add_argument("folder", help="Folder name")
    rmdir_parser.set_defaults(func=delete_folder)
    
    rename_parser = subparsers.add_parser("rename", help="Rename a folder")
    rename_parser.add_argument("folder", help="Original folder name")
    rename_parser.add_argument("new_name", help="New folder name")
    rename_parser.set_defaults(func=rename_folder)
    
    # Run filters command
    filters_parser = subparsers.add_parser("run-filters", help="Run memory filters")
    filters_parser.add_argument("--dry-run", action="store_true", help="Simulate actions without applying them")
    filters_parser.set_defaults(func=run_filters)
    
    # Parse args
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Check server health
    if not check_server_health():
        print(f"Error: Cannot connect to Memdir server at {SERVER_URL}")
        print("Make sure the server is running and the URL is correct.")
        sys.exit(1)
    
    # Execute command
    args.func(args)

if __name__ == "__main__":
    main()