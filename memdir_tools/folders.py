#!/usr/bin/env python3
"""
Memory folder management and manipulation tools

This module provides tools for:
1. Creating and organizing folder hierarchies
2. Moving and renaming folders
3. Folder statistics and summarization
4. Bulk operations on folder contents
"""

import os
import re
import shutil
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

# Import specific utils needed, including the config reader
from memdir_tools.utils import (
    ensure_memdir_structure,
    get_memdir_folders,
    # get_memdir_base_path_from_config, # REMOVED: Function was removed from utils.py
    list_memories, # Refactored to accept base_dir
    move_memory, # Refactored to accept base_dir
    STANDARD_FOLDERS,
    SPECIAL_FOLDERS,
    FLAGS
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

class MemdirFolderManager:
    """
    Memory folder management system
    """
    
    def __init__(self, base_dir: str):
        """Initialize folder manager"""
        # Base directory is now passed explicitly
        if not base_dir or not os.path.isdir(base_dir):
             # Handle cases where base_dir might not be valid yet,
             # although server.py should provide a valid one.
             # Maybe raise an error or log a warning? For now, just store it.
             # Consider if ensure_memdir_structure should be called here or later.
             pass # Or raise ValueError("Invalid base_dir provided to MemdirFolderManager")

        self.base_dir = base_dir
        # Ensure structure based on the provided base_dir
        ensure_memdir_structure(self.base_dir)
    
    def create_folder(self, folder_path: str) -> bool:
        """
        Create a new memory folder with proper structure
        
        Args:
            folder_path: Path to the folder (e.g., ".Projects/Work/ClientA")
            
        Returns:
            True if created successfully, False if already exists
        """
        # Clean up folder path - ensure it starts with "." if it's a special folder
        if not folder_path.startswith(".") and folder_path not in ("", "/"):
            folder_path = f".{folder_path}"
            
        # Replace backslashes with forward slashes
        folder_path = folder_path.replace("\\", "/")
        
        # Remove leading/trailing slashes
        folder_path = folder_path.strip("/")
        
        # Full path to the folder using the potentially updated self.base_dir
        full_path = os.path.join(self.base_dir, folder_path)

        # Check if folder already exists
        if os.path.exists(full_path):
            return False
            
        # Create the folder structure
        for folder in STANDARD_FOLDERS:
            os.makedirs(os.path.join(full_path, folder), exist_ok=True)
            
        return True
        
    def rename_folder(self, old_path: str, new_path: str) -> bool:
        """
        Rename a memory folder
        
        Args:
            old_path: Current path (e.g., ".Projects/OldName")
            new_path: New path (e.g., ".Projects/NewName")
            
        Returns:
            True if renamed successfully, False otherwise
        """
        # Clean up paths
        old_path = old_path.replace("\\", "/").strip("/")
        new_path = new_path.replace("\\", "/").strip("/")
        
        # Full paths using the potentially updated self.base_dir
        old_full_path = os.path.join(self.base_dir, old_path)
        new_full_path = os.path.join(self.base_dir, new_path)

        # Check if source folder exists and destination doesn't
        if not os.path.exists(old_full_path):
            return False
        
        if os.path.exists(new_full_path):
            return False
            
        # Create parent directories for destination if needed
        os.makedirs(os.path.dirname(new_full_path), exist_ok=True)
        
        # Rename the folder
        try:
            shutil.move(old_full_path, new_full_path)
            return True
        except Exception:
            return False
            
    def delete_folder(self, folder_path: str, force: bool = False) -> Tuple[bool, str]:
        """
        Delete a memory folder
        
        Args:
            folder_path: Path to the folder (e.g., ".Projects/Work/ClientA")
            force: Force deletion even if folder contains memories
            
        Returns:
            Tuple of (success, message)
        """
        # Clean up folder path
        folder_path = folder_path.replace("\\", "/").strip("/")
        
        # Full path to the folder using the potentially updated self.base_dir
        full_path = os.path.join(self.base_dir, folder_path)

        # Check if folder exists
        if not os.path.exists(full_path):
            return (False, f"Folder does not exist: {folder_path}")
            
        # Check if folder is a special folder
        if folder_path in SPECIAL_FOLDERS or folder_path == "":
            return (False, f"Cannot delete special folder: {folder_path}")
            
        # Check if folder contains memories
        memories = []
        for status in STANDARD_FOLDERS:
            # Pass self.base_dir to list_memories
            memories.extend(list_memories(self.base_dir, folder_path, status))

        if memories and not force:
            return (False, f"Folder contains {len(memories)} memories. Use force=True to delete anyway.")
            
        # Move memories to trash if force deleting non-empty folder
        if memories and force:
            for memory in memories:
                # Pass self.base_dir to move_memory
                move_memory(
                    self.base_dir,
                    memory["filename"],
                    folder_path,
                    ".Trash",
                    memory["status"],
                    "cur"
                )
                
        # Delete the folder
        try:
            shutil.rmtree(full_path)
            return (True, f"Folder deleted: {folder_path}")
        except Exception as e:
            return (False, f"Error deleting folder: {str(e)}")
            
    def move_folder(self, source_path: str, target_path: str) -> Tuple[bool, str]:
        """
        Move a memory folder to another location
        
        Args:
            source_path: Current path (e.g., ".Projects/Work/ClientA")
            target_path: Target path (e.g., ".Archive/Clients/ClientA")
            
        Returns:
            Tuple of (success, message)
        """
        # Clean up paths
        source_path = source_path.replace("\\", "/").strip("/")
        target_path = target_path.replace("\\", "/").strip("/")
        
        # Full paths using the potentially updated self.base_dir
        source_full_path = os.path.join(self.base_dir, source_path)
        target_full_path = os.path.join(self.base_dir, target_path)

        # Check if source folder exists
        if not os.path.exists(source_full_path):
            return (False, f"Source folder does not exist: {source_path}")
            
        # Check if target folder already exists
        if os.path.exists(target_full_path):
            return (False, f"Target folder already exists: {target_path}")
            
        # Check if source is a special folder
        if source_path in SPECIAL_FOLDERS or source_path == "":
            return (False, f"Cannot move special folder: {source_path}")
            
        # Create parent directories for target if needed
        os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
        
        # Move the folder
        try:
            shutil.move(source_full_path, target_full_path)
            return (True, f"Folder moved: {source_path} -> {target_path}")
        except Exception as e:
            return (False, f"Error moving folder: {str(e)}")
            
    def get_folder_stats(self, folder_path: str, include_subfolders: bool = False) -> Dict[str, Any]:
        """
        Get statistics for a memory folder
        
        Args:
            folder_path: Path to the folder (e.g., ".Projects/Work")
            include_subfolders: Whether to include statistics for subfolders
            
        Returns:
            Dictionary with folder statistics
        """
        # Clean up folder path
        folder_path = folder_path.replace("\\", "/").strip("/")
        
        # Get all folders using the potentially updated self.base_dir
        all_folders = get_memdir_folders(self.base_dir) # Pass self.base_dir

        # Filter subfolders if needed
        folders_to_process = []
        if include_subfolders:
            for folder in all_folders:
                if folder == folder_path or (folder.startswith(folder_path + "/") and folder != folder_path): # Ensure proper subfolder check
                    folders_to_process.append(folder)
        else:
            # Check if the specific folder exists
            target_full_path = os.path.join(self.base_dir, folder_path)
            if os.path.isdir(target_full_path):
                 folders_to_process = [folder_path]
            else:
                 # Handle case where the exact folder doesn't exist but might be root ""
                 if folder_path == "" and "" in all_folders:
                     folders_to_process = [""]
                 else:
                     # Raise error or return empty stats if folder not found
                     raise FileNotFoundError(f"Folder not found: {folder_path}")


        # Initialize statistics
        stats = {
            "folder": folder_path or "Inbox",
            "total_memories": 0,
            "memory_counts": {
                "cur": 0,
                "new": 0,
                "tmp": 0
            },
            "flag_counts": {
                "S": 0,  # Seen
                "R": 0,  # Replied
                "F": 0,  # Flagged
                "P": 0   # Priority
            },
            "tags": {},
            "subfolders": [],
            "newest_memory": None,
            "oldest_memory": None
        }
        
        # Process each folder
        for folder in folders_to_process:
            subfolder_stats = {
                "folder": folder or "Inbox",
                "memory_counts": {
                    "cur": 0,
                    "new": 0,
                    "tmp": 0
                },
                "total_memories": 0
            }
            
            # Get memories for each status, passing self.base_dir
            for status in STANDARD_FOLDERS:
                memories = list_memories(self.base_dir, folder, status) # Pass self.base_dir
                subfolder_stats["memory_counts"][status] = len(memories)
                subfolder_stats["total_memories"] += len(memories)
                stats["total_memories"] += len(memories)
                stats["memory_counts"][status] += len(memories)
                
                # Process each memory
                for memory in memories:
                    # Update flag counts
                    for flag in memory["metadata"]["flags"]:
                        if flag in stats["flag_counts"]:
                            stats["flag_counts"][flag] += 1
                            
                    # Update tag counts
                    if "Tags" in memory["headers"]:
                        tags = [tag.strip() for tag in memory["headers"]["Tags"].split(",")]
                        for tag in tags:
                            if tag: # Avoid counting empty tags
                                stats["tags"][tag] = stats["tags"].get(tag, 0) + 1
                            
                    # Update newest/oldest memory
                    memory_date = memory["metadata"]["date"]
                    
                    if stats["newest_memory"] is None or memory_date > stats["newest_memory"]["date"]:
                        stats["newest_memory"] = {
                            "id": memory["metadata"]["unique_id"],
                            "subject": memory["headers"].get("Subject", "No subject"),
                            "date": memory_date
                        }
                        
                    if stats["oldest_memory"] is None or memory_date < stats["oldest_memory"]["date"]:
                        stats["oldest_memory"] = {
                            "id": memory["metadata"]["unique_id"],
                            "subject": memory["headers"].get("Subject", "No subject"),
                            "date": memory_date
                        }
            
            # Add subfolder stats if needed
            if include_subfolders and folder != folder_path:
                stats["subfolders"].append(subfolder_stats)
                
        return stats
        
    def list_folders(self, parent_folder: str = "", recursive: bool = False) -> List[Dict[str, Any]]:
        """
        List all memory folders
        
        Args:
            parent_folder: Filter by parent folder
            recursive: Whether to list recursively
            
        Returns:
            List of folder information dictionaries
        """
        # Clean up parent folder path
        parent_folder = parent_folder.replace("\\", "/").strip("/")
        
        # Get all folders using stored base_dir
        all_folders = get_memdir_folders(self.base_dir) # Pass self.base_dir

        # Filter by parent folder if specified
        if parent_folder:
            filtered_folders = []
            for folder in all_folders:
                # Match direct children or all descendants based on recursive flag
                if recursive:
                     # Match if folder starts with parent_folder/ or is exactly parent_folder (if parent is not root)
                     if folder.startswith(parent_folder + "/") or (folder == parent_folder and parent_folder != ""):
                         # Exclude the parent itself unless it's the root being listed recursively
                         if folder != parent_folder or parent_folder == "":
                             filtered_folders.append(folder)
                else:
                    # Direct children only - one level deeper
                    if folder.startswith(parent_folder + "/") and folder.count("/") == parent_folder.count("/") + (1 if parent_folder else 0) :
                         filtered_folders.append(folder)
                    # Handle direct children of root ""
                    elif parent_folder == "" and "/" not in folder and folder != "":
                         filtered_folders.append(folder)

            folders = filtered_folders
        else: # No parent specified
            if recursive:
                folders = all_folders # List all including root ""
            else:
                # Only top-level folders (no slashes) + root ""
                folders = [""] + [f for f in all_folders if "/" not in f and f]

        # Get information for each folder
        folder_info = []
        for folder in folders:
            try: # Add try-except for get_folder_stats
                stats = self.get_folder_stats(folder) # Uses self.base_dir implicitly via list_memories

                # Create folder entry
                entry = {
                    "path": folder or "Inbox",
                    "name": os.path.basename(folder) or "Inbox",
                    "is_special": folder in SPECIAL_FOLDERS,
                    "memory_counts": stats["memory_counts"],
                    "total_memories": stats["total_memories"]
                }
                folder_info.append(entry)
            except FileNotFoundError:
                 print(f"Warning: Folder '{folder}' not found during list_folders stats gathering.")
                 continue # Skip folders that might have been deleted concurrently


        # Sort by special folders first, then name
        folder_info.sort(key=lambda x: (0 if x["is_special"] else 1, x["name"]))
        
        return folder_info
        
    def make_symlinks(self, folder_path: str, symlink_root: str) -> Tuple[bool, str]:
        """
        Create symlinks to a memory folder for external tools
        
        Args:
            folder_path: Path to the memory folder
            symlink_root: Root directory for symlinks
            
        Returns:
            Tuple of (success, message)
        """
        # Clean up folder path
        folder_path = folder_path.replace("\\", "/").strip("/")
        
        # Full paths using stored base_dir
        folder_full_path = os.path.join(self.base_dir, folder_path)
        symlink_full_path = os.path.join(symlink_root, folder_path)

        # Check if folder exists
        if not os.path.exists(folder_full_path):
            return (False, f"Folder does not exist: {folder_path}")
            
        # Create parent directories for symlink if needed
        os.makedirs(os.path.dirname(symlink_full_path), exist_ok=True)
        
        # Create symlinks for each standard folder
        success = True
        for folder in STANDARD_FOLDERS:
            source = os.path.join(folder_full_path, folder)
            target = os.path.join(symlink_full_path, folder)
            
            # Create symlink
            try:
                if os.path.exists(target):
                    if os.path.islink(target):
                        os.unlink(target)
                    else:
                        return (False, f"Target already exists and is not a symlink: {target}")
                        
                os.symlink(source, target, target_is_directory=True)
            except Exception as e:
                success = False
                return (False, f"Error creating symlink: {str(e)}")
                
        return (True, f"Symlinks created in {symlink_full_path}")
        
    def copy_folder(self, source_path: str, target_path: str) -> Tuple[bool, str]:
        """
        Copy a memory folder to another location
        
        Args:
            source_path: Source folder path
            target_path: Target folder path
            
        Returns:
            Tuple of (success, message)
        """
        # Clean up paths
        source_path = source_path.replace("\\", "/").strip("/")
        target_path = target_path.replace("\\", "/").strip("/")
        
        # Full paths using stored base_dir
        source_full_path = os.path.join(self.base_dir, source_path)
        target_full_path = os.path.join(self.base_dir, target_path)

        # Check if source folder exists
        if not os.path.exists(source_full_path):
            return (False, f"Source folder does not exist: {source_path}")
            
        # Check if target folder already exists
        if os.path.exists(target_full_path):
            return (False, f"Target folder already exists: {target_path}")
            
        # Create parent directories for target if needed
        os.makedirs(os.path.dirname(target_full_path), exist_ok=True)
        
        # Copy the folder structure first
        for folder in STANDARD_FOLDERS:
            os.makedirs(os.path.join(target_full_path, folder), exist_ok=True)
            
        # Copy all memory files
        for folder in STANDARD_FOLDERS:
            source_folder = os.path.join(source_full_path, folder)
            target_folder = os.path.join(target_full_path, folder)
            
            # Skip if source doesn't exist
            if not os.path.exists(source_folder):
                continue
                
            # Copy all files
            for filename in os.listdir(source_folder):
                source_file = os.path.join(source_folder, filename)
                target_file = os.path.join(target_folder, filename)
                
                try:
                    shutil.copy2(source_file, target_file)
                except Exception as e:
                    return (False, f"Error copying file {filename}: {str(e)}")
                    
        return (True, f"Folder copied: {source_path} -> {target_path}")

    def bulk_tag_folder(self, folder_path: str, tags: List[str], 
                       statuses: List[str] = None, 
                       operation: str = "add") -> Tuple[int, List[Dict[str, Any]]]:
        """
        Add, remove, or replace tags for all memories in a folder
        
        Args:
            folder_path: Path to the folder
            tags: List of tags
            statuses: List of statuses to process (default: ["cur"])
            operation: Operation to perform ("add", "remove", "replace")
            
        Returns:
            Tuple of (count of affected memories, list of affected memory info)
        """
        # Clean up folder path
        folder_path = folder_path.replace("\\", "/").strip("/")
        
        # Default statuses
        if statuses is None:
            statuses = ["cur"]
            
        # Get all memories, passing base_dir
        memories = []
        for status in statuses:
            memories.extend(list_memories(self.base_dir, folder_path, status))
            
        # Track affected memories
        affected_count = 0
        affected_memories = []
        
        # Process each memory
        for memory in memories:
            file_path = os.path.join(
                self.base_dir, # Use stored base_dir
                folder_path,
                memory["status"],
                memory["filename"]
            )
            
            # Skip if file doesn't exist
            if not os.path.exists(file_path):
                continue
                
            # Get existing tags
            existing_tags = []
            if "Tags" in memory["headers"]:
                existing_tags = [tag.strip() for tag in memory["headers"]["Tags"].split(",") if tag.strip()] # Ensure no empty tags
                
            # Apply operation
            if operation == "add":
                # Add new tags
                updated_tags = existing_tags + [tag for tag in tags if tag not in existing_tags]
            elif operation == "remove":
                # Remove specified tags
                updated_tags = [tag for tag in existing_tags if tag not in tags]
            elif operation == "replace":
                # Replace all tags
                updated_tags = tags
            else:
                continue
                
            # Skip if no changes
            if sorted(existing_tags) == sorted(updated_tags):
                continue
                
            # Format new tags
            new_tags_str = ", ".join(updated_tags)
            
            # Read file content
            with open(file_path, "r") as f:
                content = f.read()
                
            # Update tags
            if "Tags:" in content:
                # Replace existing tags line
                content = re.sub(
                    r"^Tags:.*$", # Match start of line
                    f"Tags: {new_tags_str}",
                    content,
                    count=1, # Replace only the first occurrence
                    flags=re.MULTILINE
                )
            else:
                # Add tags header if not present (insert before ---)
                 parts = content.split("---", 1)
                 if len(parts) == 2:
                     header_part, body_part = parts
                     # Add Tags line, ensuring newline if header_part wasn't empty
                     new_header = header_part.strip() + ("\n" if header_part.strip() else "") + f"Tags: {new_tags_str}"
                     content = f"{new_header}\n---\n{body_part}"
                 else: # Should not happen if parsing worked, but handle defensively
                     content = f"Tags: {new_tags_str}\n---\n{content}"


            # Write back
            with open(file_path, "w") as f:
                f.write(content)
                
            # Track affected memory
            affected_count += 1
            affected_memories.append({
                "id": memory["metadata"]["unique_id"],
                "subject": memory["headers"].get("Subject", "No subject"),
                "old_tags": ", ".join(existing_tags),
                "new_tags": new_tags_str
            })
            
        return (affected_count, affected_memories)


# Command-line interface
def parse_args():
    """Parse command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Folder Management")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # create-folder command
    create_parser = subparsers.add_parser("create-folder", help="Create a new memory folder")
    create_parser.add_argument("folder", help="Folder path (e.g., .Projects/Work)")
    
    # rename-folder command
    rename_parser = subparsers.add_parser("rename-folder", help="Rename a memory folder")
    rename_parser.add_argument("old_path", help="Current folder path")
    rename_parser.add_argument("new_path", help="New folder path")
    
    # delete-folder command
    delete_parser = subparsers.add_parser("delete-folder", help="Delete a memory folder")
    delete_parser.add_argument("folder", help="Folder path")
    delete_parser.add_argument("--force", action="store_true", help="Force deletion even if folder contains memories")
    
    # move-folder command
    move_parser = subparsers.add_parser("move-folder", help="Move a memory folder")
    move_parser.add_argument("source", help="Source folder path")
    move_parser.add_argument("target", help="Target folder path")
    
    # copy-folder command
    copy_parser = subparsers.add_parser("copy-folder", help="Copy a memory folder")
    copy_parser.add_argument("source", help="Source folder path")
    copy_parser.add_argument("target", help="Target folder path")
    
    # list-folders command
    list_parser = subparsers.add_parser("list-folders", help="List memory folders")
    list_parser.add_argument("--parent", help="Filter by parent folder")
    list_parser.add_argument("--recursive", action="store_true", help="List recursively")
    
    # folder-stats command
    stats_parser = subparsers.add_parser("folder-stats", help="Get folder statistics")
    stats_parser.add_argument("folder", help="Folder path")
    stats_parser.add_argument("--include-subfolders", action="store_true", help="Include statistics for subfolders")
    
    # make-symlinks command
    symlinks_parser = subparsers.add_parser("make-symlinks", help="Create symlinks to a memory folder")
    symlinks_parser.add_argument("folder", help="Memory folder path")
    symlinks_parser.add_argument("symlink_root", help="Root directory for symlinks")
    
    # bulk-tag command
    tag_parser = subparsers.add_parser("bulk-tag", help="Add tags to all memories in a folder")
    tag_parser.add_argument("folder", help="Folder path")
    tag_parser.add_argument("--tags", help="Comma-separated list of tags")
    tag_parser.add_argument("--operation", choices=["add", "remove", "replace"], default="add", 
                           help="Operation to perform")
    tag_parser.add_argument("--statuses", help="Comma-separated list of statuses to process (default: cur)")
    
    return parser.parse_args()
    
def main():
    """Main entry point"""
    args = parse_args()
    
    # Create folder manager
    manager = MemdirFolderManager()
    
    # Process commands
    if args.command == "create-folder":
        success = manager.create_folder(args.folder)
        if success:
            print(colorize(f"Created folder: {args.folder}", "green"))
        else:
            print(colorize(f"Folder already exists: {args.folder}", "yellow"))
    
    elif args.command == "rename-folder":
        success = manager.rename_folder(args.old_path, args.new_path)
        if success:
            print(colorize(f"Renamed folder: {args.old_path} -> {args.new_path}", "green"))
        else:
            print(colorize(f"Failed to rename folder", "red"))
            
    elif args.command == "delete-folder":
        success, message = manager.delete_folder(args.folder, args.force)
        if success:
            print(colorize(message, "green"))
        else:
            print(colorize(message, "red"))
            
    elif args.command == "move-folder":
        success, message = manager.move_folder(args.source, args.target)
        if success:
            print(colorize(message, "green"))
        else:
            print(colorize(message, "red"))
            
    elif args.command == "copy-folder":
        success, message = manager.copy_folder(args.source, args.target)
        if success:
            print(colorize(message, "green"))
        else:
            print(colorize(message, "red"))
            
    elif args.command == "list-folders":
        parent = args.parent if args.parent else ""
        folder_list = manager.list_folders(parent, args.recursive)
        
        if not folder_list:
            print(colorize("No folders found", "yellow"))
        else:
            print(colorize(f"Found {len(folder_list)} folders:", "bold"))
            for folder in folder_list:
                # Format memory counts
                counts = f"(cur: {folder['memory_counts']['cur']}, new: {folder['memory_counts']['new']}, tmp: {folder['memory_counts']['tmp']})"
                
                # Display folder info
                special_marker = "*" if folder["is_special"] else " "
                print(f"{special_marker} {folder['path']} - {folder['total_memories']} memories {counts}")
                
    elif args.command == "folder-stats":
        stats = manager.get_folder_stats(args.folder, args.include_subfolders)
        
        print(colorize(f"Statistics for folder: {stats['folder']}", "bold"))
        print(f"Total memories: {stats['total_memories']}")
        print(f"Memory counts: cur: {stats['memory_counts']['cur']}, new: {stats['memory_counts']['new']}, tmp: {stats['memory_counts']['tmp']}")
        
        if stats["flag_counts"]:
            print(colorize("\nFlag counts:", "yellow"))
            for flag, count in stats["flag_counts"].items():
                if count > 0:
                    print(f"  {flag} ({FLAGS[flag] if flag in FLAGS else 'Unknown'}): {count}")
        
        if stats["tags"]:
            print(colorize("\nTop tags:", "yellow"))
            # Sort tags by count (descending)
            sorted_tags = sorted(stats["tags"].items(), key=lambda x: x[1], reverse=True)
            for tag, count in sorted_tags[:10]:  # Show top 10 tags
                print(f"  {tag}: {count}")
                
        if stats["newest_memory"]:
            print(colorize("\nNewest memory:", "yellow"))
            print(f"  {stats['newest_memory']['subject']} ({stats['newest_memory']['id']})")
            print(f"  Date: {stats['newest_memory']['date'].strftime('%Y-%m-%d %H:%M:%S')}")
            
        if stats["oldest_memory"]:
            print(colorize("\nOldest memory:", "yellow"))
            print(f"  {stats['oldest_memory']['subject']} ({stats['oldest_memory']['id']})")
            print(f"  Date: {stats['oldest_memory']['date'].strftime('%Y-%m-%d %H:%M:%S')}")
            
        if args.include_subfolders and stats["subfolders"]:
            print(colorize("\nSubfolders:", "yellow"))
            for subfolder in stats["subfolders"]:
                counts = f"(cur: {subfolder['memory_counts']['cur']}, new: {subfolder['memory_counts']['new']}, tmp: {subfolder['memory_counts']['tmp']})"
                print(f"  {subfolder['folder']} - {subfolder['total_memories']} memories {counts}")
                
    elif args.command == "make-symlinks":
        success, message = manager.make_symlinks(args.folder, args.symlink_root)
        if success:
            print(colorize(message, "green"))
        else:
            print(colorize(message, "red"))
            
    elif args.command == "bulk-tag":
        # Parse tags
        if not args.tags:
            print(colorize("No tags provided", "red"))
            return
            
        tags = [tag.strip() for tag in args.tags.split(",")]
        
        # Parse statuses
        statuses = ["cur"]
        if args.statuses:
            statuses = [status.strip() for status in args.statuses.split(",")]
            
        # Apply tags
        count, affected = manager.bulk_tag_folder(
            args.folder,
            tags,
            statuses,
            args.operation
        )
        
        if count == 0:
            print(colorize("No memories were affected", "yellow"))
        else:
            print(colorize(f"Updated tags for {count} memories", "green"))
            
            if affected:
                print(colorize("\nAffected memories:", "yellow"))
                for memory in affected:
                    print(f"  {memory['subject']} ({memory['id']})")
                    print(f"    Old tags: {memory['old_tags']}")
                    print(f"    New tags: {memory['new_tags']}")
                    

if __name__ == "__main__":
    main()