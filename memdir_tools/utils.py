"""
Utility functions for Memdir memory management
"""

import os
import time
import socket
import uuid
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Standard folders
STANDARD_FOLDERS = ["cur", "new", "tmp"]

# Special folders
SPECIAL_FOLDERS = [".Trash", ".ToDoLater", ".Projects", ".Archive"]

# Flag definitions
FLAGS = {
    "S": "Seen",
    "R": "Replied",
    "F": "Flagged",
    "P": "Priority"
}

# Function to get the base directory dynamically (used by callers like server.py)
def get_memdir_base_path_from_config() -> str:
    """Gets the Memdir base path from config or env var."""
    # Import flask locally to avoid potential circular dependencies at module level
    try:
        from flask import current_app
    except ImportError:
        current_app = None # Handle cases where Flask might not be available

    default_path = os.path.join(os.getcwd(), "Memdir")
    base_path = default_path # Start with default

    if current_app and 'MEMDIR_DATA_DIR' in current_app.config:
        # Prioritize Flask app config if available and key exists
        base_path = current_app.config['MEMDIR_DATA_DIR']
        # print(f"DEBUG utils: Using base_path from Flask config: {base_path}")
    else:
        # Fallback to environment variable
        env_path = os.environ.get("MEMDIR_DATA_DIR")
        if env_path:
            base_path = env_path
            # print(f"DEBUG utils: Using base_path from ENV var: {base_path}")
        # else:
            # print(f"DEBUG utils: Using default base_path: {base_path}")

    return base_path

# Define MEMDIR_BASE using the function for module-level use if needed elsewhere,
# but functions below should ideally call the function directly or receive base_dir.
MEMDIR_BASE = get_memdir_base_path_from_config()


def ensure_memdir_structure() -> None:
    """Ensure that the base Memdir structure exists"""
    base_dir = get_memdir_base_path_from_config() # Get current base path
    # Create base directories
    for folder in STANDARD_FOLDERS:
        os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

    # Create special folders
    for special in SPECIAL_FOLDERS:
        for folder in STANDARD_FOLDERS:
            os.makedirs(os.path.join(base_dir, special, folder), exist_ok=True)

def get_memdir_folders() -> List[str]:
    """Get list of all memdir folders"""
    base_dir = get_memdir_base_path_from_config() # Get current base path
    folders = []

    # Walk through the memdir structure
    if not os.path.isdir(base_dir):
        return []
    for root, dirs, _ in os.walk(base_dir):
        is_maildir_folder = all(sd in dirs for sd in STANDARD_FOLDERS)
        if is_maildir_folder:
            rel_path = os.path.relpath(root, base_dir)
            if rel_path == ".":
                folders.append("")
            else:
                folders.append(rel_path.replace(os.path.sep, "/"))
            dirs[:] = [d for d in dirs if d not in STANDARD_FOLDERS]

    if "" not in folders and os.path.isdir(os.path.join(base_dir, "cur")):
         folders.insert(0, "")

    return sorted(list(set(folders)))

def generate_memory_filename(flags: str = "") -> str:
    """
    Generate a memory filename in Maildir format
    
    Format: timestamp.unique_id.hostname:2,flags
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    hostname = socket.gethostname()
    valid_flags = ''.join(sorted(list(set(f for f in flags if f in FLAGS))))
    return f"{timestamp}.{unique_id}.{hostname}:2,{valid_flags}"

def parse_memory_filename(filename: str) -> Dict[str, Any]:
    """
    Parse a memory filename and extract its components
    
    Returns:
        Dict with timestamp, unique_id, hostname, and flags
    """
    pattern = r"(\d+)\.([a-f0-9]+)\.([^:]+):2(?:,([A-Z]*))?"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Invalid memory filename format: {filename}")
    timestamp_str, unique_id, hostname, flags_str = match.groups()
    flags = list(flags_str) if flags_str is not None else []
    timestamp = int(timestamp_str)
    return {
        "timestamp": timestamp, "unique_id": unique_id, "hostname": hostname,
        "flags": flags, "date": datetime.fromtimestamp(timestamp)
    }

def parse_memory_content(content: str) -> Tuple[Dict[str, str], str]:
    """
    Parse memory content into headers and body
    
    Returns:
        Tuple of (headers dict, body string)
    """
    parts = re.split(r"^\-\-\-.*?\n", content, 1, re.MULTILINE)
    if len(parts) < 2: return {}, content.strip()
    header_text, body = parts
    headers = {}
    for line in header_text.strip().split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()
    return headers, body.strip()

def create_memory_content(headers: Dict[str, str], body: str) -> str:
    """
    Create memory content from headers and body
    
    Returns:
        Formatted memory content
    """
    header_lines = [f"{key}: {value}" for key, value in headers.items()]
    header_text = "\n".join(header_lines)
    separator = "\n---\n" if header_text else "---\n"
    return f"{header_text}{separator}{body}"

def get_memory_path(folder: str, status: str = "new") -> str:
    """
    Get the path to a memory folder status directory (cur, new, tmp).
    DEPRECATED: Use version accepting base_dir. Retained for potential compatibility.
    """
    base_dir = get_memdir_base_path_from_config() # Uses dynamic lookup
    if status not in STANDARD_FOLDERS:
        raise ValueError(f"Invalid status: {status}. Must be one of {STANDARD_FOLDERS}")
    folder = folder.replace("\\", "/").strip("/")
    full_path = os.path.join(base_dir, folder, status) if folder else os.path.join(base_dir, status)
    return full_path

# Keep the refactored versions accepting base_dir alongside for now
def get_memory_path_explicit(base_dir: str, folder: str, status: str = "new") -> str:
    """Get the path to a memory folder status directory (cur, new, tmp)."""
    if status not in STANDARD_FOLDERS:
        raise ValueError(f"Invalid status: {status}. Must be one of {STANDARD_FOLDERS}")
    folder = folder.replace("\\", "/").strip("/")
    full_path = os.path.join(base_dir, folder, status) if folder else os.path.join(base_dir, status)
    return full_path


def save_memory(folder: str,
                content: str, 
                headers: Dict[str, str] = None, 
                flags: str = "") -> str:
    """
    Save a memory to the specified folder. Uses dynamically determined base path.
    
    Args:
        folder: The relative memory folder path (e.g., "", ".Projects/Work").
        content: The memory content (body).
        headers: Optional headers for the memory.
        flags: Optional flags for the memory.
        
    Returns:
        The filename of the saved memory.
    """
    base_dir = get_memdir_base_path_from_config() # Get current base path
    ensure_memdir_structure(base_dir)
    folder = folder.replace("\\", "/").strip("/")
    tmp_folder_path = get_memory_path_explicit(base_dir, folder, "tmp")
    new_folder_path = get_memory_path_explicit(base_dir, folder, "new")
    os.makedirs(tmp_folder_path, exist_ok=True)
    os.makedirs(new_folder_path, exist_ok=True)
    
    filename = generate_memory_filename(flags)
    if headers is None: headers = {}
    if "Date" not in headers: headers["Date"] = datetime.now().isoformat()
    if "Subject" not in headers: headers["Subject"] = f"Memory {filename.split('.')[1]}"
    full_content = create_memory_content(headers, content)
    
    tmp_path = os.path.join(tmp_folder_path, filename)
    try:
        with open(tmp_path, "w", encoding='utf-8') as f: f.write(full_content)
    except Exception as e:
        raise IOError(f"Failed to write to temporary file {tmp_path}: {e}") from e

    new_path = os.path.join(new_folder_path, filename)
    try:
        os.rename(tmp_path, new_path)
    except Exception as e:
        if os.path.exists(tmp_path): os.remove(tmp_path)
        raise IOError(f"Failed to move memory from tmp to new ({tmp_path} -> {new_path}): {e}") from e
        
    return filename

def list_memories(folder: str, status: str = "cur", include_content: bool = False) -> List[Dict[str, Any]]:
    """
    List memories in the specified folder and status. Uses dynamically determined base path.
    
    Args:
        folder: The relative memory folder path.
        status: The status folder ("new", "cur", "tmp").
        include_content: Whether to include the memory content.
        
    Returns:
        List of memory info dictionaries.
    """
    base_dir = get_memdir_base_path_from_config() # Get current base path
    memories = []
    folder_path = get_memory_path_explicit(base_dir, folder, status)
    
    if not os.path.isdir(folder_path): return []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path): continue
        try:
            file_info = parse_memory_filename(filename)
            with open(file_path, "r", encoding='utf-8') as f: content = f.read()
            headers, body = parse_memory_content(content)
            memory_info = {
                "filename": filename, "folder": folder, "status": status,
                "headers": headers, "metadata": file_info
            }
            if include_content: memory_info["content"] = body
            memories.append(memory_info)
        except ValueError: pass
        except Exception as e: print(f"Error processing file {filename} in {folder_path}: {e}")
    
    memories.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
    return memories

def move_memory(filename: str, 
                source_folder: str, 
                target_folder: str, 
                source_status: str = "new", 
                target_status: str = "cur",
                new_flags: Optional[str] = None) -> bool:
    """
    Move a memory from one folder/status to another. Uses dynamically determined base path.
    
    Args:
        filename: The current memory filename.
        source_folder: The source relative memory folder path.
        target_folder: The target relative memory folder path.
        source_status: The source status folder ("new", "cur", "tmp").
        target_status: The target status folder ("new", "cur", "tmp").
        new_flags: Optional new flags. If provided, the filename will be updated.
        
    Returns:
        True if successful, False otherwise.
    """
    base_dir = get_memdir_base_path_from_config() # Get current base path
    source_filename = filename
    source_path = os.path.join(get_memory_path_explicit(base_dir, source_folder, source_status), source_filename)
    
    if not os.path.isfile(source_path): return False
    
    target_filename = source_filename
    if new_flags is not None:
        try:
            valid_new_flags = ''.join(sorted(list(set(f for f in new_flags if f in FLAGS))))
            parts = source_filename.split(":2,")
            target_filename = f"{parts[0]}:2,{valid_new_flags}" if len(parts) == 2 else f"{source_filename.split(':')[0]}:2,{valid_new_flags}"
        except Exception as e:
             print(f"Error parsing/generating filename with new flags: {e}")
             return False

    target_dir = get_memory_path_explicit(base_dir, target_folder, target_status)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, target_filename)
    
    try:
        os.rename(source_path, target_path)
        return True
    except Exception as e:
        print(f"Error moving file {source_path} to {target_path}: {e}")
        return False

def search_memories(query: str, 
                   folders: List[str] = None, 
                   statuses: List[str] = None,
                   headers_only: bool = False) -> List[Dict[str, Any]]:
    """
    Search memories for a query string (simple substring search). Uses dynamically determined base path.
    
    Args:
        query: The search query string.
        folders: List of relative folder paths to search (None = all folders).
        statuses: List of statuses ("new", "cur", "tmp") to search (None = all statuses).
        headers_only: Whether to search only in headers.
        
    Returns:
        List of matching memory info dictionaries.
    """
    base_dir = get_memdir_base_path_from_config() # Get current base path
    results = []
    query_lower = query.lower()
    
    folders_to_search = get_memdir_folders(base_dir) if folders is None else [f for f in folders if f in get_memdir_folders(base_dir)]
    statuses_to_search = statuses if statuses is not None else STANDARD_FOLDERS
    
    for folder in folders_to_search:
        for status in statuses_to_search:
            memories_in_status = list_memories(folder, status, include_content=not headers_only) # Uses dynamic path internally
            
            for memory in memories_in_status:
                found = False
                for key, value in memory["headers"].items():
                    if query_lower in str(value).lower(): found = True; break
                if not found and not headers_only and "content" in memory:
                    if query_lower in memory["content"].lower(): found = True
                if found:
                    if "content" in memory and not headers_only:
                         memory["content_preview"] = memory["content"][:100] + ("..." if len(memory["content"]) > 100 else "")
                         del memory["content"]
                    results.append(memory)
    return results

def update_memory_flags(filename: str, 
                       folder: str, 
                       status: str, 
                       flags: str) -> bool:
    """
    Update the flags of a memory by renaming the file. Uses dynamically determined base path.
    
    Args:
        filename: The current memory filename.
        folder: The relative memory folder path.
        status: The current status folder ("new", "cur", "tmp").
        flags: The new flags string (e.g., "SP").
        
    Returns:
        True if successful, False otherwise.
    """
    # Use move_memory to handle the rename, moving within the same folder/status
    # move_memory internally calls get_memdir_base_path_from_config
    return move_memory(
        filename=filename,
        source_folder=folder,
        target_folder=folder,
        source_status=status,
        target_status=status,
        new_flags=flags
    )
