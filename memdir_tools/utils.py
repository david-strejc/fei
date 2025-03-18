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

# Base directory for all memories
MEMDIR_BASE = os.path.join(os.getcwd(), "Memdir")

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

def ensure_memdir_structure() -> None:
    """Ensure that the base Memdir structure exists"""
    # Create base directories
    for folder in STANDARD_FOLDERS:
        os.makedirs(os.path.join(MEMDIR_BASE, folder), exist_ok=True)
    
    # Create special folders
    for special in SPECIAL_FOLDERS:
        for folder in STANDARD_FOLDERS:
            os.makedirs(os.path.join(MEMDIR_BASE, special, folder), exist_ok=True)

def get_memdir_folders() -> List[str]:
    """Get list of all memdir folders"""
    folders = []
    
    # Walk through the memdir structure
    for root, dirs, _ in os.walk(MEMDIR_BASE):
        if any(folder in dirs for folder in STANDARD_FOLDERS):
            # This is a valid memdir folder
            rel_path = os.path.relpath(root, MEMDIR_BASE)
            if rel_path == ".":
                folders.append("")  # Root folder
            else:
                folders.append(rel_path)
    
    return folders

def generate_memory_filename(flags: str = "") -> str:
    """
    Generate a memory filename in Maildir format
    
    Format: timestamp.unique_id.hostname:2,flags
    """
    timestamp = int(time.time())
    unique_id = uuid.uuid4().hex[:8]
    hostname = socket.gethostname()
    
    # Validate flags - only allow known flags
    valid_flags = ''.join([f for f in flags if f in FLAGS])
    
    return f"{timestamp}.{unique_id}.{hostname}:2,{valid_flags}"

def parse_memory_filename(filename: str) -> Dict[str, Any]:
    """
    Parse a memory filename and extract its components
    
    Returns:
        Dict with timestamp, unique_id, hostname, and flags
    """
    pattern = r"(\d+)\.([a-z0-9]+)\.([^:]+):2,([A-Z]*)"
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Invalid memory filename: {filename}")
    
    timestamp, unique_id, hostname, flags = match.groups()
    
    return {
        "timestamp": int(timestamp),
        "unique_id": unique_id,
        "hostname": hostname,
        "flags": list(flags),
        "date": datetime.fromtimestamp(int(timestamp))
    }

def parse_memory_content(content: str) -> Tuple[Dict[str, str], str]:
    """
    Parse memory content into headers and body
    
    Returns:
        Tuple of (headers dict, body string)
    """
    # Split on the first '---' line
    parts = content.split("---", 1)
    
    if len(parts) < 2:
        # No proper separator, assume it's all body
        return {}, content.strip()
    
    header_text, body = parts
    
    # Parse headers
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
    
    return f"{header_text}\n---\n{body}"

def get_memory_path(folder: str, status: str = "new") -> str:
    """
    Get the path to a memory folder
    
    Args:
        folder: The memory folder (e.g., "", ".Projects", ".Archive/2023")
        status: The status folder ("new", "cur", or "tmp")
        
    Returns:
        Full path to the memory folder
    """
    if status not in STANDARD_FOLDERS:
        raise ValueError(f"Invalid status: {status}. Must be one of {STANDARD_FOLDERS}")
    
    if folder:
        return os.path.join(MEMDIR_BASE, folder, status)
    else:
        return os.path.join(MEMDIR_BASE, status)

def save_memory(folder: str, 
                content: str, 
                headers: Dict[str, str] = None, 
                flags: str = "") -> str:
    """
    Save a memory to the specified folder
    
    Args:
        folder: The memory folder (e.g., "", ".Projects", ".Archive/2023")
        content: The memory content (body)
        headers: Optional headers for the memory
        flags: Optional flags for the memory
        
    Returns:
        The filename of the saved memory
    """
    # Ensure the memdir structure exists
    ensure_memdir_structure()
    
    # Make sure the target folder exists
    folder_path = get_memory_path(folder, "tmp")
    os.makedirs(folder_path, exist_ok=True)
    
    # Generate filename
    filename = generate_memory_filename(flags)
    
    # Prepare content
    if headers is None:
        headers = {}
    
    # Add default headers if not present
    if "Date" not in headers:
        headers["Date"] = datetime.now().isoformat()
    if "Subject" not in headers:
        headers["Subject"] = f"Memory {filename.split('.')[1]}"
    
    full_content = create_memory_content(headers, content)
    
    # Save to tmp folder
    tmp_path = os.path.join(folder_path, filename)
    with open(tmp_path, "w") as f:
        f.write(full_content)
    
    # Move to new folder (atomic operation)
    new_path = os.path.join(get_memory_path(folder, "new"), filename)
    os.rename(tmp_path, new_path)
    
    return filename

def list_memories(folder: str, status: str = "cur", include_content: bool = False) -> List[Dict[str, Any]]:
    """
    List memories in the specified folder
    
    Args:
        folder: The memory folder (e.g., "", ".Projects", ".Archive/2023")
        status: The status folder ("new", "cur", "tmp")
        include_content: Whether to include the memory content
        
    Returns:
        List of memory info dictionaries
    """
    memories = []
    folder_path = get_memory_path(folder, status)
    
    if not os.path.exists(folder_path):
        return []
    
    for filename in os.listdir(folder_path):
        try:
            # Skip files that don't match the memory filename pattern
            if not re.match(r"\d+\.[a-z0-9]+\.[^:]+:2,[A-Z]*", filename):
                continue
                
            file_info = parse_memory_filename(filename)
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, "r") as f:
                content = f.read()
            
            headers, body = parse_memory_content(content)
            
            memory_info = {
                "filename": filename,
                "folder": folder,
                "status": status,
                "headers": headers,
                "metadata": file_info
            }
            
            if include_content:
                memory_info["content"] = body
                
            memories.append(memory_info)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    # Sort by timestamp (newest first)
    memories.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
    
    return memories

def move_memory(filename: str, 
                source_folder: str, 
                target_folder: str, 
                source_status: str = "new", 
                target_status: str = "cur",
                new_flags: Optional[str] = None) -> bool:
    """
    Move a memory from one folder to another
    
    Args:
        filename: The memory filename
        source_folder: The source memory folder
        target_folder: The target memory folder
        source_status: The source status folder
        target_status: The target status folder
        new_flags: Optional new flags for the memory
        
    Returns:
        True if successful, False otherwise
    """
    source_path = os.path.join(get_memory_path(source_folder, source_status), filename)
    
    if not os.path.exists(source_path):
        return False
    
    # Parse existing flags
    if new_flags is not None:
        file_info = parse_memory_filename(filename)
        old_flags = ''.join(file_info["flags"])
        
        # Generate new filename with updated flags
        parts = filename.split(":2,")
        if len(parts) == 2:
            filename = f"{parts[0]}:2,{new_flags}"
    
    # Ensure target folder exists
    target_path = os.path.join(get_memory_path(target_folder, target_status), filename)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    # Move the file
    os.rename(source_path, target_path)
    
    return True

def search_memories(query: str, 
                   folders: List[str] = None, 
                   statuses: List[str] = None,
                   headers_only: bool = False) -> List[Dict[str, Any]]:
    """
    Search memories for a query string
    
    Args:
        query: The search query
        folders: List of folders to search (None = all folders)
        statuses: List of statuses to search (None = all statuses)
        headers_only: Whether to search only in headers
        
    Returns:
        List of matching memory info dictionaries
    """
    results = []
    
    # Default to all folders if not specified
    if folders is None:
        folders = get_memdir_folders()
    
    # Default to all standard folders if not specified
    if statuses is None:
        statuses = STANDARD_FOLDERS
    
    # Search in each folder and status
    for folder in folders:
        for status in statuses:
            memories = list_memories(folder, status, include_content=True)
            
            for memory in memories:
                found = False
                
                # Search in headers
                for key, value in memory["headers"].items():
                    if query.lower() in value.lower():
                        found = True
                        break
                
                # Search in content if not headers_only
                if not found and not headers_only and "content" in memory:
                    if query.lower() in memory["content"].lower():
                        found = True
                
                if found:
                    # Remove content to save memory
                    if "content" in memory and not headers_only:
                        memory["content_preview"] = memory["content"][:100] + "..." if len(memory["content"]) > 100 else memory["content"]
                        del memory["content"]
                    
                    results.append(memory)
    
    return results

def update_memory_flags(filename: str, folder: str, status: str, flags: str) -> bool:
    """
    Update the flags of a memory
    
    Args:
        filename: The memory filename
        folder: The memory folder
        status: The status folder
        flags: The new flags
        
    Returns:
        True if successful, False otherwise
    """
    # Parse the original filename to get components
    try:
        file_info = parse_memory_filename(filename)
    except ValueError:
        return False
        
    # Generate new filename with updated flags
    parts = filename.split(":2,")
    if len(parts) != 2:
        return False
        
    new_filename = f"{parts[0]}:2,{flags}"
    
    # Rename the file (equivalent to updating flags)
    return move_memory(
        filename,
        folder,
        folder,
        status,
        status,
        flags
    )