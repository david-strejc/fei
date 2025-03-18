#!/usr/bin/env python3
"""
HTTP REST API server for Memdir memory management system.
Provides remote access to Memdir functionality with API key authentication.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, Response
from werkzeug.security import safe_str_cmp

from memdir_tools.utils import (
    ensure_memdir_structure,
    get_memdir_folders,
    save_memory,
    list_memories,
    move_memory,
    update_memory_flags,
    STANDARD_FOLDERS,
    FLAGS
)
from memdir_tools.search import (
    SearchQuery,
    search_memories as search_memories_advanced,
    parse_search_args
)
from memdir_tools.folders import (
    create_folder,
    delete_folder,
    rename_folder,
    list_folders,
    get_folder_stats
)
from memdir_tools.filter import run_filters

# Default API key - replace with a secure value in production
DEFAULT_API_KEY = "YOUR_API_KEY_HERE"

# Get the API key from environment variable or use default
API_KEY = os.environ.get("MEMDIR_API_KEY", DEFAULT_API_KEY)

app = Flask(__name__)

# Ensure the memdir structure exists on server startup
ensure_memdir_structure()

def require_api_key(f):
    """Decorator to require API key for all requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key')
        if not provided_key or not safe_str_cmp(provided_key, API_KEY):
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint that doesn't require authentication"""
    return jsonify({"status": "ok", "service": "memdir-api"})

@app.route('/memories', methods=['GET'])
@require_api_key
def list_all_memories():
    """List memories with optional filtering parameters"""
    folder = request.args.get('folder', '')
    status = request.args.get('status', 'cur')
    with_content = request.args.get('with_content', 'false').lower() == 'true'
    
    # Validate status
    if status not in STANDARD_FOLDERS:
        return jsonify({"error": f"Invalid status: {status}. Must be one of {STANDARD_FOLDERS}"}), 400
    
    memories = list_memories(folder, status, include_content=with_content)
    
    return jsonify({
        "count": len(memories),
        "folder": folder or "root",
        "status": status,
        "memories": memories
    })

@app.route('/memories', methods=['POST'])
@require_api_key
def create_memory():
    """Create a new memory"""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract required parameters
    content = data.get('content', '')
    if not content:
        return jsonify({"error": "Content is required"}), 400
    
    # Extract optional parameters
    folder = data.get('folder', '')
    headers = data.get('headers', {})
    flags = data.get('flags', '')
    
    # Create the memory
    try:
        filename = save_memory(folder, content, headers, flags)
        return jsonify({
            "success": True,
            "message": f"Memory created successfully",
            "filename": filename,
            "folder": folder or "root"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create memory: {str(e)}"}), 500

@app.route('/memories/<memory_id>', methods=['GET'])
@require_api_key
def get_memory(memory_id):
    """Retrieve a specific memory by ID"""
    folder = request.args.get('folder', '')
    
    # First try to find by unique ID
    all_memories = []
    for s in STANDARD_FOLDERS:
        all_memories.extend(list_memories(folder, s, include_content=True))
    
    for memory in all_memories:
        if memory_id in (memory["filename"], memory["metadata"]["unique_id"]):
            return jsonify(memory)
    
    return jsonify({"error": f"Memory not found: {memory_id}"}), 404

@app.route('/memories/<memory_id>', methods=['PUT'])
@require_api_key
def update_memory(memory_id):
    """Update a memory's flags or move it to another folder"""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Extract parameters
    source_folder = data.get('source_folder', '')
    target_folder = data.get('target_folder')
    source_status = data.get('source_status')
    target_status = data.get('target_status', 'cur')
    flags = data.get('flags')
    
    # First find the memory
    all_memories = []
    for s in STANDARD_FOLDERS if not source_status else [source_status]:
        all_memories.extend(list_memories(source_folder, s))
    
    for memory in all_memories:
        if memory_id in (memory["filename"], memory["metadata"]["unique_id"]):
            # Found the memory
            if target_folder is not None and target_folder != source_folder:
                # Move the memory to another folder
                result = move_memory(
                    memory["filename"],
                    source_folder,
                    target_folder,
                    memory["status"] if not source_status else source_status,
                    target_status,
                    flags
                )
                
                if result:
                    return jsonify({
                        "success": True,
                        "message": f"Memory moved successfully",
                        "memory_id": memory_id,
                        "source": f"{source_folder or 'root'}/{memory['status']}",
                        "destination": f"{target_folder or 'root'}/{target_status}"
                    })
                else:
                    return jsonify({"error": f"Failed to move memory: {memory_id}"}), 500
            elif flags is not None:
                # Update flags only
                result = update_memory_flags(
                    memory["filename"],
                    source_folder,
                    memory["status"] if not source_status else source_status,
                    flags
                )
                
                if result:
                    return jsonify({
                        "success": True,
                        "message": f"Memory flags updated successfully",
                        "memory_id": memory_id,
                        "new_flags": flags
                    })
                else:
                    return jsonify({"error": f"Failed to update flags for memory: {memory_id}"}), 500
    
    return jsonify({"error": f"Memory not found: {memory_id}"}), 404

@app.route('/memories/<memory_id>', methods=['DELETE'])
@require_api_key
def delete_memory(memory_id):
    """Move a memory to the trash folder"""
    folder = request.args.get('folder', '')
    
    # First find the memory
    all_memories = []
    for s in STANDARD_FOLDERS:
        all_memories.extend(list_memories(folder, s))
    
    for memory in all_memories:
        if memory_id in (memory["filename"], memory["metadata"]["unique_id"]):
            # Move to trash folder
            result = move_memory(
                memory["filename"],
                folder,
                ".Trash",
                memory["status"],
                "cur"
            )
            
            if result:
                return jsonify({
                    "success": True,
                    "message": f"Memory moved to trash successfully",
                    "memory_id": memory_id
                })
            else:
                return jsonify({"error": f"Failed to move memory to trash: {memory_id}"}), 500
    
    return jsonify({"error": f"Memory not found: {memory_id}"}), 404

@app.route('/search', methods=['GET'])
@require_api_key
def search():
    """Search memories using query parameters or query string"""
    query_string = request.args.get('q', '')
    folder = request.args.get('folder')
    status = request.args.get('status')
    format_type = request.args.get('format', 'json')
    limit = request.args.get('limit')
    offset = request.args.get('offset', '0')
    with_content = request.args.get('with_content', 'false').lower() == 'true'
    debug = request.args.get('debug', 'false').lower() == 'true'
    
    # Build search query
    search_query = SearchQuery()
    
    # If a query string is provided, parse it
    if query_string:
        search_query = parse_search_args(query_string)
    
    # Add pagination
    if limit:
        try:
            search_query.set_pagination(limit=int(limit), offset=int(offset))
        except ValueError:
            return jsonify({"error": "Invalid limit or offset value"}), 400
    
    # Include content if requested
    search_query.with_content(with_content)
    
    # Execute search
    folders = [folder] if folder else None
    statuses = [status] if status else None
    
    try:
        results = search_memories_advanced(search_query, folders, statuses, debug=debug)
        
        return jsonify({
            "count": len(results),
            "query": query_string,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/folders', methods=['GET'])
@require_api_key
def get_folders():
    """List all folders in the Memdir structure"""
    folders = list_folders()
    return jsonify({"folders": folders})

@app.route('/folders', methods=['POST'])
@require_api_key
def create_folder_endpoint():
    """Create a new folder"""
    data = request.json
    
    if not data or 'folder' not in data:
        return jsonify({"error": "Folder name is required"}), 400
    
    folder_name = data['folder']
    
    try:
        create_folder(folder_name)
        return jsonify({
            "success": True,
            "message": f"Folder created successfully: {folder_name}"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to create folder: {str(e)}"}), 500

@app.route('/folders/<path:folder_path>', methods=['DELETE'])
@require_api_key
def delete_folder_endpoint(folder_path):
    """Delete a folder"""
    try:
        delete_folder(folder_path)
        return jsonify({
            "success": True,
            "message": f"Folder deleted successfully: {folder_path}"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to delete folder: {str(e)}"}), 500

@app.route('/folders/<path:folder_path>', methods=['PUT'])
@require_api_key
def rename_folder_endpoint(folder_path):
    """Rename a folder"""
    data = request.json
    
    if not data or 'new_name' not in data:
        return jsonify({"error": "New folder name is required"}), 400
    
    new_name = data['new_name']
    
    try:
        rename_folder(folder_path, new_name)
        return jsonify({
            "success": True,
            "message": f"Folder renamed successfully from {folder_path} to {new_name}"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to rename folder: {str(e)}"}), 500

@app.route('/folders/<path:folder_path>/stats', methods=['GET'])
@require_api_key
def folder_stats_endpoint(folder_path):
    """Get stats for a specific folder"""
    try:
        stats = get_folder_stats(folder_path)
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"Failed to get folder stats: {str(e)}"}), 500

@app.route('/filters/run', methods=['POST'])
@require_api_key
def run_filters_endpoint():
    """Run all filters to organize memories"""
    data = request.json or {}
    dry_run = data.get('dry_run', False)
    
    try:
        results = run_filters(dry_run=dry_run)
        return jsonify({
            "success": True,
            "message": "Filters executed successfully",
            "actions": results
        })
    except Exception as e:
        return jsonify({"error": f"Failed to run filters: {str(e)}"}), 500

if __name__ == '__main__':
    # Check if API key is still the default
    if API_KEY == DEFAULT_API_KEY:
        print("WARNING: Using default API key. Set MEMDIR_API_KEY environment variable for security.")
    
    # Get port from environment variable or use default 5000
    port = int(os.environ.get("MEMDIR_PORT", 5000))
    
    # Run the server
    app.run(host='0.0.0.0', port=port)