#!/usr/bin/env python3
"""
HTTP REST API server for Memdir memory management system.
Provides remote access to Memdir functionality with API key authentication.
"""

import os
import json
import argparse # Added import
from typing import Dict, List, Any, Optional, Union
import os # Ensure os is imported if not already
from datetime import datetime
from functools import wraps
from typing import Optional # Added import

from flask import Flask, request, jsonify, Response
import hmac # Import hmac for compare_digest
# from werkzeug.security import safe_str_cmp # Removed import

from memdir_tools.utils import (
    ensure_memdir_structure,
    get_memdir_folders,
    # get_memdir_base_path_from_config, # REMOVED: Function was removed from utils.py
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
# Import the manager class instead of individual functions
from memdir_tools.folders import MemdirFolderManager
from memdir_tools.filter import run_filters

# Default API key - replace with a secure value in production
DEFAULT_API_KEY = "YOUR_API_KEY_HERE"

app = Flask(__name__)
# Set a default config value, which will be overwritten by run_server.py
app.config['MEMDIR_API_KEY'] = DEFAULT_API_KEY
# Initialize config with default or env var, run_server.py will update if --data-dir is used
app.config['MEMDIR_DATA_DIR'] = os.environ.get("MEMDIR_DATA_DIR", os.path.join(os.getcwd(), "Memdir"))

# Declare folder manager, instantiate later when base_dir is confirmed
folder_manager: Optional[MemdirFolderManager] = None # Added Optional import needed

# Ensure the memdir structure exists before the first request
# Use @app.before_request as @app.before_first_request is deprecated
_structure_ensured = False
@app.before_request
def ensure_structure_once():
    global _structure_ensured, folder_manager
    if not _structure_ensured:
        # Use the config value set by run_server.py or the default/env var
        base_dir = app.config.get('MEMDIR_DATA_DIR')
        if not base_dir:
             # This shouldn't happen if run_server.py sets it, but handle defensively
             print("ERROR: MEMDIR_DATA_DIR not configured!")
             # Optionally raise an error or exit
             return # Or raise ConfigurationError("MEMDIR_DATA_DIR not set")

        print(f"Initializing FolderManager and ensuring structure for base_dir: {base_dir}")
        # Instantiate the folder manager here with the correct base_dir
        folder_manager = MemdirFolderManager(base_dir=base_dir)
        # ensure_memdir_structure is now called within MemdirFolderManager.__init__
        _structure_ensured = True # Mark structure as ensured
        # Also ensure the folder manager knows the correct path
        folder_manager.base_dir = base_dir # Update manager's base_dir
        print(f"Folder manager base dir set to: {folder_manager.base_dir}")
        _structure_ensured = True

def require_api_key(f):
    """Decorator to require API key for all requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key')
        expected_key = app.config.get('MEMDIR_API_KEY', DEFAULT_API_KEY) # Get key from app config

        # Use hmac.compare_digest for secure comparison
        # Note: Both arguments must be bytes
        is_valid = False
        if provided_key and expected_key:
            # Ensure keys are bytes
            expected_key_bytes = expected_key.encode('utf-8') if isinstance(expected_key, str) else expected_key
            provided_key_bytes = provided_key.encode('utf-8') if isinstance(provided_key, str) else provided_key
            is_valid = hmac.compare_digest(provided_key_bytes, expected_key_bytes)

        if not is_valid:
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
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
    folder = request.args.get('folder', '')
    status = request.args.get('status', 'cur')
    with_content = request.args.get('with_content', 'false').lower() == 'true'
    
    # Validate status
    if status not in STANDARD_FOLDERS:
        return jsonify({"error": f"Invalid status: {status}. Must be one of {STANDARD_FOLDERS}"}), 400
    
    # Pass base_dir to list_memories
    memories = list_memories(base_dir, folder, status, include_content=with_content) # Pass base_dir (already correct)
    
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
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
    app.logger.info(f"API /memories POST: Using base_dir from app.config: {base_dir}") # ADD LOGGING
    app.logger.info(f"API /memories POST: Using base_dir from app.config: {base_dir}") # ADD LOGGING
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
    
    # Create the memory, passing base_dir
    try:
        filename = save_memory(base_dir, folder, content, headers, flags) # Pass base_dir (already correct)
        return jsonify({
            "success": True,
            "message": f"Memory created successfully",
            "filename": filename,
            "folder": folder or "root"
        })
    except Exception as e:
        # Log the full exception for debugging, including the base_dir used
        app.logger.error(f"Error in create_memory using base_dir '{base_dir}': {e}", exc_info=True)
        return jsonify({"error": f"Failed to create memory: {str(e)}"}), 500

@app.route('/memories/<memory_id>', methods=['GET'])
@require_api_key
def get_memory(memory_id):
    """Retrieve a specific memory by ID"""
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
    folder = request.args.get('folder', '')
    
    # First try to find by unique ID
    all_memories = []
    for s in STANDARD_FOLDERS:
        # Pass base_dir to list_memories
        all_memories.extend(list_memories(base_dir, folder, s, include_content=True)) # Pass base_dir (already correct)
    
    for memory in all_memories:
        if memory_id in (memory["filename"], memory["metadata"]["unique_id"]):
            return jsonify(memory)
    
    return jsonify({"error": f"Memory not found: {memory_id}"}), 404

@app.route('/memories/<memory_id>', methods=['PUT'])
@require_api_key
def update_memory(memory_id):
    """Update a memory's flags or move it to another folder"""
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
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
        # Pass base_dir to list_memories
        all_memories.extend(list_memories(base_dir, source_folder, s)) # Pass base_dir (already correct)
    
    found_memory = None
    for memory in all_memories:
        if memory_id in (memory["filename"], memory["metadata"]["unique_id"]):
            found_memory = memory
            break

    if not found_memory:
         return jsonify({"error": f"Memory not found: {memory_id}"}), 404

    # Determine the actual source status if not provided
    actual_source_status = source_status or found_memory["status"]

    if target_folder is not None and target_folder != source_folder:
        # Move the memory to another folder, passing base_dir
        result = move_memory( # Pass base_dir (already correct)
            base_dir,
            found_memory["filename"], # Use the actual filename found
            source_folder,
            target_folder,
            actual_source_status,
            target_status,
            flags # Pass flags here too if move_memory handles flag updates during move
        )
        
        if result:
            return jsonify({
                "success": True,
                "message": f"Memory moved successfully",
                "memory_id": memory_id,
                "source": f"{source_folder or 'root'}/{actual_source_status}",
                "destination": f"{target_folder or 'root'}/{target_status}"
            })
        else:
            return jsonify({"error": f"Failed to move memory: {memory_id}"}), 500
    elif flags is not None:
        # Update flags only, passing base_dir
        result = update_memory_flags( # Pass base_dir (already correct)
            base_dir,
            found_memory["filename"], # Use the actual filename found
            source_folder,
            actual_source_status,
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
    else:
        # No operation specified (neither move nor flag update)
        return jsonify({"error": "No update operation specified (target_folder or flags)"}), 400


@app.route('/memories/<memory_id>', methods=['DELETE'])
@require_api_key
def delete_memory(memory_id):
    """Move a memory to the trash folder"""
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
    folder = request.args.get('folder', '')
    
    # First find the memory
    all_memories = []
    for s in STANDARD_FOLDERS:
        # Pass base_dir to list_memories
        all_memories.extend(list_memories(base_dir, folder, s)) # Pass base_dir (already correct)
    
    found_memory = None
    for memory in all_memories:
        if memory_id in (memory["filename"], memory["metadata"]["unique_id"]):
            found_memory = memory
            break

    if not found_memory:
        return jsonify({"error": f"Memory not found: {memory_id}"}), 404

    # Move to trash folder, passing base_dir
    result = move_memory( # Pass base_dir (already correct)
        base_dir,
        found_memory["filename"], # Use actual filename
        folder,
        ".Trash",
        found_memory["status"], # Use actual status
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
    

@app.route('/search', methods=['GET'])
@require_api_key
def search():
    """Search memories using query parameters or query string"""
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
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
        # Pass base_dir to search_memories_advanced
        results = search_memories_advanced(base_dir, search_query, folders, statuses, debug=debug) # Pass base_dir (already correct)
        
        return jsonify({
            "count": len(results),
            "query": query_string,
            "results": results
        })
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

# --- Folder Management Endpoints ---
# Note: These use the folder_manager instance which should have its base_dir updated by the before_request handler

@app.route('/folders', methods=['GET'])
@require_api_key
def get_folders():
    """List all folders in the Memdir structure"""
    # folder_manager.base_dir should be correctly set by before_request
    folder_info = folder_manager.list_folders(recursive=True) # Get all folders recursively
    folder_paths = [f["path"] for f in folder_info] # Extract paths
    return jsonify({"folders": folder_paths})

@app.route('/folders', methods=['POST'])
@require_api_key
def create_folder_endpoint():
    """Create a new folder"""
    data = request.json
    
    if not data or 'folder' not in data:
        return jsonify({"error": "Folder name is required"}), 400
    
    folder_name = data['folder']
    
    try:
        # Use the manager instance (base_dir should be correct)
        success = folder_manager.create_folder(folder_name)
        if success:
            return jsonify({
                "success": True,
                "message": f"Folder created successfully: {folder_name}"
            })
        else:
             # Handle case where folder might already exist
             return jsonify({"error": f"Folder already exists or could not be created: {folder_name}"}), 409
    except Exception as e:
        # Correct indentation for the except block
        return jsonify({"error": f"Failed to create folder: {str(e)}"}), 500

@app.route('/folders/<path:folder_path>', methods=['DELETE'])
@require_api_key
def delete_folder_endpoint(folder_path):
    """Delete a folder"""
    try:
        # Use the manager instance (base_dir should be correct)
        success, message = folder_manager.delete_folder(folder_path)
        if success:
            return jsonify({
                "success": True,
                "message": message
            })
        else:
            # Determine appropriate status code based on message (e.g., 404 if not found)
            status_code = 404 if "does not exist" in message else 400
            return jsonify({"error": message}), status_code
    except Exception as e:
        # Correct indentation for the except block
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
        # Use the manager instance (base_dir should be correct)
        success = folder_manager.rename_folder(folder_path, new_name)
        if success:
            return jsonify({
                "success": True,
                "message": f"Folder renamed successfully from {folder_path} to {new_name}"
            })
        else:
             # Determine appropriate status code (e.g., 404 if not found, 409 if exists)
             return jsonify({"error": f"Failed to rename folder {folder_path}"}), 400
    except Exception as e:
        # Correct indentation for the except block
        return jsonify({"error": f"Failed to rename folder: {str(e)}"}), 500

@app.route('/folders/<path:folder_path>/stats', methods=['GET'])
@require_api_key
def folder_stats_endpoint(folder_path):
    """Get stats for a specific folder"""
    try:
        # Use the manager instance (base_dir should be correct)
        stats = folder_manager.get_folder_stats(folder_path)
        return jsonify(stats)
    except FileNotFoundError:
         return jsonify({"error": f"Folder not found: {folder_path}"}), 404
    except Exception as e:
        return jsonify({"error": f"Failed to get folder stats: {str(e)}"}), 500

@app.route('/filters/run', methods=['POST'])
@require_api_key
def run_filters_endpoint():
    """Run all filters to organize memories"""
    base_dir = app.config.get('MEMDIR_DATA_DIR') # Get base dir for this request
    data = request.json or {}
    dry_run = data.get('dry_run', False)
    
    try:
        # Pass base_dir to run_filters if it needs it (assuming it uses utils internally)
        # If run_filters doesn't need base_dir directly, this isn't required.
        # Check run_filters implementation if issues arise.
        # Assuming run_filters needs base_dir based on other functions
        results = run_filters(base_dir=base_dir, dry_run=dry_run) # Pass base_dir (already correct)
        return jsonify({
            "success": True,
            "message": "Filters executed successfully",
            "actions": results
        })
    except Exception as e:
        # Need to import run_filters from memdir_tools.filter first
        # This block likely needs adjustment based on how run_filters is implemented
        # For now, just return a generic error
        app.logger.error(f"Error running filters: {e}", exc_info=True)
        return jsonify({"error": f"Failed to run filters: {str(e)}"}), 500

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Memdir HTTP API Server")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("MEMDIR_PORT", 5000)), # Default to env var or 5000
        help="Port to run the server on"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("MEMDIR_DATA_DIR", os.path.join(os.getcwd(), "Memdir")), # Default to env var or ./Memdir
        help="Path to the Memdir data directory"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("MEMDIR_API_KEY", DEFAULT_API_KEY), # Default to env var or the hardcoded default
        help="API key required for authentication"
    )
    args = parser.parse_args()

    # --- Update Flask Config ---
    # Use parsed arguments to override defaults/env vars set earlier
    app.config['MEMDIR_DATA_DIR'] = args.data_dir
    app.config['MEMDIR_API_KEY'] = args.api_key

    if app.config['MEMDIR_API_KEY'] == DEFAULT_API_KEY:
        print("WARNING: Using default API key. Set MEMDIR_API_KEY environment variable or use --api-key for security.")

    # --- Run Server ---
    print(f"Starting Memdir server on port {args.port} with data directory: {args.data_dir}")
    # Ensure host is '0.0.0.0' or '127.0.0.1' as needed. '127.0.0.1' is safer for local testing.
    app.run(host='127.0.0.1', port=args.port)
