#!/usr/bin/env python3
"""
Tool handlers for Fei code assistant

This module provides handlers for Claude Universal Assistant tools.
"""

import os
import re
import json
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Set
import subprocess # Added for TimeoutExpired if needed, though TimeoutError is preferred

from fei.tools.code import (
    glob_finder,
    grep_tool,
    code_editor,
    file_viewer,
    directory_explorer,
    shell_runner # Import the instance
)
from fei.tools.repomap import (
    generate_repo_map,
    generate_repo_summary,
    RepoMapper
)
from fei.utils.logging import get_logger

logger = get_logger(__name__)

# Export all handlers so they can be imported from this module
__all__ = [
    "glob_tool_handler",
    "grep_tool_handler",
    "view_handler",
    "edit_handler",
    "replace_handler",
    "ls_handler",
    "regex_edit_handler",
    "batch_glob_handler",
    "find_in_files_handler",
    "smart_search_handler",
    "repo_map_handler",
    "repo_summary_handler",
    "repo_deps_handler",
    "shell_handler",
    "view_process_output_handler",
    "send_process_input_handler",
    "check_process_status_handler",
    "terminate_process_handler"
]

def glob_tool_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for GlobTool"""
    pattern = args.get("pattern")
    path = args.get("path")

    if not pattern:
        return {"error": "Pattern is required"}

    try:
        files = glob_finder.find(pattern, path)
        return {"files": files, "count": len(files)}
    except Exception as e:
        logger.error(f"Error in glob_tool_handler: {e}")
        return {"error": str(e)}

def grep_tool_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for GrepTool"""
    pattern = args.get("pattern")
    include = args.get("include")
    path = args.get("path")

    if not pattern:
        return {"error": "Pattern is required"}

    try:
        results = grep_tool.search(pattern, include, path)

        # Format results for better readability
        formatted_results = {}
        for file_path, matches in results.items():
            formatted_results[file_path] = [{"line": line_num, "content": content} for line_num, content in matches]

        return {
            "results": formatted_results,
            "file_count": len(formatted_results),
            "match_count": sum(len(matches) for matches in formatted_results.values())
        }
    except Exception as e:
        logger.error(f"Error in grep_tool_handler: {e}")
        return {"error": str(e)}

def view_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for View tool"""
    file_path = args.get("file_path")
    limit = args.get("limit")
    offset = args.get("offset", 0)

    if not file_path:
        return {"error": "File path is required"}

    try:
        success, message, lines = file_viewer.view(file_path, limit, offset)

        if not success:
            return {"error": message}

        # Get file info
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        success_count, line_count = file_viewer.count_lines(file_path) # Renamed success variable

        return {
            "content": "\n".join(lines),
            "lines": lines,
            "line_count": line_count if success_count else len(lines),
            "file_size": file_size,
            "file_path": file_path,
            "truncated": limit is not None and (line_count > limit + offset if success_count else False)
        }
    except Exception as e:
        logger.error(f"Error in view_handler: {e}")
        return {"error": str(e)}

def edit_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for Edit tool"""
    file_path = args.get("file_path")
    old_string = args.get("old_string")
    new_string = args.get("new_string")

    if not file_path:
        return {"error": "File path is required"}

    backup_path = None # Initialize backup_path
    if old_string is None:
        # Creating a new file
        if new_string is None:
            return {"error": "New string is required"}

        success, message = code_editor.create_file(file_path, new_string)
    else:
        # Editing existing file
        if new_string is None:
            return {"error": "New string is required"}

        # Correctly unpack the three return values
        success, message, backup_path = code_editor.edit_file(file_path, old_string, new_string)

    # Include backup_path in the response if it exists
    response = {"success": success, "message": message}
    if backup_path:
        response["backup_path"] = backup_path
    return response

def replace_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for Replace tool"""
    file_path = args.get("file_path")
    # Accept either 'content' or 'new_content' from the LLM
    content = args.get("content") if args.get("content") is not None else args.get("new_content")

    if not file_path:
        return {"error": "File path is required"}

    # Handle potential absolute path issue by making it relative if it starts with '/'
    if file_path.startswith('/'):
        logger.warning(f"Received absolute path '{file_path}', treating as relative.")
        file_path = file_path.lstrip('/')

    if content is None:
        # Check both keys before erroring
        return {"error": "Content ('content' or 'new_content') is required"}

    # Correctly unpack the three return values from replace_file
    success, message, backup_path = code_editor.replace_file(file_path, content)

    # Include backup_path in the response if it exists
    response = {"success": success, "message": message}
    if backup_path:
        response["backup_path"] = backup_path
    return response

def ls_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for LS tool"""
    path = args.get("path")
    ignore = args.get("ignore")

    if not path:
        return {"error": "Path is required"}

    success, message, content = directory_explorer.list_directory(path, ignore)

    if not success:
        return {"error": message}

    return {
        "directories": content["dirs"],
        "files": content["files"],
        "directory_count": len(content["dirs"]),
        "file_count": len(content["files"]),
        "path": path
    }

def regex_edit_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for RegexEdit tool"""
    file_path = args.get("file_path")
    pattern = args.get("pattern")
    replacement = args.get("replacement")
    validate = args.get("validate", True)
    max_retries = args.get("max_retries", 3)
    validators = args.get("validators")

    if not file_path:
        return {"error": "File path is required"}

    if not pattern:
        return {"error": "Pattern is required"}

    if replacement is None:
        return {"error": "Replacement is required"}

    try:
        success, message, count = code_editor.regex_replace(
            file_path,
            pattern,
            replacement,
            validate=validate,
            max_retries=max_retries,
            validators=validators
        )

        return {
            "success": success,
            "message": message,
            "count": count,
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Error in regex_edit_handler: {e}")
        return {"error": str(e)}

def batch_glob_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for BatchGlob tool"""
    patterns = args.get("patterns")
    path = args.get("path")
    limit_per_pattern = args.get("limit_per_pattern", 20)

    if not patterns:
        return {"error": "Patterns list is required"}

    try:
        results = {}
        total_count = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(patterns), 5)) as executor:
            future_to_pattern = {
                executor.submit(glob_finder.find, pattern, path, True): pattern
                for pattern in patterns
            }

            for future in as_completed(future_to_pattern):
                pattern = future_to_pattern[future]
                try:
                    files = future.result()
                    # Limit results per pattern if needed
                    if limit_per_pattern and len(files) > limit_per_pattern:
                        files = files[:limit_per_pattern]

                    results[pattern] = files
                    total_count += len(files)
                except Exception as e:
                    results[pattern] = {"error": str(e)}

        return {
            "results": results,
            "total_file_count": total_count,
            "pattern_count": len(patterns)
        }
    except Exception as e:
        logger.error(f"Error in batch_glob_handler: {e}")
        return {"error": str(e)}

def find_in_files_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for FindInFiles tool"""
    files = args.get("files")
    pattern = args.get("pattern")
    case_sensitive = args.get("case_sensitive", False)

    if not files:
        return {"error": "Files list is required"}

    if not pattern:
        return {"error": "Pattern is required"}

    try:
        results = {}
        total_matches = 0

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}

        # Process each file
        for file_path in files:
            if not os.path.isfile(file_path):
                results[file_path] = {"error": "File not found"}
                continue

            try:
                # Use grep_tool's method directly for consistency
                matches = grep_tool.search_single_file(file_path, pattern, case_sensitive)
                if matches:
                    # Ensure matches are in the expected format (list of tuples)
                    if isinstance(matches, list) and all(isinstance(m, tuple) and len(m) == 2 for m in matches):
                         formatted_matches = [{"line": line_num, "content": content} for line_num, content in matches]
                         results[file_path] = formatted_matches
                         total_matches += len(formatted_matches)
                    else:
                         logger.warning(f"Unexpected match format from search_single_file for {file_path}: {matches}")
                         results[file_path] = {"error": "Unexpected match format received"}

            except Exception as e:
                logger.error(f"Error searching file {file_path}: {e}", exc_info=True)
                results[file_path] = {"error": str(e)}

        return {
            "results": results,
            "match_count": total_matches,
            "file_count": len(files),
            "files_with_matches": len([f for f, res in results.items() if isinstance(res, list) and res])
        }
    except Exception as e:
        logger.error(f"Error in find_in_files_handler: {e}")
        return {"error": str(e)}


def smart_search_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for SmartSearch tool"""
    query = args.get("query")
    context = args.get("context")
    language = args.get("language", "python")  # Default to Python

    if not query:
        return {"error": "Query is required"}

    try:
        # Determine file patterns based on language
        file_patterns = {
            "python": ["**/*.py"],
            "javascript": ["**/*.js", "**/*.jsx", "**/*.ts", "**/*.tsx"],
            "java": ["**/*.java"],
            "c": ["**/*.c", "**/*.h"],
            "cpp": ["**/*.cpp", "**/*.hpp", "**/*.cc", "**/*.h"],
            "csharp": ["**/*.cs"],
            "go": ["**/*.go"],
            "ruby": ["**/*.rb"],
            "php": ["**/*.php"],
            "rust": ["**/*.rs"],
            "swift": ["**/*.swift"],
            "kotlin": ["**/*.kt"]
        }

        patterns = file_patterns.get(language.lower(), ["**/*"])

        # Parse query to create search patterns
        search_patterns = []

        # Process class and function definitions
        if "class" in query.lower():
            class_name = re.search(r'class\s+([A-Za-z0-9_]+)', query)
            if class_name:
                name = class_name.group(1)
                if language.lower() == "python":
                    search_patterns.append(f"class\\s+{name}\\b")
                elif language.lower() in ["javascript", "typescript", "java", "csharp", "cpp"]:
                    search_patterns.append(f"class\\s+{name}\\b")

        elif "function" in query.lower() or "def" in query.lower():
            func_name = re.search(r'(function|def)\s+([A-Za-z0-9_]+)', query)
            if func_name:
                name = func_name.group(2)
                if language.lower() == "python":
                    search_patterns.append(f"def\\s+{name}\\b")
                elif language.lower() in ["javascript", "typescript"]:
                    search_patterns.append(f"function\\s+{name}\\b")
                    # Also catch method definitions and arrow functions
                    search_patterns.append(f"\\b{name}\\s*=\\s*function")
                    search_patterns.append(f"\\b{name}\\s*[=:]\\s*\\(")

        # If no specific patterns created, use the query as a general search term
        if not search_patterns:
            # Extract potential identifier
            words = re.findall(r'\b[A-Za-z0-9_]+\b', query)
            for word in words:
                # Avoid common keywords and very short words
                if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that', 'def', 'class', 'function', 'import', 'from', 'return', 'if', 'else', 'try', 'except', 'while', 'for']:
                    search_patterns.append(f"\\b{word}\\b")
            # If still no patterns, use the raw query words as fallback
            if not search_patterns:
                 search_patterns = [f"\\b{re.escape(word)}\\b" for word in query.split() if len(word) > 1]


        all_results = {}
        files_searched_count = 0

        # Search for files first
        files = set() # Use set to avoid duplicates
        for pattern in patterns:
            try:
                found_files = glob_finder.find(pattern)
                files.update(found_files)
            except Exception as glob_e:
                 logger.warning(f"Error during globbing pattern '{pattern}': {glob_e}")
        files = list(files)
        files_searched_count = len(files)


        # Then search in files
        for search_pattern in search_patterns:
            results = {}
            for file_path in files:
                try:
                    # Assuming search_single_file returns list of (line_num, content) tuples or empty list
                    matches = grep_tool.search_single_file(file_path, search_pattern)
                    if matches:
                        # Format matches correctly
                        formatted_matches = [{"line": line_num, "content": content} for line_num, content in matches]
                        results[file_path] = formatted_matches
                except Exception as search_e:
                     logger.warning(f"Error searching pattern '{search_pattern}' in file '{file_path}': {search_e}")
                     # Optionally add error info to results: results[file_path] = {"error": str(search_e)}

            all_results[search_pattern] = results

        # Process results to summarize findings
        summary = []
        for pattern, results in all_results.items():
            if results:
                files_with_matches = len(results)
                total_matches = sum(len(matches) for matches in results.values() if isinstance(matches, list)) # Sum only if list

                # Get a short sample of code for context
                samples = []
                for file_path, matches in list(results.items())[:3]:  # Take first 3 files
                    if isinstance(matches, list) and matches: # Check if it's a list and not empty
                        filename = os.path.basename(file_path)
                        # Use the formatted match dictionary
                        first_match = matches[0]
                        line_num = first_match.get("line", "?")
                        line_content = first_match.get("content", "").strip()
                        samples.append(f"{filename}:{line_num}: {line_content}")

                summary.append({
                    "pattern": pattern,
                    "files": files_with_matches,
                    "matches": total_matches,
                    "samples": samples
                })

        return {
            "patterns_searched": len(search_patterns),
            "files_searched": files_searched_count,
            "summary": summary,
            "language": language,
            "detailed_results": all_results # Keep detailed results if needed
        }
    except Exception as e:
        logger.error(f"Error in smart_search_handler: {e}", exc_info=True)
        return {"error": str(e)}

def repo_map_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for RepoMap tool"""
    path = args.get("path", os.getcwd())
    token_budget = args.get("token_budget", 1000)
    exclude_patterns = args.get("exclude_patterns")

    try:
        # Generate repository map
        repo_map = generate_repo_map(path, token_budget, exclude_patterns)

        # Split into lines for better display
        map_lines = repo_map.strip().split("\n")

        return {
            "map": repo_map,
            "lines": map_lines,
            "token_count": len(repo_map.split()) * 1.3,  # Rough token estimation
            "repository": os.path.basename(os.path.abspath(path))
        }
    except Exception as e:
        logger.error(f"Error in repo_map_handler: {e}")
        return {"error": str(e)}

def repo_summary_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for RepoSummary tool"""
    path = args.get("path", os.getcwd())
    max_tokens = args.get("max_tokens", 500)
    exclude_patterns = args.get("exclude_patterns")

    try:
        # Generate repository summary
        repo_summary = generate_repo_summary(path, max_tokens, exclude_patterns)

        # Split into lines for better display
        summary_lines = repo_summary.strip().split("\n")

        # Extract some key stats from the summary
        module_count = len([line for line in summary_lines if line.startswith("## ")])

        return {
            "summary": repo_summary,
            "lines": summary_lines,
            "token_count": len(repo_summary.split()) * 1.3,  # Rough token estimation
            "repository": os.path.basename(os.path.abspath(path)),
            "module_count": module_count
        }
    except Exception as e:
        logger.error(f"Error in repo_summary_handler: {e}")
        return {"error": str(e)}

def repo_deps_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for RepoDependencies tool"""
    path = args.get("path", os.getcwd())
    module = args.get("module")
    depth = args.get("depth", 1) # Depth currently unused, but kept for potential future use

    try:
        # Create a repo mapper
        mapper = RepoMapper(path)

        # Get JSON map with all dependency information
        repo_map_json = mapper.generate_json()
        repo_data = json.loads(repo_map_json)

        if 'error' in repo_data:
            return {"error": repo_data['error']}

        # Extract dependencies at the module level
        module_deps = {}
        file_deps = {}

        for file_data in repo_data.get('mapped_files', []):
            file_path = file_data['path']
            file_deps[file_path] = file_data.get('dependencies', [])

            # Extract module from path (simple split, might need refinement for complex structures)
            if '/' in file_path:
                file_module = file_path.split('/')[0]
                # Filter by requested module if specified
                if module and module != file_module:
                    continue

                # Add to module dependencies
                if file_module not in module_deps:
                    module_deps[file_module] = set()

                # Add all dependencies, ensuring they are different modules
                for dep_file in file_data.get('dependencies', []):
                    if '/' in dep_file:
                        dep_module = dep_file.split('/')[0]
                        if dep_module != file_module:
                            module_deps[file_module].add(dep_module)

        # Convert sets to lists for JSON serialization
        for mod in module_deps:
            module_deps[mod] = sorted(list(module_deps[mod])) # Sort for consistent output

        # Format a visual representation of the dependencies
        visual_deps = []
        visual_deps.append("# Repository Dependencies")
        visual_deps.append(f"Repository: {repo_data['repository']}")
        visual_deps.append(f"Total files analyzed: {repo_data['file_count']}")
        visual_deps.append("")

        # Module dependencies
        visual_deps.append("## Module Dependencies")
        if module_deps:
            for mod, deps in sorted(module_deps.items()): # Sort modules
                if deps:
                    deps_str = ", ".join(deps[:5])
                    if len(deps) > 5:
                        deps_str += f" and {len(deps) - 5} more"
                    visual_deps.append(f"- {mod} → {deps_str}")
                else:
                    visual_deps.append(f"- {mod} → (No external module dependencies found)")
        else:
             visual_deps.append("(No inter-module dependencies found)")


        return {
            "module_dependencies": module_deps,
            "file_dependencies": file_deps, # Consider limiting this if it gets too large
            "visual": "\n".join(visual_deps),
            "repository": repo_data['repository'],
            "file_count": repo_data['file_count']
        }
    except Exception as e:
        logger.error(f"Error in repo_deps_handler: {e}", exc_info=True)
        return {"error": str(e)}

def shell_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for Shell tool"""
    command = args.get("command")
    timeout = args.get("timeout", 60)
    current_dir = args.get("current_dir")
    background = args.get("background")  # This will be None if not specified
    # Note: do_as_i_say can only be set via command line, not by the AI

    if not command:
        return {"error": "Command is required"}

    try:
        # Change directory if specified
        original_dir = None
        if current_dir:
            abs_current_dir = os.path.abspath(current_dir) # Ensure absolute path
            if os.path.isdir(abs_current_dir):
                original_dir = os.getcwd()
                try:
                    os.chdir(abs_current_dir)
                    logger.debug(f"Changed directory to {abs_current_dir}")
                except Exception as cd_err:
                     logger.error(f"Failed to change directory to {abs_current_dir}: {cd_err}")
                     return {"error": f"Failed to change directory: {cd_err}"}
            else:
                 logger.warning(f"Specified current_dir '{current_dir}' is not a valid directory.")
                 # Decide whether to error or proceed in cwd
                 return {"error": f"Invalid current_dir specified: {current_dir}"}


        try:
            # Run the command with background parameter
            # Note: do_as_i_say is controlled by command line, not by the AI
            result = shell_runner.run_command(command, timeout, background)

            # Format the result
            response = {
                "success": result["success"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"]
            }

            # Add error if present
            if "error" in result and result["error"]: # Check if error is not empty
                response["error"] = result["error"]

            # Add background info if present
            if "background" in result and result["background"]: # Check if background is True
                response["background"] = result["background"]
                response["pid"] = result.get("pid")
                response["note"] = result.get("note")

            return response
        except Exception as e: # Catch errors from shell_runner.run_command
            logger.error(f"Error running command in shell_handler: {e}", exc_info=True)
            # Ensure result is initialized if error happens before assignment
            if 'result' not in locals():
                 result = {"success": False, "error": str(e), "stdout": "", "stderr": str(e), "exit_code": -1}
            # Format error response
            response = {
                "success": False,
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", "") + f"\nHandler Error: {str(e)}",
                "exit_code": result.get("exit_code", -1),
                "error": f"Handler Error: {str(e)}"
            }
            if "pid" in result: response["pid"] = result["pid"]
            return response
        finally:
            # Restore original directory if changed
            if original_dir:
                try:
                    os.chdir(original_dir)
                    logger.debug(f"Restored directory to {original_dir}")
                except Exception as cd_back_err:
                     logger.error(f"Failed to restore original directory {original_dir}: {cd_back_err}")
                     # This is problematic, but we probably shouldn't overwrite the primary error
    except Exception as outer_e:
        # Catch errors related to directory changes or unexpected issues
        logger.error(f"Outer error in shell_handler: {outer_e}", exc_info=True)
        return {"error": f"Outer handler error: {str(outer_e)}"}


# --- Process Management Handlers ---

def view_process_output_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handler for view_process_output tool.
    NOTE: This is a placeholder. Retrieving arbitrary historical output
    from background processes requires changes to ShellRunner to redirect
    stdout/stderr to files or use async pipes from the start.
    """
    pid = args.get("pid")
    max_lines = args.get("max_lines", 50) # Parameter currently unused

    if pid is None:
        return {"error": "PID is required"}

    try:
        process = shell_runner.running_processes.get(pid)
        if not process:
            # Check if the PID exists but isn't managed
            try:
                os.kill(pid, 0)
                return {"error": f"Process with PID {pid} exists but is not managed by Fei."}
            except ProcessLookupError:
                 return {"error": f"Process with PID {pid} not found."}
            except PermissionError:
                 return {"error": f"Permission denied to check status of process {pid}."}
            except Exception as check_err:
                 return {"error": f"Error checking unmanaged process {pid}: {check_err}"}


        # --- Placeholder Logic ---
        # Actual implementation requires ShellRunner changes.
        status = "running" if process.poll() is None else f"exited (code: {process.poll()})"
        output_note = ("NOTE: Real-time/historical output retrieval is not fully implemented. "
                       "This handler currently only confirms process status and provides placeholders.")
        # --- End Placeholder Logic ---

        # Attempt to read recent output if available (basic implementation)
        stdout_recent = "[Not Available]"
        stderr_recent = "[Not Available]"
        try:
            # This will only work if output was piped and is non-blocking
            # It's highly likely to be incomplete or unavailable in the current ShellRunner
            if process.stdout and not process.stdout.closed:
                 # Non-blocking read attempt (might need adjustments)
                 # stdout_recent = process.stdout.read(1024).decode('utf-8', errors='ignore') # Example
                 pass # Placeholder - reading requires async handling or file redirection
            if process.stderr and not process.stderr.closed:
                 # stderr_recent = process.stderr.read(1024).decode('utf-8', errors='ignore') # Example
                 pass # Placeholder
        except Exception as read_err:
             logger.warning(f"Could not read placeholder output for PID {pid}: {read_err}")


        return {
            "pid": pid,
            "stdout": stdout_recent,
            "stderr": stderr_recent,
            "status": status,
            "note": output_note
        }
    except Exception as e:
        logger.error(f"Error in view_process_output_handler for PID {pid}: {e}", exc_info=True)
        return {"error": str(e)}

# Corrected send_process_input_handler
def send_process_input_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for send_process_input tool. Sends input string to the stdin of a managed background process."""
    pid = args.get("pid")
    input_string = args.get("input_string") # Changed from input to input_string

    if pid is None:
        return {"error": "PID is required"}
    if input_string is None: # Check the corrected key
        return {"error": "input_string is required"}

    try:
        # Ensure the process is managed by ShellRunner
        process = shell_runner.running_processes.get(pid)
        if not process:
            return {"error": f"Process with PID {pid} not found or not managed by Fei."}

        # Check if process is still running
        if process.poll() is not None:
            return {"error": f"Process with PID {pid} has already exited (code: {process.poll()})."}

        # Check if stdin is available and open
        if not process.stdin or process.stdin.closed:
             # Corrected indentation for the return statement below
             return {"error": f"Stdin for process {pid} is not available or closed. Ensure the command was started with stdin=subprocess.PIPE."}

        # Write input to process stdin
        try:
            # Add newline if not present, as many CLI tools expect line-based input
            if not input_string.endswith('\n'):
                input_string += '\n'
            process.stdin.write(input_string.encode('utf-8'))
            process.stdin.flush() # Ensure data is sent immediately
            logger.info(f"Sent input to process {pid}: {input_string.strip()}")
            return {"success": True, "message": f"Sent input to process {pid}."}
        except (OSError, BrokenPipeError, ValueError) as e: # Catch potential errors during write/flush
            logger.error(f"Error writing to stdin for process {pid}: {e}", exc_info=True)
            return {"error": f"Failed to send input to process {pid}: {e}"}

    except Exception as e:
        logger.error(f"Error in send_process_input_handler for PID {pid}: {e}", exc_info=True) # Added exc_info
        return {"error": str(e)}

# Corrected check_process_status_handler
def check_process_status_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for check_process_status tool. Checks if a process (managed or unmanaged) is running."""
    pid = args.get("pid")

    if pid is None:
        return {"error": "PID is required"}

    try: # Outer try for the handler logic
        # Check if it's a managed process first
        process = shell_runner.running_processes.get(pid)
        if process:
            exit_code = process.poll()
            if exit_code is None:
                # Managed and running
                return {"pid": pid, "status": "running", "managed": True, "exit_code": None, "message": "Process is actively managed by Fei and running."}
            else:
                # Managed but exited, remove from tracking
                logger.info(f"Managed process {pid} found exited (code: {exit_code}). Removing from tracking.")
                with shell_runner._lock:
                    # Use pop with default None to avoid KeyError if removed concurrently
                    shell_runner.running_processes.pop(pid, None)
                return {"pid": pid, "status": "exited", "managed": True, "exit_code": exit_code, "message": f"Managed process exited with code {exit_code}."}
        else:
            # Not managed by ShellRunner, check if the PID exists using signal 0
            try:
                os.kill(pid, 0)
                # If os.kill doesn't raise error, process exists but isn't managed by us
                return {"pid": pid, "status": "running", "managed": False, "exit_code": None, "message": "Process exists but is not managed by Fei."}
            except ProcessLookupError:
                # If os.kill raises ProcessLookupError (subclass of OSError), process does not exist
                return {"pid": pid, "status": "not_found", "managed": False, "exit_code": None, "message": "Process not found or already terminated."}
            except PermissionError:
                 # We don't have permission to signal the process, but it likely exists
                 return {"pid": pid, "status": "running", "managed": False, "exit_code": None, "message": "Process exists (not managed by Fei), but permission denied to check status fully."}
            except Exception as kill_err: # Catch other potential errors from os.kill
                 logger.error(f"Unexpected error checking unmanaged process {pid} with os.kill: {kill_err}", exc_info=True)
                 return {"error": f"Error checking process {pid}: {kill_err}"}

    except Exception as e: # Catch errors in the handler logic itself (Corrects error on Line 577)
        logger.error(f"Error in check_process_status_handler for PID {pid}: {e}", exc_info=True)
        return {"error": f"Internal error checking process status: {str(e)}"}


# Corrected terminate_process_handler
def terminate_process_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for terminate_process tool. Attempts to terminate a process group gracefully (SIGTERM) then forcefully (SIGKILL)."""
    pid = args.get("pid")
    force = args.get("force", False) # Default to graceful termination
    timeout = args.get("timeout", 3.0) # Seconds to wait after SIGTERM before SIGKILL if force=True

    if pid is None:
        return {"error": "PID is required"}

    logger.info(f"Attempting to terminate process PID={pid} (Force: {force}, Timeout: {timeout}s)")

    try: # Outer try block for the whole handler
        # Check if it's managed by ShellRunner
        process = shell_runner.running_processes.get(pid)
        is_managed = bool(process)

        # --- Check if already exited ---
        if is_managed and process.poll() is not None:
            logger.info(f"Managed process {pid} already exited (code: {process.poll()}). Removing.")
            with shell_runner._lock:
                shell_runner.running_processes.pop(pid, None)
            return {"success": True, "message": f"Managed process {pid} had already exited."}
        elif not is_managed:
            try:
                os.kill(pid, 0) # Check if exists using signal 0
            except ProcessLookupError:
                logger.info(f"Unmanaged process {pid} not found (already terminated?).")
                return {"success": True, "message": f"Process {pid} not found (already terminated?)."}
            except PermissionError:
                logger.warning(f"Permission denied to check unmanaged process {pid}. Cannot confirm status before termination attempt.")
                # Proceed with termination attempt, but outcome is uncertain.
            except Exception as e:
                 logger.error(f"Error checking unmanaged process {pid} before termination: {e}", exc_info=True)
                 # Proceed cautiously, but log the error

        # --- Determine Target: PGID or PID ---
        target_pgid = None
        try:
            # Get the process group ID. Targeting the PGID is generally safer for cleanup.
            target_pgid = os.getpgid(pid)
            logger.debug(f"Targeting process group PGID={target_pgid} for PID={pid}")
        except ProcessLookupError:
            # Process likely died between check and getpgid
            logger.info(f"Process {pid} not found when getting PGID. Assuming terminated.")
            if is_managed: # Clean up if managed and still in dict
                with shell_runner._lock:
                    shell_runner.running_processes.pop(pid, None)
            return {"success": True, "message": f"Process {pid} not found (already terminated?)."}
        except Exception as e:
            logger.warning(f"Could not get PGID for PID={pid}: {e}. Will target PID directly.", exc_info=True)
            target_pgid = None # Fallback to targeting PID

        # --- Send SIGTERM ---
        sigterm_success = False
        try:
            if target_pgid is not None and target_pgid != os.getpid(): # Avoid killing self if PGID is own PID
                logger.info(f"Sending SIGTERM to process group {target_pgid} (for PID {pid})")
                os.killpg(target_pgid, signal.SIGTERM)
            else: # Target PID directly if PGID failed or is self
                logger.info(f"Sending SIGTERM to process PID {pid}")
                os.kill(pid, signal.SIGTERM)
            sigterm_success = True
        except ProcessLookupError:
            logger.info(f"Process/Group {target_pgid or pid} not found during SIGTERM. Assuming terminated.")
            if is_managed:
                with shell_runner._lock:
                    shell_runner.running_processes.pop(pid, None)
            return {"success": True, "message": f"Process {pid} not found during SIGTERM (already terminated?)."}
        except PermissionError:
            logger.error(f"Permission denied sending SIGTERM to {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}.")
            return {"error": f"Permission denied to send SIGTERM to process {pid}."}
        except Exception as e:
            logger.error(f"Error sending SIGTERM to {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}: {e}", exc_info=True)
            # Don't return error yet, might still try SIGKILL if forced
            sigterm_success = False # Mark as failed

        # --- Wait and Check (Managed Processes Only) ---
        if is_managed:
            logger.debug(f"Waiting up to {timeout}s for managed process {pid} to terminate after SIGTERM...")
            try:
                # process.wait can raise TimeoutExpired or TimeoutError depending on Python version
                process.wait(timeout=timeout)
                exit_code = process.poll() # Should have exited now
                logger.info(f"Managed process {pid} terminated gracefully after SIGTERM (code: {exit_code}).")
                with shell_runner._lock:
                    shell_runner.running_processes.pop(pid, None)
                return {"success": True, "message": f"Successfully terminated managed process {pid}."}
            except (TimeoutError, subprocess.TimeoutExpired): # Catch both potential timeout errors
                logger.warning(f"Managed process {pid} did not terminate within {timeout}s after SIGTERM.")
                if not force:
                    return {"error": f"Managed process {pid} did not terminate within {timeout}s. Use force=true to kill."}
                # Proceed to SIGKILL if force=True
            except Exception as wait_err:
                 logger.error(f"Error waiting for managed process {pid}: {wait_err}", exc_info=True)
                 if not force:
                     return {"error": f"Error waiting for process {pid} termination: {str(wait_err)}"}
                 # Proceed to SIGKILL if force=True

        # --- Send SIGKILL (if force=True and process didn't terminate or if unmanaged and force=True) ---
        # Corrected logic: Only SIGKILL if force=True AND the process is likely still running
        process_likely_running = False
        if is_managed and process.poll() is None: # Managed and didn't terminate after wait
             process_likely_running = True
        elif not is_managed: # Unmanaged, assume it might be running if SIGTERM was attempted
             try:
                 if target_pgid: os.killpg(target_pgid, 0)
                 else: os.kill(pid, 0)
                 process_likely_running = True # Signal 0 succeeded, it's running
             except (ProcessLookupError, OSError):
                 process_likely_running = False # Signal 0 failed, it's gone
             except Exception as check_err:
                  logger.warning(f"Could not confirm unmanaged process {pid} status before potential SIGKILL: {check_err}")
                  process_likely_running = True # Assume running if check fails

        if force and process_likely_running:
            logger.warning(f"Process {pid} likely still alive. Sending SIGKILL to {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}.")
            try:
                if target_pgid is not None and target_pgid != os.getpid():
                    os.killpg(target_pgid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
                time.sleep(0.1) # Give SIGKILL a moment

                # Final check (optional, SIGKILL is usually effective)
                final_code = process.poll() if is_managed else None # Check final status if managed
                message = f"Force-killed {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}."
                if is_managed:
                    message += f" Final exit code: {final_code}." if final_code is not None else " Final state unknown."
                    with shell_runner._lock: # Clean up managed process
                        shell_runner.running_processes.pop(pid, None)
                return {"success": True, "message": message}
            except ProcessLookupError:
                logger.info(f"Process/Group {target_pgid or pid} disappeared during SIGKILL attempt (or was already gone).")
                if is_managed:
                    with shell_runner._lock:
                        shell_runner.running_processes.pop(pid, None)
                return {"success": True, "message": f"Process {pid} terminated after SIGKILL attempt."}
            except PermissionError:
                logger.error(f"Permission denied sending SIGKILL to {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}.")
                # Still clean up managed process entry if permission denied
                if is_managed:
                     with shell_runner._lock:
                         shell_runner.running_processes.pop(pid, None)
                return {"error": f"Permission denied to send SIGKILL to process {pid}."}
            except Exception as e:
                logger.error(f"Error sending SIGKILL to {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}: {e}", exc_info=True)
                # Clean up managed process entry even if SIGKILL fails
                if is_managed:
                     with shell_runner._lock:
                         shell_runner.running_processes.pop(pid, None)
                return {"error": f"Error sending SIGKILL: {str(e)}"}

        # --- Final Reporting Section (if SIGKILL wasn't sent or needed) ---
        elif not is_managed and sigterm_success:
            # We sent SIGTERM to an unmanaged process, but didn't force kill. Status is unknown.
            return {"success": True, "message": f"Sent SIGTERM to unmanaged {'group ' + str(target_pgid) if target_pgid else 'PID ' + str(pid)}. Final status unknown as force=False."}
        elif is_managed and not force and process.poll() is None:
            # This case should have been handled by the timeout error return, but as a fallback:
             return {"error": f"Managed process {pid} did not terminate after SIGTERM and force=False."}
        elif not sigterm_success and not force:
             # SIGTERM failed, and we are not forcing. Report the likely earlier error.
             return {"error": f"Failed to send SIGTERM to process {pid} and force=False. Check previous errors."}
        elif not process_likely_running and force:
             # Force was true, but process was already gone before SIGKILL attempt
             logger.info(f"Process {pid} was already terminated before SIGKILL was needed (force=True).")
             if is_managed: # Ensure cleanup
                  with shell_runner._lock:
                      shell_runner.running_processes.pop(pid, None)
             return {"success": True, "message": f"Process {pid} already terminated before SIGKILL needed."}
        else:
             # Fallback for any unhandled scenario
             logger.warning(f"Reached unexpected state in terminate_process_handler for PID {pid}. is_managed={is_managed}, force={force}, sigterm_success={sigterm_success}, process_likely_running={process_likely_running}")
             # Clean up managed state just in case
             if is_managed:
                  with shell_runner._lock:
                      shell_runner.running_processes.pop(pid, None)
             return {"error": "Reached unexpected state during termination."}


    except Exception as e: # Catch-all for unexpected errors in the handler's outer scope
        logger.error(f"Unhandled error in terminate_process_handler for PID {pid}: {e}", exc_info=True)
        # Clean up managed state if error occurs early
        if 'is_managed' in locals() and is_managed and pid in shell_runner.running_processes:
             with shell_runner._lock:
                 shell_runner.running_processes.pop(pid, None)
        return {"error": f"Internal error terminating process: {str(e)}"}

# --- End Process Management Handlers ---
