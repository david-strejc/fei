#!/usr/bin/env python3
"""
Tool handlers for Fei code assistant

This module provides handlers for Claude Universal Assistant tools.
"""

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple, Set

from fei.tools.code import (
    glob_finder,
    grep_tool,
    code_editor,
    file_viewer,
    directory_explorer,
    shell_runner
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
    "shell_handler"
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
        success, line_count = file_viewer.count_lines(file_path)
        
        return {
            "content": "\n".join(lines),
            "lines": lines,
            "line_count": line_count if success else len(lines),
            "file_size": file_size,
            "file_path": file_path,
            "truncated": limit is not None and line_count > limit + offset if success else False
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
    
    if old_string is None:
        # Creating a new file
        if new_string is None:
            return {"error": "New string is required"}
        
        success, message = code_editor.create_file(file_path, new_string)
    else:
        # Editing existing file
        if new_string is None:
            return {"error": "New string is required"}
        
        success, message = code_editor.edit_file(file_path, old_string, new_string)
    
    return {"success": success, "message": message}

def replace_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for Replace tool"""
    file_path = args.get("file_path")
    content = args.get("content")
    
    if not file_path:
        return {"error": "File path is required"}
    
    if content is None:
        return {"error": "Content is required"}
    
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
                matches = grep_tool.search_single_file(file_path, pattern)
                if matches:
                    results[file_path] = matches
                    total_matches += len(matches)
            except Exception as e:
                results[file_path] = {"error": str(e)}
        
        return {
            "results": results,
            "match_count": total_matches,
            "file_count": len(files),
            "files_with_matches": len([f for f in results if isinstance(results[f], list) and results[f]])
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
                if len(word) > 2 and word.lower() not in ['the', 'and', 'for', 'with', 'this', 'that']:
                    search_patterns.append(f"\\b{word}\\b")
        
        all_results = {}
        
        # Search for files first
        files = []
        for pattern in patterns:
            files.extend(glob_finder.find(pattern))
        
        # Then search in files
        for search_pattern in search_patterns:
            results = {}
            for file_path in files:
                matches = grep_tool.search_single_file(file_path, search_pattern)
                if matches:
                    results[file_path] = matches
            
            all_results[search_pattern] = results
        
        # Process results to summarize findings
        summary = []
        for pattern, results in all_results.items():
            if results:
                files_with_matches = len(results)
                total_matches = sum(len(matches) for matches in results.values())
                
                # Get a short sample of code for context
                samples = []
                for file_path, matches in list(results.items())[:3]:  # Take first 3 files
                    if matches:
                        filename = os.path.basename(file_path)
                        line_num, line_content = matches[0]  # First match
                        samples.append(f"{filename}:{line_num}: {line_content.strip()}")
                
                summary.append({
                    "pattern": pattern,
                    "files": files_with_matches,
                    "matches": total_matches,
                    "samples": samples
                })
        
        return {
            "patterns_searched": len(search_patterns),
            "files_searched": len(files),
            "summary": summary,
            "language": language,
            "detailed_results": all_results
        }
    except Exception as e:
        logger.error(f"Error in smart_search_handler: {e}")
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
    depth = args.get("depth", 1)
    
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
            
            # Extract module from path
            if '/' in file_path:
                file_module = file_path.split('/')[0]
                if module and module != file_module:
                    continue
                    
                # Add to module dependencies
                if file_module not in module_deps:
                    module_deps[file_module] = set()
                
                # Add all dependencies
                for dep_file in file_data.get('dependencies', []):
                    if '/' in dep_file:
                        dep_module = dep_file.split('/')[0]
                        if dep_module != file_module:
                            module_deps[file_module].add(dep_module)
        
        # Convert sets to lists for JSON serialization
        for mod in module_deps:
            module_deps[mod] = list(module_deps[mod])
        
        # Format a visual representation of the dependencies
        visual_deps = []
        visual_deps.append("# Repository Dependencies")
        visual_deps.append(f"Repository: {repo_data['repository']}")
        visual_deps.append(f"Total files analyzed: {repo_data['file_count']}")
        visual_deps.append("")
        
        # Module dependencies
        visual_deps.append("## Module Dependencies")
        for mod, deps in module_deps.items():
            if deps:
                deps_str = ", ".join(deps[:5])
                if len(deps) > 5:
                    deps_str += f" and {len(deps) - 5} more"
                visual_deps.append(f"- {mod} → {deps_str}")
        
        return {
            "module_dependencies": module_deps,
            "file_dependencies": file_deps,
            "visual": "\n".join(visual_deps),
            "repository": repo_data['repository'],
            "file_count": repo_data['file_count']
        }
    except Exception as e:
        logger.error(f"Error in repo_deps_handler: {e}")
        return {"error": str(e)}

def shell_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for Shell tool"""
    command = args.get("command")
    timeout = args.get("timeout", 60)
    current_dir = args.get("current_dir")
    background = args.get("background")  # This will be None if not specified
    
    if not command:
        return {"error": "Command is required"}
    
    try:
        # Change directory if specified
        original_dir = None
        if current_dir and os.path.isdir(current_dir):
            original_dir = os.getcwd()
            os.chdir(current_dir)
        
        try:
            # Run the command with background parameter
            result = shell_runner.run_command(command, timeout, background)
            
            # Format the result
            response = {
                "success": result["success"],
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "exit_code": result["exit_code"]
            }
            
            # Add error if present
            if "error" in result:
                response["error"] = result["error"]
            
            # Add background info if present
            if "background" in result:
                response["background"] = result["background"]
                response["pid"] = result.get("pid")
                response["note"] = result.get("note")
                
            return response
            
        finally:
            # Restore original directory if changed
            if original_dir:
                os.chdir(original_dir)
                
    except Exception as e:
        logger.error(f"Error in shell_handler: {e}")
        return {"error": str(e)}
