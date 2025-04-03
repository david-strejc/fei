#!/usr/bin/env python3
"""
Code tools implementation for Fei

This module provides tools for file searching, pattern matching, and code editing
with improved security and performance.
"""

import os
import re
import glob
import shutil
import hashlib
import tempfile
import mimetypes
import subprocess
import threading
import asyncio
import signal
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Pattern, Set, Callable, BinaryIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from fei.utils.logging import get_logger

logger = get_logger(__name__)

# Initialize mimetypes
mimetypes.init()

# Constants
DEFAULT_MAX_FILE_SIZE_MB = 10
DEFAULT_COMMAND_TIMEOUT = 60  # seconds
MAX_OUTPUT_SIZE = 50000

class FileAccessError(Exception):
    """Exception raised for file access errors"""
    pass

class CommandExecutionError(Exception):
    """Exception raised for command execution errors"""
    pass

class ValidationError(Exception):
    """Exception raised for validation errors"""
    pass

class GlobFinder:
    """Fast file pattern matching tool with efficient caching"""
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize with optional base path
        
        Args:
            base_path: Base directory for relative paths
        """
        self.base_path = base_path or os.getcwd()
        # Cache for glob results to improve performance
        self._cache = {}
        self._cache_timestamp = {}
        # Cache expiration time (seconds)
        self._cache_expiration = 60
    
    def _check_path_safety(self, path: str) -> None:
        """
        Check if a path is safe to access
        
        Args:
            path: Path to check
            
        Raises:
            FileAccessError: If path is not safe
        """
        # Get absolute path
        abs_path = os.path.abspath(path)
        
        # Check if path is outside the base path
        if not abs_path.startswith(self.base_path):
            raise FileAccessError(f"Cannot access path outside base directory: {path}")
        
        # TODO: Add more security checks if needed
    
    def _get_cache_key(self, pattern: str, path: str) -> str:
        """
        Get a cache key for glob results
        
        Args:
            pattern: Glob pattern
            path: Search path
            
        Returns:
            Cache key
        """
        # Include pattern and normalized path in cache key
        return f"{pattern}:{os.path.normpath(path)}"
    
    def _is_cache_valid(self, key: str) -> bool:
        """
        Check if cache is still valid
        
        Args:
            key: Cache key
            
        Returns:
            Whether cache is valid
        """
        # Cache is valid if it exists and hasn't expired
        if key in self._cache and key in self._cache_timestamp:
            age = time.time() - self._cache_timestamp[key]
            return age < self._cache_expiration
        return False
    
    def clear_cache(self) -> None:
        """Clear the glob cache"""
        self._cache = {}
        self._cache_timestamp = {}
    
    def find(
        self, 
        pattern: str, 
        path: Optional[str] = None, 
        sort_by_modified: bool = True,
        use_cache: bool = True
    ) -> List[str]:
        """
        Find files using glob pattern with efficient caching
        
        Args:
            pattern: Glob pattern to match (e.g., "**/*.py")
            path: Directory to search in (defaults to base_path)
            sort_by_modified: Whether to sort results by modification time
            use_cache: Whether to use cached results
            
        Returns:
            List of matching file paths
            
        Raises:
            FileAccessError: If path is not safe to access
        """
        search_path = path or self.base_path
        
        # Ensure path exists
        if not os.path.exists(search_path):
            return []
        
        # Check path safety
        self._check_path_safety(search_path)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(pattern, search_path)
            if self._is_cache_valid(cache_key):
                matches = self._cache[cache_key]
                
                # Filter out files that no longer exist
                matches = [m for m in matches if os.path.exists(m)]
                
                # Sort if needed
                if sort_by_modified:
                    matches.sort(key=lambda m: os.path.getmtime(m), reverse=True)
                    
                return matches
        
        # Create absolute pattern
        if os.path.isabs(pattern):
            absolute_pattern = pattern
        else:
            absolute_pattern = os.path.join(search_path, pattern)
        
        # Find files
        try:
            matches = glob.glob(absolute_pattern, recursive=True)
            
            # Only return files, not directories
            matches = [m for m in matches if os.path.isfile(m)]
            
            # Cache the results
            if use_cache:
                cache_key = self._get_cache_key(pattern, search_path)
                self._cache[cache_key] = matches.copy()
                self._cache_timestamp[cache_key] = time.time()
            
            # Sort by modification time if requested
            if sort_by_modified:
                matches.sort(key=os.path.getmtime, reverse=True)
                
            return matches
        except Exception as e:
            logger.error(f"Error finding files with pattern {pattern}: {e}")
            raise FileAccessError(f"Error finding files: {str(e)}")
    
    def find_with_ignore(
        self, 
        pattern: str, 
        ignore_patterns: List[str], 
        path: Optional[str] = None, 
        sort_by_modified: bool = True,
        use_cache: bool = True
    ) -> List[str]:
        """
        Find files using glob pattern with ignore patterns
        
        Args:
            pattern: Glob pattern to match
            ignore_patterns: List of glob patterns to ignore
            path: Directory to search in
            sort_by_modified: Whether to sort results by modification time
            use_cache: Whether to use cached results
            
        Returns:
            List of matching file paths
            
        Raises:
            FileAccessError: If path is not safe to access
        """
        search_path = path or self.base_path
        
        # Check cache if enabled
        if use_cache:
            # Create a combined cache key
            ignore_str = ":".join(sorted(ignore_patterns))
            cache_key = self._get_cache_key(f"{pattern}:{ignore_str}", search_path)
            
            if self._is_cache_valid(cache_key):
                matches = self._cache[cache_key]
                
                # Filter out files that no longer exist
                matches = [m for m in matches if os.path.exists(m)]
                
                # Sort if needed
                if sort_by_modified:
                    matches.sort(key=lambda m: os.path.getmtime(m), reverse=True)
                    
                return matches
        
        # Get initial matches
        matches = self.find(pattern, search_path, False, use_cache)
        
        # Use a set to efficiently apply ignore patterns
        ignore_files = set()
        
        # Apply ignore patterns in parallel for better performance
        with ThreadPoolExecutor(max_workers=min(8, len(ignore_patterns) or 1)) as executor:
            future_to_pattern = {
                executor.submit(self.find, ignore, search_path, False, use_cache): ignore
                for ignore in ignore_patterns
            }
            
            for future in as_completed(future_to_pattern):
                ignore_files.update(future.result())
        
        # Filter out ignored files
        matches = [m for m in matches if m not in ignore_files]
        
        # Cache the results
        if use_cache:
            ignore_str = ":".join(sorted(ignore_patterns))
            cache_key = self._get_cache_key(f"{pattern}:{ignore_str}", search_path)
            self._cache[cache_key] = matches.copy()
            self._cache_timestamp[cache_key] = time.time()
        
        # Sort if requested
        if sort_by_modified:
            matches.sort(key=os.path.getmtime, reverse=True)
            
        return matches

    def is_binary_file(self, file_path: str, sample_size: int = 4096) -> bool:
        """
        Check if a file is binary
        
        Args:
            file_path: Path to file
            sample_size: Number of bytes to check
            
        Returns:
            Whether the file is binary
        """
        # First check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            if not mime_type.startswith(('text/', 'application/json', 'application/xml')):
                return True
                
        # If MIME type check is inconclusive, check for null bytes
        try:
            with open(file_path, 'rb') as f:
                return b'\0' in f.read(sample_size)
        except Exception:
            # If we can't open the file, assume it's binary
            return True


class GrepTool:
    """Fast content search tool with improved performance and safety"""
    
    def __init__(
        self, 
        base_path: Optional[str] = None, 
        glob_finder: Optional[GlobFinder] = None,
        max_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    ):
        """
        Initialize with optional base path
        
        Args:
            base_path: Base directory for relative paths
            glob_finder: GlobFinder instance for file search
            max_size_mb: Maximum file size to search (in MB)
        """
        self.base_path = base_path or os.getcwd()
        self.glob_finder = glob_finder or GlobFinder(self.base_path)
        self.max_size_mb = max_size_mb
        
        # Simple cache for compiled regex patterns
        self._regex_cache = {}
        self._max_cache_size = 100
    
    def _get_compiled_regex(self, pattern: str, case_sensitive: bool = True) -> Pattern:
        """
        Get a compiled regex pattern from cache or compile it
        
        Args:
            pattern: Regex pattern
            case_sensitive: Whether the pattern is case sensitive
            
        Returns:
            Compiled regex pattern
            
        Raises:
            re.error: If pattern is invalid
        """
        # Create cache key
        cache_key = f"{pattern}:{case_sensitive}"
        
        # Check cache
        if cache_key in self._regex_cache:
            return self._regex_cache[cache_key]
        
        # Compile new pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        regex = re.compile(pattern, flags)
        
        # Manage cache size
        if len(self._regex_cache) >= self._max_cache_size:
            # Remove a random item
            self._regex_cache.pop(next(iter(self._regex_cache)))
        
        # Add to cache
        self._regex_cache[cache_key] = regex
        
        return regex
    
    def search(
        self, 
        pattern: str, 
        include: Optional[str] = None, 
        path: Optional[str] = None, 
        max_size_mb: Optional[int] = None, 
        sort_by_modified: bool = True,
        case_sensitive: bool = True,
        max_matches_per_file: int = 1000
    ) -> Dict[str, List[Tuple[int, str]]]:
        """
        Search for pattern in file contents
        
        Args:
            pattern: Regular expression pattern to search for
            include: File pattern to include (e.g., "*.py")
            path: Directory to search in
            max_size_mb: Maximum file size to search in MB
            sort_by_modified: Whether to sort results by modification time
            case_sensitive: Whether the search is case sensitive
            max_matches_per_file: Maximum number of matches per file
            
        Returns:
            Dict of {file_path: [(line_number, line_content), ...]}
            
        Raises:
            FileAccessError: If path is not safe to access
            ValueError: If pattern is invalid
        """
        search_path = path or self.base_path
        
        # Use provided max_size or default
        max_size = (max_size_mb or self.max_size_mb) * 1024 * 1024  # Convert to bytes
        
        # Find files to search
        try:
            if include:
                files = self.glob_finder.find(include, search_path, sort_by_modified)
            else:
                files = self.glob_finder.find("**/*", search_path, sort_by_modified)
        except FileAccessError as e:
            raise FileAccessError(f"Error finding files to search: {str(e)}")
        
        # Compile pattern for better performance
        try:
            regex = self._get_compiled_regex(pattern, case_sensitive)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {str(e)}")
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel searching
        with ThreadPoolExecutor(max_workers=min(os.cpu_count() or 1, 8)) as executor:
            future_to_file = {
                executor.submit(
                    self.search_single_file, 
                    file_path, 
                    pattern, 
                    case_sensitive,
                    max_size,
                    max_matches_per_file
                ): file_path
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    matches = future.result()
                    if matches:
                        results[file_path] = matches
                except Exception as e:
                    logger.debug(f"Error searching {file_path}: {e}")
        
        return results
    
    def search_single_file(
        self, 
        file_path: str, 
        pattern: str, 
        case_sensitive: bool = True,
        max_size: Optional[int] = None,
        max_matches: int = 1000
    ) -> List[Tuple[int, str]]:
        """
        Search for pattern in a single file
        
        Args:
            file_path: Path to file
            pattern: Regular expression pattern to search for
            case_sensitive: Whether the search is case sensitive
            max_size: Maximum file size in bytes
            max_matches: Maximum number of matches to return
            
        Returns:
            List of (line_number, line_content) tuples
            
        Raises:
            FileAccessError: If file is not safe to access
            ValueError: If pattern is invalid
        """
        if not os.path.isfile(file_path):
            return []
        
        try:
            # Check file size
            size = os.path.getsize(file_path)
            if max_size and size > max_size:
                logger.debug(f"Skipping large file: {file_path} ({size} bytes)")
                return []
            
            # Check if file is binary
            if self.glob_finder.is_binary_file(file_path):
                logger.debug(f"Skipping binary file: {file_path}")
                return []
            
            # Compile pattern if not already compiled
            if isinstance(pattern, str):
                regex = self._get_compiled_regex(pattern, case_sensitive)
            else:
                regex = pattern
            
            matches = []
            
            # Search file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        matches.append((i, line.rstrip()))
                        if len(matches) >= max_matches:
                            # Add a note that there may be more matches
                            matches.append((0, f"Note: Limited to {max_matches} matches"))
                            break
            
            return matches
            
        except (IOError, OSError) as e:
            logger.debug(f"Error reading {file_path}: {e}")
            raise FileAccessError(f"Error reading {file_path}: {str(e)}")
        except re.error as e:
            logger.debug(f"Invalid regex pattern: {e}")
            raise ValueError(f"Invalid regex pattern: {str(e)}")
        except Exception as e:
            logger.debug(f"Error searching {file_path}: {e}")
            return []


class CodeEditor:
    """Tool for precise code editing with validation and backup"""
    
    def __init__(
        self, 
        backup: bool = True,
        backup_dir: Optional[str] = None,
        max_backup_count: int = 10
    ):
        """
        Initialize code editor
        
        Args:
            backup: Whether to create backups
            backup_dir: Directory for backups (defaults to .fei_backups in each dir)
            max_backup_count: Maximum number of backups per file
        """
        self.backup = backup
        self.backup_dir = backup_dir
        self.max_backup_count = max_backup_count
    
    def _get_backup_path(self, file_path: str) -> str:
        """
        Get backup path for a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Backup path
        """
        if self.backup_dir:
            # Use specified backup directory
            os.makedirs(self.backup_dir, exist_ok=True)
            filename = os.path.basename(file_path)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            return os.path.join(self.backup_dir, f"{filename}.{timestamp}.bak")
        else:
            # Use directory-specific backup directory
            file_dir = os.path.dirname(os.path.abspath(file_path))
            backup_dir = os.path.join(file_dir, ".fei_backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            return os.path.join(backup_dir, f"{filename}.{timestamp}.bak")
    
    def _cleanup_backups(self, file_path: str) -> None:
        """
        Clean up old backups
        
        Args:
            file_path: Path to file
        """
        if not self.backup:
            return
            
        try:
            # Get backup directory
            if self.backup_dir:
                backup_dir = self.backup_dir
            else:
                file_dir = os.path.dirname(os.path.abspath(file_path))
                backup_dir = os.path.join(file_dir, ".fei_backups")
            
            if not os.path.exists(backup_dir):
                return
                
            # Find backups for this file
            filename = os.path.basename(file_path)
            backups = [
                os.path.join(backup_dir, f)
                for f in os.listdir(backup_dir)
                if f.startswith(filename) and f.endswith(".bak")
            ]
            
            # Sort by modification time (oldest first)
            backups.sort(key=os.path.getmtime)
            
            # Remove oldest backups if we have too many
            while len(backups) > self.max_backup_count:
                os.remove(backups[0])
                backups.pop(0)
                
        except Exception as e:
            logger.warning(f"Error cleaning up backups: {e}")
    
    def _make_backup(self, file_path: str) -> Optional[str]:
        """
        Create a backup of a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Backup path or None if backup failed
        """
        if not self.backup:
            return None
            
        try:
            if not os.path.exists(file_path):
                return None
                
            backup_path = self._get_backup_path(file_path)
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups
            self._cleanup_backups(file_path)
            
            return backup_path
        except Exception as e:
            logger.warning(f"Error creating backup of {file_path}: {e}")
            return None
    
    def edit_file(self, file_path: str, old_string: str, new_string: str) -> Tuple[bool, str, Optional[str]]:
        """
        Edit a file by replacing old_string with new_string
        
        Args:
            file_path: Path to file
            old_string: String to replace (must match exactly)
            new_string: Replacement string
            
        Returns:
            Tuple of (success, message, backup_path)
            
        Raises:
            FileAccessError: If file cannot be accessed
            ValueError: If old_string is not unique
        """
        if not os.path.isfile(file_path):
            raise FileAccessError(f"File not found: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Check if old_string exists
            if old_string not in content:
                return False, f"String not found in {file_path}", None
            
            # Count occurrences
            count = content.count(old_string)
            if count > 1:
                raise ValueError(f"Found {count} occurrences of the string in {file_path}. Must be unique.")
            
            # Create backup if needed
            backup_path = self._make_backup(file_path)
            
            # Replace string
            new_content = content.replace(old_string, new_string, 1)
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True, f"Successfully edited {file_path}", backup_path
        
        except (IOError, OSError) as e:
            logger.error(f"Error editing {file_path}: {e}")
            raise FileAccessError(f"Error editing {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error editing {file_path}: {e}")
            return False, f"Error editing {file_path}: {str(e)}", None
    
    def create_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        """
        Create a new file with content
        
        Args:
            file_path: Path to file
            content: File content
            
        Returns:
            Tuple of (success, message)
            
        Raises:
            FileAccessError: If file cannot be created
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Create file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"Successfully created {file_path}"
        
        except (IOError, OSError) as e:
            logger.error(f"Error creating {file_path}: {e}")
            raise FileAccessError(f"Error creating {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error creating {file_path}: {e}")
            return False, f"Error creating {file_path}: {str(e)}"
    
    def replace_file(self, file_path: str, content: str) -> Tuple[bool, str, Optional[str]]:
        """
        Replace file content
        
        Args:
            file_path: Path to file
            content: New file content
            
        Returns:
            Tuple of (success, message, backup_path)
            
        Raises:
            FileAccessError: If file cannot be accessed
        """
        try:
            # Create backup if file exists and backup is enabled
            backup_path = None
            if os.path.isfile(file_path):
                backup_path = self._make_backup(file_path)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Write content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, f"Successfully replaced {file_path}", backup_path
        
        except (IOError, OSError) as e:
            logger.error(f"Error replacing {file_path}: {e}")
            raise FileAccessError(f"Error replacing {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error replacing {file_path}: {e}")
            return False, f"Error replacing {file_path}: {str(e)}", None
    
    def regex_replace(
        self, 
        file_path: str, 
        pattern: str, 
        replacement: str, 
        validate: bool = True, 
        max_retries: int = 3, 
        validators: Optional[List[str]] = None
    ) -> Tuple[bool, str, int]:
        """
        Replace text using regex pattern with validation
        
        Args:
            file_path: Path to file
            pattern: Regular expression pattern
            replacement: Replacement string or expression
            validate: Whether to validate the resulting code
            max_retries: Maximum number of retry attempts if validation fails
            validators: List of validators to use
            
        Returns:
            Tuple of (success, message, count)
            
        Raises:
            FileAccessError: If file cannot be accessed
            ValueError: If pattern is invalid
        """
        if not os.path.isfile(file_path):
            raise FileAccessError(f"File not found: {file_path}")
        
        # Default validators based on file extension
        if validators is None:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.py':
                validators = ["ast"]
            elif ext in ['.js', '.jsx', '.ts', '.tsx']:
                validators = ["esprima"]
            else:
                validators = []
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                original_content = f.read()
            
            # Compile regex
            try:
                regex = re.compile(pattern, re.MULTILINE)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {str(e)}")
            
            # Apply replacement
            new_content, count = regex.subn(replacement, original_content)
            
            if count == 0:
                return False, f"No matches found in {file_path}", 0
            
            # Create backup
            backup_path = self._make_backup(file_path)
            
            # If validation is enabled, perform checks before saving
            if validate and validators:
                # Validate the new content before writing
                validation_result, validation_error = self._validate_code(file_path, new_content, validators)
                
                if not validation_result:
                    retry_count = 0
                    while retry_count < max_retries:
                        retry_count += 1
                        logger.warning(f"Validation failed: {validation_error}. Retry {retry_count}/{max_retries}")
                        
                        # Return validation error
                        return False, f"Validation failed: {validation_error}", 0
            
            # Write updated content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            return True, f"Successfully replaced {count} occurrences in {file_path}", count
        
        except (IOError, OSError) as e:
            logger.error(f"Error replacing in {file_path}: {e}")
            raise FileAccessError(f"Error replacing in {file_path}: {str(e)}")
        except ValueError as e:
            # Re-raise ValueError (for invalid regex)
            raise
        except Exception as e:
            logger.error(f"Error replacing in {file_path}: {e}")
            return False, f"Error replacing in {file_path}: {str(e)}", 0
    
    def _validate_code(self, file_path: str, content: str, validators: List[str]) -> Tuple[bool, str]:
        """
        Validate code using specified validators
        
        Args:
            file_path: Path to file (used to determine language)
            content: Code content to validate
            validators: List of validators to use
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        for validator in validators:
            if validator == "ast" and ext == '.py':
                # Python AST validation
                try:
                    import ast
                    ast.parse(content)
                except SyntaxError as e:
                    return False, f"Python syntax error: {str(e)}"
                
            elif validator == "esprima" and ext in ['.js', '.jsx', '.ts', '.tsx']:
                # JavaScript/TypeScript validation
                try:
                    # Try to import esprima
                    try:
                        import esprima
                        esprima.parseScript(content)
                    except ImportError:
                        # Try to import a different JS parser
                        try:
                            import acorn
                            acorn.parse(content)
                        except ImportError:
                            logger.warning("JS parser not installed, skipping JavaScript validation")
                except Exception as e:
                    return False, f"JavaScript syntax error: {str(e)}"
                    
            elif validator == "pylint" and ext == '.py':
                # Use pylint for more advanced Python validation
                try:
                    import pylint.lint
                    import io
                    from pylint.reporters.text import TextReporter
                    
                    output = io.StringIO()
                    reporter = TextReporter(output)
                    
                    # Write content to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(content)
                    
                    try:
                        # Run pylint on the temporary file
                        pylint.lint.Run([temp_file_path], reporter=reporter, exit=False)
                        
                        # Check if there are errors
                        lint_output = output.getvalue()
                        if "error" in lint_output.lower():
                            return False, f"Pylint errors: {lint_output}"
                    finally:
                        # Clean up
                        try:
                            os.remove(temp_file_path)
                        except:
                            pass
                        
                except ImportError:
                    logger.warning("Pylint not installed, skipping advanced Python validation")
                except Exception as e:
                    logger.warning(f"Error during pylint validation: {str(e)}")
            
            elif validator == "flake8" and ext == '.py':
                # Use flake8 for Python style validation
                try:
                    import subprocess
                    
                    # Write content to a temporary file
                    with tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False) as temp_file:
                        temp_file_path = temp_file.name
                        temp_file.write(content)
                    
                    try:
                        # Run flake8 on the temporary file
                        result = subprocess.run(['flake8', temp_file_path], capture_output=True, text=True)
                        
                        # Check if there are errors
                        if result.returncode != 0:
                            return False, f"Flake8 errors: {result.stdout}"
                    finally:
                        # Clean up
                        try:
                            os.remove(temp_file_path)
                        except:
                            pass
                        
                except ImportError:
                    logger.warning("Flake8 not installed, skipping Python style validation")
                except Exception as e:
                    logger.warning(f"Error during flake8 validation: {str(e)}")
        
        # All validations passed
        return True, ""


class FileViewer:
    """Tool for viewing file contents with safety checks"""
    
    def __init__(self, max_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB):
        """
        Initialize file viewer
        
        Args:
            max_size_mb: Maximum file size in MB
        """
        self.max_size_mb = max_size_mb
        self.glob_finder = GlobFinder()
    
    def view(
        self, 
        file_path: str, 
        limit: Optional[int] = None, 
        offset: int = 0
    ) -> Tuple[bool, str, List[str]]:
        """
        View file contents
        
        Args:
            file_path: Path to file
            limit: Maximum number of lines to read
            offset: Line number to start from (0-indexed)
            
        Returns:
            Tuple of (success, message, lines)
            
        Raises:
            FileAccessError: If file cannot be accessed
        """
        if not os.path.isfile(file_path):
            raise FileAccessError(f"File not found: {file_path}")
        
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = self.max_size_mb * 1024 * 1024
            
            if file_size > max_size:
                raise FileAccessError(
                    f"File too large: {file_path} ({file_size / (1024*1024):.2f} MB)"
                )
            
            # Check if binary
            if self.glob_finder.is_binary_file(file_path):
                return False, f"Binary file detected: {file_path}", []
            
            # Read file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                if offset > 0:
                    # Skip lines
                    for _ in range(offset):
                        next(f, None)
                
                if limit is not None:
                    # Read limited number of lines
                    lines = [line.rstrip('\n') for line in f.readlines()[:limit]]
                else:
                    # Read all lines
                    lines = [line.rstrip('\n') for line in f]
            
            return True, f"Successfully read {len(lines)} lines from {file_path}", lines
        
        except (IOError, OSError) as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise FileAccessError(f"Error reading {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return False, f"Error reading {file_path}: {str(e)}", []
    
    def count_lines(self, file_path: str) -> Tuple[bool, int]:
        """
        Count lines in a file
        
        Args:
            file_path: Path to file
            
        Returns:
            Tuple of (success, line_count)
            
        Raises:
            FileAccessError: If file cannot be accessed
        """
        if not os.path.isfile(file_path):
            raise FileAccessError(f"File not found: {file_path}")
        
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            max_size = self.max_size_mb * 1024 * 1024
            
            if file_size > max_size:
                return False, 0
            
            # Check if binary
            if self.glob_finder.is_binary_file(file_path):
                return False, 0
            
            # Count lines efficiently
            with open(file_path, 'rb') as f:
                # Fast counting method using bytearray
                chunk_size = 1024 * 1024  # 1MB chunks
                line_count = 0
                buffer = bytearray(chunk_size)
                
                while True:
                    bytes_read = f.readinto(buffer)
                    if bytes_read == 0:
                        break
                    line_count += buffer[:bytes_read].count(b'\n')
                
                # Check if the file doesn't end with a newline
                f.seek(0, os.SEEK_END)
                if file_size > 0:
                    f.seek(file_size - 1, os.SEEK_SET)
                    if f.read(1) != b'\n':
                        line_count += 1
            
            return True, line_count
        
        except (IOError, OSError) as e:
            logger.error(f"Error counting lines in {file_path}: {e}")
            raise FileAccessError(f"Error counting lines in {file_path}: {str(e)}")
        except Exception:
            return False, 0
    
    def get_hash(self, file_path: str, algorithm: str = 'sha256') -> str:
        """
        Get a hash of file contents
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm (md5, sha1, sha256, sha512)
            
        Returns:
            File hash
            
        Raises:
            FileAccessError: If file cannot be accessed
            ValueError: If algorithm is invalid
        """
        if not os.path.isfile(file_path):
            raise FileAccessError(f"File not found: {file_path}")
        
        # Choose hash algorithm
        if algorithm == 'md5':
            hash_obj = hashlib.md5()
        elif algorithm == 'sha1':
            hash_obj = hashlib.sha1()
        elif algorithm == 'sha256':
            hash_obj = hashlib.sha256()
        elif algorithm == 'sha512':
            hash_obj = hashlib.sha512()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        try:
            # Calculate hash
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            
            return hash_obj.hexdigest()
        
        except (IOError, OSError) as e:
            logger.error(f"Error hashing {file_path}: {e}")
            raise FileAccessError(f"Error hashing {file_path}: {str(e)}")


class DirectoryExplorer:
    """Tool for listing directory contents"""
    
    def __init__(self, glob_finder: Optional[GlobFinder] = None):
        """
        Initialize directory explorer
        
        Args:
            glob_finder: GlobFinder instance for pattern matching
        """
        self.glob_finder = glob_finder or GlobFinder()
    
    def list_directory(
        self, 
        path: str, 
        ignore: Optional[List[str]] = None,
        include_hidden: bool = False,
        recursive: bool = False
    ) -> Tuple[bool, str, Dict[str, Union[List[str], Dict[str, Any]]]]:
        """
        List directory contents
        
        Args:
            path: Directory path
            ignore: List of glob patterns to ignore
            include_hidden: Whether to include hidden files
            recursive: Whether to list subdirectories recursively
            
        Returns:
            Tuple of (success, message, content)
            
        Raises:
            FileAccessError: If directory cannot be accessed
        """
        if not os.path.isdir(path):
            raise FileAccessError(f"Directory not found: {path}")
        
        try:
            # Build result
            result = {"dirs": [], "files": []}
            
            if recursive:
                # Use os.walk for recursive listing
                for root, dirs, files in os.walk(path):
                    # Skip hidden directories if needed
                    if not include_hidden:
                        dirs[:] = [d for d in dirs if not d.startswith('.')]
                    
                    # Get relative paths
                    rel_root = os.path.relpath(root, path)
                    if rel_root == '.':
                        rel_root = ''
                    
                    # Add directories
                    for d in dirs:
                        dir_path = os.path.join(rel_root, d)
                        if not is_ignored(dir_path, ignore):
                            result["dirs"].append(dir_path)
                    
                    # Add files
                    for f in files:
                        # Skip hidden files if needed
                        if not include_hidden and f.startswith('.'):
                            continue
                            
                        file_path = os.path.join(rel_root, f)
                        if not is_ignored(file_path, ignore):
                            result["files"].append(file_path)
            else:
                # List directory contents
                entries = os.listdir(path)
                
                # Process ignore patterns
                ignored_entries = set()
                if ignore:
                    for pattern in ignore:
                        pattern_path = os.path.join(path, pattern)
                        ignored = self.glob_finder.find(pattern, path)
                        ignored_entries.update([os.path.basename(i) for i in ignored])
                
                # Separate directories and files
                for entry in entries:
                    # Skip hidden entries if needed
                    if not include_hidden and entry.startswith('.'):
                        continue
                        
                    if entry in ignored_entries:
                        continue
                    
                    entry_path = os.path.join(path, entry)
                    if os.path.isdir(entry_path):
                        result["dirs"].append(entry)
                    else:
                        result["files"].append(entry)
            
            # Sort results
            result["dirs"].sort()
            result["files"].sort()
            
            return True, f"Listed {len(result['dirs'])} directories and {len(result['files'])} files", result
            
        except (IOError, OSError) as e:
            logger.error(f"Error listing directory {path}: {e}")
            raise FileAccessError(f"Error listing directory {path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error listing directory {path}: {e}")
            return False, f"Error listing directory {path}: {str(e)}", {"dirs": [], "files": []}

def is_ignored(path: str, ignore_patterns: Optional[List[str]]) -> bool:
    """
    Check if a path matches any ignore pattern
    
    Args:
        path: Path to check
        ignore_patterns: List of glob patterns to ignore
        
    Returns:
        Whether the path matches any pattern
    """
    if not ignore_patterns:
        return False
        
    for pattern in ignore_patterns:
        if glob.fnmatch.fnmatch(path, pattern):
            return True
    
    return False


class SystemInfo:
    """Tool for getting system information"""
    
    @staticmethod
    def get_os_info() -> Dict[str, Any]:
        """
        Get operating system information
        
        Returns:
            Dictionary of OS information
        """
        import platform
        
        # Get OS info
        result = {
            "system": platform.system(),
            "version": platform.version(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "node": platform.node(),
            "python": platform.python_version()
        }
        
        # Get more detailed info based on platform
        if result["system"] == "Linux":
            try:
                # Try to get distribution info
                import distro
                result["distribution"] = distro.name(pretty=True)
                result["distribution_version"] = distro.version()
                result["distribution_id"] = distro.id()
            except ImportError:
                # Fall back to platform
                result["distribution"] = platform.linux_distribution() if hasattr(platform, 'linux_distribution') else "Unknown"
                
        return result
    
    @staticmethod
    def get_memory_info() -> Dict[str, Any]:
        """
        Get memory information
        
        Returns:
            Dictionary of memory information
        """
        result = {}
        
        try:
            import psutil
            
            # Get memory info
            vm = psutil.virtual_memory()
            result = {
                "total": vm.total,
                "available": vm.available,
                "used": vm.used,
                "percent": vm.percent,
                "total_gb": round(vm.total / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "used_gb": round(vm.used / (1024**3), 2)
            }
        except ImportError:
            # Fall back to basic info
            result = {"error": "psutil not available"}
            
        return result
    
    @staticmethod
    def get_disk_info(path: str = ".") -> Dict[str, Any]:
        """
        Get disk information for a path
        
        Args:
            path: Path to check
            
        Returns:
            Dictionary of disk information
        """
        result = {}
        
        try:
            import psutil
            
            # Get disk usage for path
            usage = psutil.disk_usage(path)
            result = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": usage.percent,
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2)
            }
        except ImportError:
            # Fall back to basic info
            import shutil
            usage = shutil.disk_usage(path)
            result = {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "percent": round((usage.used / usage.total) * 100, 1),
                "total_gb": round(usage.total / (1024**3), 2),
                "used_gb": round(usage.used / (1024**3), 2),
                "free_gb": round(usage.free / (1024**3), 2)
            }
            
        return result


class ShellRunner:
    """Tool for executing shell commands with advanced security and process management"""
    
    # Global allowlist of safe commands
    ALLOWED_COMMANDS = {
        # File system (non-destructive)
        "ls", "find", "cat", "head", "tail", "less", "more", "grep", "tree", "stat", "du",
        "file", "whereis", "which", "locate", "pwd", "dirname", "basename", "realpath",
        
        # File management (potentially destructive but needed)
        "mkdir", "touch", "rm", "cp", "mv", "ln", "chmod", "chown", "tar", "zip", "unzip",
        "gzip", "gunzip", "bzip2", "bunzip2", "rsync",
        
        # Process management
        "ps", "top", "htop", "kill", "pkill", "pgrep", "nice", "renice", "time",
        
        # Network (read-only)
        "ping", "traceroute", "dig", "host", "nslookup", "netstat", "ss", "ifconfig",
        "ip", "arp", "route", "wget", "curl",
        
        # System info
        "uname", "uptime", "free", "df", "mount", "lsblk", "lsusb", "lspci", "dmidecode",
        "getconf", "ulimit", "env", "printenv", "hostname", "date", "cal",
        
        # Text processing
        "echo", "cat", "sort", "uniq", "tr", "sed", "awk", "cut", "paste", "join",
        "wc", "fmt", "tee", "md5sum", "sha1sum", "sha256sum", "diff", "cmp",
        
        # Package management
        "apt", "apt-get", "dpkg", "yum", "dnf", "rpm", "pip", "npm", "gem", "composer",
        
        # Development
        "gcc", "clang", "make", "cmake", "python", "python3", "node", "npm", "git", "svn",
        "javac", "java", "go", "rust", "cargo", "mvn",
        
        # Utilities
        "xargs", "watch", "yes", "sleep", "timeout", "printf", "open", "bc", "hexdump", "xxd"
    }
    
    # Denylist of dangerous commands
    DENIED_COMMANDS = {
        # Dangerous system commands
        "rm -rf /", "dd", "mkfs", "shutdown", "reboot", "poweroff",
        "passwd", "sudo", "su", "chroot", "crontab", "at",
        
        # Network attacks
        "nc", "ncat", "telnet", "ssh", "nmap", "tcpdump", "wireshark",
        
        # Dangerous scripting
        "eval", "exec", "`", "$(",
        
        # Web access
        "wget", "curl", "lynx", "w3m",
        
        # File system manipulation
        ">", ">>", "|"
    }
    
    def __init__(
        self, 
        default_timeout: int = DEFAULT_COMMAND_TIMEOUT,
        max_output_size: int = MAX_OUTPUT_SIZE,
        enforce_allowlist: bool = True,
        sandbox: bool = True
    ):
        """
        Initialize shell runner
        
        Args:
            default_timeout: Default command timeout in seconds
            max_output_size: Maximum output size in characters
            enforce_allowlist: Whether to enforce the command allowlist
            sandbox: Whether to use sandboxing
        """
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.enforce_allowlist = enforce_allowlist
        self.sandbox = sandbox
        
        # Process tracking
        self.running_processes = {}
        self._lock = threading.RLock()
        
        # Register cleanup on exit
        import atexit
        atexit.register(self.cleanup_processes)
    
    def cleanup_processes(self) -> None:
        """Clean up any running processes"""
        with self._lock:
            for pid, process in list(self.running_processes.items()):
                try:
                    logger.info(f"Terminating process {pid}")
                    # Try to terminate the process group
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    
                    # Wait briefly for termination
                    for _ in range(5):
                        if process.poll() is not None:
                            break
                        time.sleep(0.1)
                    
                    # Force kill if still running
                    if process.poll() is None:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception as e:
                    logger.warning(f"Failed to terminate process {pid}: {e}")
                
                self.running_processes.pop(pid, None)
    
    def _is_allowed_command(self, command: str) -> Tuple[bool, str]:
        """
        Check if a command is allowed
        
        Args:
            command: Command to check
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        if not self.enforce_allowlist:
            return True, ""
        
        # Check for empty command
        if not command.strip():
            return False, "Empty command"
        
        # Check against denylist first
        for denied in self.DENIED_COMMANDS:
            if denied in command:
                return False, f"Command contains denied pattern: {denied}"
        
        # Check command against allowlist
        # Extract the main command (before any arguments)
        main_command = command.split()[0] if command.split() else ""
        
        # Handle command paths (e.g., /usr/bin/ls)
        if "/" in main_command:
            main_command = os.path.basename(main_command)
        
        # Check if main command is allowed
        if main_command not in self.ALLOWED_COMMANDS:
            return False, f"Command not in allowlist: {main_command}"
        
        return True, ""
    
    def is_interactive_command(self, command: str) -> bool:
        """
        Check if a command is likely to be interactive
        
        Args:
            command: Command to check
            
        Returns:
            Whether the command is likely to be interactive
        """
        # List of keywords that suggest interactive applications
        interactive_keywords = [
            # GUI applications
            "gui", "gtk", "gnome", "x11", "wayland", "qt", "tkinter", "curses",
            # Interactive tools
            "vim", "nano", "emacs", "less", "more", "top", "htop",
            # Interpreters and REPLs
            "python", "ipython", "jupyter", "node", "irb", "pry",
            # Development servers
            "npm start", "npm run", "flask run", "rails server",
            # Games
            "game", "pygame", "play", 
        ]
        
        # Check if any keyword is in the command
        return any(keyword in command.lower() for keyword in interactive_keywords)
    
    def run_command(
        self, 
        command: str, 
        timeout: Optional[int] = None, 
        background: Optional[bool] = None,
        capture_stderr: bool = True,
        env: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Run a shell command with extensive safety features
        
        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds (overrides default)
            background: Force running in background (for interactive commands)
            capture_stderr: Whether to capture stderr
            env: Environment variables
            
        Returns:
            Dict with stdout, stderr, exit_code, and error (if any)
            
        Raises:
            CommandExecutionError: If command execution fails
            ValueError: If command is not allowed
        """
        if not command:
            return {
                "success": False,
                "error": "Command is required",
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }
        
        # Check if command is allowed
        is_allowed, reason = self._is_allowed_command(command)
        if not is_allowed:
            return {
                "success": False,
                "error": f"Command not allowed: {reason}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }
        
        # Set timeout
        cmd_timeout = timeout or self.default_timeout
        
        # Auto-detect if command should run in background
        if background is None:
            background = self.is_interactive_command(command)
        
        # For interactive commands, modify to prevent blocking
        if background:
            logger.info(f"Running command in background mode: {command}")
            
            try:
                with self._lock:
                    # Create process
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        start_new_session=True,  # Create a new process group
                        env=env or os.environ.copy()
                    )
                    
                    # Store process information for cleanup
                    pid = process.pid
                    self.running_processes[pid] = process
                
                # Set up a timer to terminate the process after the timeout
                def terminate_after_timeout():
                    time.sleep(cmd_timeout)
                    with self._lock:
                        if pid in self.running_processes:
                            logger.info(f"Terminating background process {pid} after timeout")
                            try:
                                os.killpg(os.getpgid(pid), signal.SIGTERM)
                                
                                # Wait briefly for termination
                                for _ in range(5):
                                    if process.poll() is not None:
                                        break
                                    time.sleep(0.1)
                                
                                # Force kill if still running
                                if process.poll() is None:
                                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                            except Exception as e:
                                logger.warning(f"Failed to terminate process {pid}: {e}")
                            
                            self.running_processes.pop(pid, None)
                
                # Start the termination timer in a separate thread
                    timer_thread = threading.Thread(target=terminate_after_timeout, daemon=True)
                    timer_thread.start()

                # Return immediately after launching the background process
                return {
                    "success": True, # Indicates process started successfully
                    "stdout": f"Process started in background (PID: {pid}).",
                    "stderr": "", # No immediate stderr captured
                    "exit_code": None, # Exit code is unknown for background process
                    "background": True,
                    "pid": pid,
                    "note": f"Process running in background. Will be terminated after {cmd_timeout} seconds if not finished."
                }

            except Exception as e:
                logger.error(f"Error executing background command: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1
                }
        
        # For non-interactive commands, use subprocess.run
        try:
            # Run the command with timeout
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=cmd_timeout,
                env=env or os.environ.copy()
            )
            
            # Get output
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode
            
            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... [output truncated]"
            
            return {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }
        
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {cmd_timeout} seconds",
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }
        
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }


# Create instances for easier usage
glob_finder = GlobFinder()
grep_tool = GrepTool()
code_editor = CodeEditor()
file_viewer = FileViewer()
directory_explorer = DirectoryExplorer()
shell_runner = ShellRunner()
system_info = SystemInfo()


def create_code_tools(registry):
    """
    Register code tools with the given registry
    
    Args:
        registry: Tool registry to add tools to
    """
    from fei.tools.definitions import TOOL_DEFINITIONS
    from fei.tools.handlers import (
        glob_tool_handler,
        grep_tool_handler,
        view_handler,
        edit_handler,
        replace_handler,
        ls_handler,
        regex_edit_handler,
        batch_glob_handler,
        find_in_files_handler,
        smart_search_handler,
        repo_map_handler,
        repo_summary_handler,
        repo_deps_handler,
        shell_handler
    )
    
    # Register tools
    registry.register_tool(
        name="GlobTool",
        description=TOOL_DEFINITIONS[0]["description"],
        input_schema=TOOL_DEFINITIONS[0]["input_schema"],
        handler_func=glob_tool_handler,
        tags=["files", "search"]
    )
    
    registry.register_tool(
        name="GrepTool",
        description=TOOL_DEFINITIONS[1]["description"],
        input_schema=TOOL_DEFINITIONS[1]["input_schema"],
        handler_func=grep_tool_handler,
        tags=["files", "search", "content"]
    )
    
    registry.register_tool(
        name="View",
        description=TOOL_DEFINITIONS[2]["description"],
        input_schema=TOOL_DEFINITIONS[2]["input_schema"],
        handler_func=view_handler,
        tags=["files", "read"]
    )
    
    registry.register_tool(
        name="Edit",
        description=TOOL_DEFINITIONS[3]["description"],
        input_schema=TOOL_DEFINITIONS[3]["input_schema"],
        handler_func=edit_handler,
        tags=["files", "edit"]
    )
    
    registry.register_tool(
        name="Replace",
        description=TOOL_DEFINITIONS[4]["description"],
        input_schema=TOOL_DEFINITIONS[4]["input_schema"],
        handler_func=replace_handler,
        tags=["files", "edit"]
    )
    
    registry.register_tool(
        name="LS",
        description=TOOL_DEFINITIONS[5]["description"],
        input_schema=TOOL_DEFINITIONS[5]["input_schema"],
        handler_func=ls_handler,
        tags=["files", "list"]
    )
    
    registry.register_tool(
        name="RegexEdit",
        description=TOOL_DEFINITIONS[6]["description"],
        input_schema=TOOL_DEFINITIONS[6]["input_schema"],
        handler_func=regex_edit_handler,
        tags=["files", "edit", "regex"]
    )
    
    # Register new, more efficient tools
    registry.register_tool(
        name="BatchGlob",
        description=TOOL_DEFINITIONS[7]["description"],
        input_schema=TOOL_DEFINITIONS[7]["input_schema"],
        handler_func=batch_glob_handler,
        tags=["files", "search", "batch"]
    )
    
    registry.register_tool(
        name="FindInFiles",
        description=TOOL_DEFINITIONS[8]["description"],
        input_schema=TOOL_DEFINITIONS[8]["input_schema"],
        handler_func=find_in_files_handler,
        tags=["files", "search", "content"]
    )
    
    registry.register_tool(
        name="SmartSearch",
        description=TOOL_DEFINITIONS[9]["description"],
        input_schema=TOOL_DEFINITIONS[9]["input_schema"],
        handler_func=smart_search_handler,
        tags=["files", "search", "smart"]
    )
    
    # Register repository mapping tools
    registry.register_tool(
        name="RepoMap",
        description=TOOL_DEFINITIONS[10]["description"],
        input_schema=TOOL_DEFINITIONS[10]["input_schema"],
        handler_func=repo_map_handler,
        tags=["repo", "map"]
    )
    
    registry.register_tool(
        name="RepoSummary",
        description=TOOL_DEFINITIONS[11]["description"],
        input_schema=TOOL_DEFINITIONS[11]["input_schema"],
        handler_func=repo_summary_handler,
        tags=["repo", "summary"]
    )
    
    registry.register_tool(
        name="RepoDependencies",
        description=TOOL_DEFINITIONS[12]["description"],
        input_schema=TOOL_DEFINITIONS[12]["input_schema"],
        handler_func=repo_deps_handler,
        tags=["repo", "dependencies"]
    )
    
    # Register shell tool with proper restrictions
    registry.register_tool(
        name="Shell",
        description=TOOL_DEFINITIONS[13]["description"],
        input_schema=TOOL_DEFINITIONS[13]["input_schema"],
        handler_func=shell_handler,
        tags=["system", "shell"]
    )
