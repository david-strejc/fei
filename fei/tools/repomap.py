#!/usr/bin/env python3
"""
Repository mapping module for Fei

This module provides tools for creating a concise map of a code repository
to help the AI understand the codebase structure more efficiently.
Inspired by the aider.chat approach using tree-sitter.
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tree_sitter_languages import get_language, get_parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from fei.utils.logging import get_logger

logger = get_logger(__name__)

# Language extensions mapping
LANGUAGE_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".jsx"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".hpp", ".cc", ".hh", ".cxx"],
    "ruby": [".rb"],
    "go": [".go"],
    "rust": [".rs"],
    "php": [".php"],
    "csharp": [".cs"],
}

# Fallback pattern-based parsing for when tree-sitter is not available
FALLBACK_PATTERNS = {
    "python": {
        "class": r"^class\s+(\w+)(?:\(.*\))?:",
        "function": r"^def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*([^:]+))?:",
        "method": r"^\s+def\s+(\w+)\s*\((self|cls)(?:,\s*(.*?))?\)(?:\s*->\s*([^:]+))?:",
        "attribute": r"^\s+self\.(\w+)\s*=",
    },
    "javascript": {
        "class": r"^class\s+(\w+)(?:\s+extends\s+(\w+))?",
        "function": r"^(?:async\s+)?function\s+(\w+)\s*\((.*?)\)",
        "method": r"^\s+(?:async\s+)?(\w+)\s*\((.*?)\)",
        "attribute": r"^\s+this\.(\w+)\s*=",
    },
    "typescript": {
        "class": r"^class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+(\w+))?",
        "interface": r"^interface\s+(\w+)(?:\s+extends\s+(\w+))?",
        "function": r"^(?:async\s+)?function\s+(\w+)\s*\((.*?)\)(?:\s*:\s*([^{]+))?",
        "method": r"^\s+(?:async\s+)?(\w+)\s*\((.*?)\)(?:\s*:\s*([^{]+))?",
        "property": r"^\s+(\w+)\s*:\s*([^;]+)",
    },
}

class RepoMapper:
    """Creates a concise map of a code repository to help LLMs understand the codebase"""
    
    def __init__(self, repo_path: str, token_budget: int = 1000, exclude_patterns: Optional[List[str]] = None):
        """
        Initialize the repository mapper
        
        Args:
            repo_path: Path to the repository root
            token_budget: Maximum number of tokens for the repo map
            exclude_patterns: List of glob patterns to exclude
        """
        self.repo_path = os.path.abspath(repo_path)
        self.token_budget = token_budget
        self.exclude_patterns = exclude_patterns or [
            "**/.git/**", "**/node_modules/**", "**/venv/**", "**/__pycache__/**",
            "**/.venv/**", "**/build/**", "**/dist/**", "**/*.min.js"
        ]
        self.use_tree_sitter = TREE_SITTER_AVAILABLE
        
        if not self.use_tree_sitter:
            logger.warning(
                "tree-sitter-languages not available. Using fallback pattern-based parsing. "
                "Install with: pip install tree-sitter-languages"
            )
    
    def generate_map(self) -> str:
        """
        Generate a repository map
        
        Returns:
            Repository map as a formatted string
        """
        files = self._find_code_files()
        
        if not files:
            return "No code files found in the repository."
        
        if self.use_tree_sitter:
            file_symbols = self._extract_symbols_tree_sitter(files)
        else:
            file_symbols = self._extract_symbols_patterns(files)
        
        # Create graph of file dependencies
        dependencies = self._find_dependencies(file_symbols)
        
        # Get the most important files based on ranking
        important_files = self._rank_files(dependencies)
        
        # Generate the map
        return self._format_map(file_symbols, important_files)
    
    def _find_code_files(self) -> List[str]:
        """Find all code files in the repository"""
        all_files = []
        for root, dirs, files in os.walk(self.repo_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(
                os.path.relpath(os.path.join(root, d), self.repo_path).startswith(pattern.strip("**/"))
                for pattern in self.exclude_patterns
            )]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_rel_path = os.path.relpath(file_path, self.repo_path)
                
                # Skip excluded files
                if any(self._matches_pattern(file_rel_path, pattern) for pattern in self.exclude_patterns):
                    continue
                
                # Only include code files
                extension = os.path.splitext(file)[1].lower()
                if any(extension in exts for exts in LANGUAGE_EXTENSIONS.values()):
                    all_files.append(file_path)
        
        return all_files
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches glob pattern"""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)
    
    def _get_language_for_file(self, file_path: str) -> Optional[str]:
        """Get tree-sitter language for a file"""
        extension = os.path.splitext(file_path)[1].lower()
        
        for lang, exts in LANGUAGE_EXTENSIONS.items():
            if extension in exts:
                return lang
        
        return None
    
    def _extract_symbols_tree_sitter(self, files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract symbols from files using tree-sitter"""
        file_symbols = {}
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {executor.submit(self._process_file_tree_sitter, file_path): file_path for file_path in files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                try:
                    symbols = future.result()
                    if symbols:
                        file_symbols[rel_path] = symbols
                except Exception as e:
                    logger.warning(f"Error processing {rel_path}: {e}")
        
        return file_symbols
    
    def _process_file_tree_sitter(self, file_path: str) -> Dict[str, Any]:
        """Process a single file with tree-sitter to extract symbols"""
        lang_name = self._get_language_for_file(file_path)
        if not lang_name:
            return {}
        
        try:
            # Get parser and language
            language = get_language(lang_name)
            parser = get_parser(lang_name)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            # Parse the file
            tree = parser.parse(bytes(content, 'utf-8'))
            
            # Extract symbols based on language
            symbols = self._extract_symbols_from_tree(tree, content, lang_name)
            
            return {
                "language": lang_name,
                "symbols": symbols
            }
        except Exception as e:
            logger.debug(f"Tree-sitter error for {file_path}: {e}")
            return {}
    
    def _extract_symbols_from_tree(self, tree, content: str, lang_name: str) -> List[Dict[str, Any]]:
        """Extract symbols from a tree-sitter tree"""
        symbols = []
        
        # Define query patterns based on language
        # This is a simplified version - in a full implementation, you'd have more comprehensive queries
        query_text = self._get_query_for_language(lang_name)
        
        if not query_text:
            return symbols
        
        try:
            language = get_language(lang_name)
            query = language.query(query_text)
            
            captures = query.captures(tree.root_node, bytes(content, 'utf-8'))
            
            # Group captures by node to build symbol definitions
            nodes = {}
            for node, capture_name in captures:
                if node not in nodes:
                    nodes[node] = {"node": node, "captures": {}}
                nodes[node]["captures"][capture_name] = node.text.decode('utf-8')
            
            # Convert nodes to symbols
            for node_data in nodes.values():
                symbol = self._node_to_symbol(node_data, content)
                if symbol:
                    symbols.append(symbol)
            
            return symbols
        except Exception as e:
            logger.debug(f"Query error: {e}")
            return symbols
    
    def _get_query_for_language(self, lang_name: str) -> str:
        """Get tree-sitter query for a language"""
        # These are simplified queries - a full implementation would have more comprehensive ones
        queries = {
            "python": """
                (class_definition
                    name: (identifier) @class.name
                    body: (block) @class.body
                ) @class.definition
                
                (function_definition
                    name: (identifier) @function.name
                    parameters: (parameters) @function.parameters
                    body: (block) @function.body
                ) @function.definition
            """,
            "javascript": """
                (class_declaration
                    name: (identifier) @class.name
                    body: (class_body) @class.body
                ) @class.definition
                
                (method_definition
                    name: (property_identifier) @method.name
                    parameters: (formal_parameters) @method.parameters
                    body: (statement_block) @method.body
                ) @method.definition
                
                (function_declaration
                    name: (identifier) @function.name
                    parameters: (formal_parameters) @function.parameters
                    body: (statement_block) @function.body
                ) @function.definition
            """,
            # Add more languages as needed
        }
        
        return queries.get(lang_name, "")
    
    def _node_to_symbol(self, node_data: Dict[str, Any], content: str) -> Optional[Dict[str, Any]]:
        """Convert a tree-sitter node to a symbol definition"""
        captures = node_data["captures"]
        node = node_data["node"]
        
        # Extract the relevant lines from the content
        start_line = node.start_point[0]
        end_line = node.end_point[0]
        
        # Get code lines for this symbol
        lines = content.split('\n')[start_line:end_line+1]
        if not lines:
            return None
        
        # Make a simplified representation
        if "class.name" in captures:
            return {
                "type": "class",
                "name": captures["class.name"],
                "lines": [lines[0]] + ["..."],  # Just show the class definition line
                "line_range": (start_line, end_line)
            }
        elif "function.name" in captures:
            return {
                "type": "function",
                "name": captures["function.name"],
                "parameters": captures.get("function.parameters", ""),
                "lines": [lines[0]] + ["..."],  # Just show the function definition line
                "line_range": (start_line, end_line)
            }
        elif "method.name" in captures:
            return {
                "type": "method",
                "name": captures["method.name"],
                "parameters": captures.get("method.parameters", ""),
                "lines": [lines[0]] + ["..."],  # Just show the method definition line
                "line_range": (start_line, end_line)
            }
        
        return None
    
    def _extract_symbols_patterns(self, files: List[str]) -> Dict[str, Dict[str, Any]]:
        """Extract symbols from files using regex patterns (fallback method)"""
        file_symbols = {}
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_file = {executor.submit(self._process_file_patterns, file_path): file_path for file_path in files}
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                rel_path = os.path.relpath(file_path, self.repo_path)
                
                try:
                    symbols = future.result()
                    if symbols:
                        file_symbols[rel_path] = symbols
                except Exception as e:
                    logger.warning(f"Error processing {rel_path}: {e}")
        
        return file_symbols
    
    def _process_file_patterns(self, file_path: str) -> Dict[str, Any]:
        """Process a single file with regex patterns to extract symbols"""
        lang_name = self._get_language_for_file(file_path)
        if not lang_name or lang_name not in FALLBACK_PATTERNS:
            return {}
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            
            symbols = []
            patterns = FALLBACK_PATTERNS[lang_name]
            
            for i, line in enumerate(lines):
                for symbol_type, pattern in patterns.items():
                    match = re.match(pattern, line)
                    if match:
                        # Extract symbol information based on the type
                        if symbol_type == "class":
                            symbols.append({
                                "type": "class",
                                "name": match.group(1),
                                "lines": [line.strip()],
                                "line_range": (i, i)
                            })
                        elif symbol_type in ["function", "method"]:
                            params = match.group(2) if match.groups() and len(match.groups()) > 1 else ""
                            return_type = match.group(3) if match.groups() and len(match.groups()) > 2 else ""
                            
                            symbols.append({
                                "type": symbol_type,
                                "name": match.group(1),
                                "parameters": params,
                                "return_type": return_type,
                                "lines": [line.strip()],
                                "line_range": (i, i)
                            })
            
            return {
                "language": lang_name,
                "symbols": symbols
            }
        except Exception as e:
            logger.debug(f"Pattern error for {file_path}: {e}")
            return {}
    
    def _find_dependencies(self, file_symbols: Dict[str, Dict[str, Any]]) -> Dict[str, Set[str]]:
        """Find dependencies between files based on symbol references"""
        # Extract all symbol names
        all_symbols = {}
        for file_path, file_data in file_symbols.items():
            for symbol in file_data.get("symbols", []):
                symbol_name = symbol["name"]
                if symbol_name not in all_symbols:
                    all_symbols[symbol_name] = []
                all_symbols[symbol_name].append(file_path)
        
        # Find references to symbols in other files
        dependencies = {file_path: set() for file_path in file_symbols}
        
        for file_path, file_data in file_symbols.items():
            # Get file content
            full_path = os.path.join(self.repo_path, file_path)
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Check for references to symbols defined in other files
                for symbol_name, files in all_symbols.items():
                    if file_path not in files and re.search(r'\b' + re.escape(symbol_name) + r'\b', content):
                        for def_file in files:
                            # This file references a symbol defined in def_file
                            dependencies[file_path].add(def_file)
            except Exception as e:
                logger.debug(f"Error reading {file_path}: {e}")
        
        return dependencies
    
    def _rank_files(self, dependencies: Dict[str, Set[str]]) -> List[Tuple[str, float]]:
        """Rank files by importance using a simple PageRank-like algorithm"""
        # Count incoming references
        incoming = {file: 0 for file in dependencies}
        for file, deps in dependencies.items():
            for dep in deps:
                if dep in incoming:
                    incoming[dep] += 1
        
        # Calculate a simple score based on incoming and outgoing references
        scores = {}
        for file in dependencies:
            # Score = incoming references + 0.5 * outgoing references
            scores[file] = incoming[file] + 0.5 * len(dependencies[file])
        
        # Sort by score (descending)
        ranked_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_files
    
    def _format_map(self, file_symbols: Dict[str, Dict[str, Any]], ranked_files: List[Tuple[str, float]]) -> str:
        """Format the repository map as a string"""
        # Estimate tokens per file (rough approximation)
        tokens_per_file = 50  # Base estimate per file
        
        # Select files to include based on token budget
        included_files = []
        total_tokens = 0
        
        for file, score in ranked_files:
            file_tokens = tokens_per_file
            if file in file_symbols:
                # Add tokens for symbols (rough approximation)
                file_tokens += len(file_symbols[file].get("symbols", [])) * 20
            
            if total_tokens + file_tokens <= self.token_budget:
                included_files.append(file)
                total_tokens += file_tokens
            
            if total_tokens >= self.token_budget:
                break
        
        # Build the map
        map_lines = ["# Repository Map", ""]
        
        for file in included_files:
            if file not in file_symbols:
                continue
                
            rel_path = file
            language = file_symbols[file].get("language", "unknown")
            symbols = file_symbols[file].get("symbols", [])
            
            if symbols:
                map_lines.append(f"{rel_path}:")
                
                # Group symbols by type
                for symbol_type in ["class", "function", "method"]:
                    type_symbols = [s for s in symbols if s["type"] == symbol_type]
                    
                    if type_symbols:
                        for symbol in type_symbols:
                            symbol_lines = symbol.get("lines", [])
                            if symbol_lines:
                                map_lines.append(f"│{symbol_lines[0]}")
                                
                                # Add ellipsis to indicate there's more code
                                if len(symbol_lines) > 1:
                                    map_lines.append("│...")
                        
                map_lines.append("")
        
        return "\n".join(map_lines)
    
    def generate_json(self) -> str:
        """
        Generate a repository map in JSON format for easier parsing
        
        Returns:
            Repository map as a JSON string
        """
        files = self._find_code_files()
        
        if not files:
            return json.dumps({"error": "No code files found in the repository."})
        
        if self.use_tree_sitter:
            file_symbols = self._extract_symbols_tree_sitter(files)
        else:
            file_symbols = self._extract_symbols_patterns(files)
        
        # Create graph of file dependencies
        dependencies = self._find_dependencies(file_symbols)
        
        # Get the most important files based on ranking
        important_files = self._rank_files(dependencies)
        
        # Prepare data for JSON
        repo_data = {
            "repository": os.path.basename(self.repo_path),
            "file_count": len(files),
            "mapped_files": [],
        }
        
        # Select files to include based on token budget
        included_files = [file for file, _ in important_files[:50]]  # Limit to top 50 files
        
        # Add file data
        for file in included_files:
            if file not in file_symbols:
                continue
                
            file_data = {
                "path": file,
                "language": file_symbols[file].get("language", "unknown"),
                "symbols": file_symbols[file].get("symbols", []),
                "dependencies": list(dependencies.get(file, set())),
            }
            
            repo_data["mapped_files"].append(file_data)
        
        return json.dumps(repo_data, indent=2)


class RepoMapSummary:
    """Creates a concise summary of a repository map focused on LLM token efficiency"""
    
    def __init__(self, repo_mapper: RepoMapper):
        """
        Initialize with a RepoMapper instance
        
        Args:
            repo_mapper: RepoMapper instance
        """
        self.repo_mapper = repo_mapper
    
    def generate_summary(self, max_tokens: int = 500) -> str:
        """
        Generate a concise repository summary
        
        Args:
            max_tokens: Maximum number of tokens for the summary
            
        Returns:
            Repository summary as a formatted string
        """
        # Get the full map first
        repo_map_json = self.repo_mapper.generate_json()
        repo_data = json.loads(repo_map_json)
        
        if 'error' in repo_data:
            return repo_data['error']
        
        # Build a summary of the repository
        summary_lines = [
            f"# Repository Summary: {repo_data['repository']}",
            f"Total files: {repo_data['file_count']}",
            "Key components:",
            ""
        ]
        
        # Find important modules/packages
        modules = self._identify_modules(repo_data)
        
        # Add module information
        for module_name, module_info in modules.items():
            summary_lines.append(f"## {module_name}")
            summary_lines.append(f"Files: {module_info['file_count']}")
            
            # Top symbols in this module
            if module_info['key_symbols']:
                summary_lines.append("Key elements:")
                for symbol_type, symbols in module_info['key_symbols'].items():
                    if symbols:
                        symbol_names = ", ".join([s['name'] for s in symbols[:5]])
                        summary_lines.append(f"- {symbol_type.capitalize()}s: {symbol_names}")
            
            summary_lines.append("")
        
        # Main dependency structure
        summary_lines.append("## Main Dependencies")
        
        # Find top-level dependencies
        dependencies = self._extract_dependencies(repo_data)
        for module, deps in dependencies.items():
            if deps:
                dep_list = ", ".join(deps[:3])
                if len(deps) > 3:
                    dep_list += f" and {len(deps) - 3} more"
                summary_lines.append(f"- {module} → {dep_list}")
        
        return "\n".join(summary_lines)
    
    def _identify_modules(self, repo_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Identify main modules in the repository"""
        modules = {}
        
        for file_data in repo_data.get('mapped_files', []):
            path = file_data['path']
            
            # Extract module/package name from path
            parts = path.split('/')
            module_name = parts[0] if len(parts) > 0 else "root"
            
            # Initialize module info if not exists
            if module_name not in modules:
                modules[module_name] = {
                    'file_count': 0,
                    'key_symbols': {
                        'class': [],
                        'function': [],
                        'method': []
                    }
                }
            
            # Count files
            modules[module_name]['file_count'] += 1
            
            # Extract symbols
            for symbol in file_data.get('symbols', []):
                symbol_type = symbol.get('type')
                if symbol_type in modules[module_name]['key_symbols']:
                    modules[module_name]['key_symbols'][symbol_type].append(symbol)
        
        return modules
    
    def _extract_dependencies(self, repo_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract module-level dependencies"""
        dependencies = {}
        
        for file_data in repo_data.get('mapped_files', []):
            path = file_data['path']
            module_name = path.split('/')[0] if '/' in path else "root"
            
            if module_name not in dependencies:
                dependencies[module_name] = []
            
            # Extract dependencies
            for dep_file in file_data.get('dependencies', []):
                dep_module = dep_file.split('/')[0] if '/' in dep_file else "root"
                
                if dep_module != module_name and dep_module not in dependencies[module_name]:
                    dependencies[module_name].append(dep_module)
        
        return dependencies


# Main function to use the mapper
def generate_repo_map(repo_path: str, token_budget: int = 1000, exclude_patterns: Optional[List[str]] = None) -> str:
    """
    Generate a repository map
    
    Args:
        repo_path: Path to the repository root
        token_budget: Maximum number of tokens for the repo map
        exclude_patterns: List of glob patterns to exclude
        
    Returns:
        Repository map as a formatted string
    """
    mapper = RepoMapper(repo_path, token_budget, exclude_patterns)
    return mapper.generate_map()

def generate_repo_summary(repo_path: str, max_tokens: int = 500, exclude_patterns: Optional[List[str]] = None) -> str:
    """
    Generate a concise repository summary
    
    Args:
        repo_path: Path to the repository root
        max_tokens: Maximum number of tokens for the summary
        exclude_patterns: List of glob patterns to exclude
        
    Returns:
        Repository summary as a formatted string
    """
    mapper = RepoMapper(repo_path, max_tokens * 2, exclude_patterns)
    summary = RepoMapSummary(mapper)
    return summary.generate_summary(max_tokens)


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = os.getcwd()
    
    repo_map = generate_repo_map(repo_path)
    print(repo_map)