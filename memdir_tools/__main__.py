#!/usr/bin/env python3
"""
Main entry point for Memdir memory management tools
"""

import sys
import argparse
from memdir_tools.cli import main as cli_main
from memdir_tools.utils import ensure_memdir_structure

def main():
    """Main entry point"""
    # Ensure the Memdir structure exists
    ensure_memdir_structure()
    
    # Define special commands
    special_commands = ["init-samples", "run-filters", "maintenance", "archive", 
                        "cleanup", "empty-trash", "retention", "update-status",
                        "create-folder", "rename-folder", "delete-folder", "move-folder",
                        "copy-folder", "list-folders", "folder-stats", "make-symlinks", "bulk-tag",
                        "search", "find", "advanced-search"]
    
    # Dispatch to CLI main if no special commands
    if len(sys.argv) < 2 or sys.argv[1] not in special_commands:
        cli_main()
        return
    
    # Sample generation commands
    if sys.argv[1] == "init-samples":
        from memdir_tools.create_samples import create_samples
        
        # Parse count argument if provided
        count = 20
        if len(sys.argv) > 2:
            try:
                count = int(sys.argv[2])
            except ValueError:
                pass
                
        # Create samples
        create_samples(count)
        print(f"Sample memories have been created in the Memdir structure.")
    
    # Filter commands
    elif sys.argv[1] == "run-filters":
        from memdir_tools.filter import run_filters
        
        # Parse arguments for run-filters
        parser = argparse.ArgumentParser(description="Run memory filters")
        parser.add_argument("--dry-run", action="store_true", help="Simulate actions without applying them")
        parser.add_argument("--all", action="store_true", help="Process all memories (not just new)")
        
        # Parse only the remaining arguments
        args = parser.parse_args(sys.argv[2:])
        
        # Run filters
        run_filters(dry_run=args.dry_run)
    
    # Archiver commands
    elif sys.argv[1] in ["maintenance", "archive", "cleanup", "empty-trash", "retention", "update-status"]:
        from memdir_tools.archiver import main as archiver_main
        
        # Replace the command name in sys.argv
        sys.argv[0] = "memdir_tools.archiver"
        
        # Run the archiver
        archiver_main()
        
    # Folder management commands
    elif sys.argv[1] in ["create-folder", "rename-folder", "delete-folder", "move-folder", 
                     "copy-folder", "list-folders", "folder-stats", "make-symlinks", "bulk-tag"]:
        from memdir_tools.folders import main as folders_main
        
        # Replace the command name in sys.argv
        sys.argv[0] = "memdir_tools.folders"
        
        # Run the folder manager
        folders_main()
        
    # Search commands
    elif sys.argv[1] in ["search", "find", "advanced-search"]:
        from memdir_tools.search import main as search_main
        
        # Replace the command arguments
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        
        # Run the search tool
        search_main()

if __name__ == "__main__":
    main()