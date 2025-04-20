#!/usr/bin/env python3
"""
Memory filtering system based on headers - similar to email filtering rules
"""

import os
import re
import json
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from memdir_tools.utils import (
    list_memories,
    move_memory,
    update_memory_flags,
    parse_memory_content,
    STANDARD_FOLDERS
)

class MemoryFilter:
    """
    Filter for memories based on conditions
    """
    
    def __init__(self, name: str):
        """Initialize a filter with a name"""
        self.name = name
        self.conditions = []
        self.actions = []
    
    def add_condition(self, field: str, pattern: str, negate: bool = False) -> 'MemoryFilter':
        """
        Add a condition to the filter
        
        Args:
            field: The field to check (e.g., "Subject", "Tags", "content")
            pattern: Regular expression pattern to match
            negate: Whether to negate the match
            
        Returns:
            Self for chaining
        """
        self.conditions.append({
            "field": field,
            "pattern": pattern,
            "negate": negate
        })
        return self
    
    def add_action(self, action_type: str, **params) -> 'MemoryFilter':
        """
        Add an action to the filter
        
        Args:
            action_type: Type of action ("move", "flag", "copy")
            **params: Parameters for the action
            
        Returns:
            Self for chaining
        """
        self.actions.append({
            "type": action_type,
            **params
        })
        return self
    
    def matches(self, memory: Dict[str, Any]) -> bool:
        """
        Check if a memory matches all conditions
        
        Args:
            memory: Memory info dictionary
            
        Returns:
            True if all conditions match, False otherwise
        """
        if not self.conditions:
            return True
            
        for condition in self.conditions:
            field = condition["field"]
            pattern = condition["pattern"]
            negate = condition["negate"]
            
            # Check the field
            value = None
            
            if field == "content":
                value = memory.get("content", "")
            elif field in memory["headers"]:
                value = memory["headers"][field]
            elif field == "flags":
                value = "".join(memory["metadata"]["flags"])
            elif field in memory["metadata"]:
                value = str(memory["metadata"][field])
            
            if value is None:
                # Field not found, condition fails
                if negate:
                    continue
                else:
                    return False
            
            # Check if pattern matches
            match = re.search(pattern, value, re.IGNORECASE)
            if (match and negate) or (not match and not negate):
                return False
                
        return True
    
    def apply_actions(self, memory: Dict[str, Any]) -> List[str]:
        """
        Apply actions to a memory
        
        Args:
            memory: Memory info dictionary
            
        Returns:
            List of messages describing the actions taken
        """
        messages = []
        
        for action in self.actions:
            action_type = action["type"]
            
            if action_type == "move":
                target_folder = action.get("target_folder", "")
                target_status = action.get("target_status", "cur")
                
                result = move_memory(
                    memory["filename"],
                    memory["folder"],
                    target_folder,
                    memory["status"],
                    target_status
                )
                
                if result:
                    messages.append(f"Moved to {target_folder or 'Inbox'}/{target_status}")
            
            elif action_type == "flag":
                flags = action.get("flags", "")
                mode = action.get("mode", "add")  # "add", "remove", "set"
                
                current_flags = "".join(memory["metadata"]["flags"])
                
                if mode == "add":
                    new_flags = current_flags + flags
                    # Remove duplicates
                    new_flags = "".join(sorted(set(new_flags)))
                elif mode == "remove":
                    new_flags = "".join([f for f in current_flags if f not in flags])
                else:  # "set"
                    new_flags = flags
                
                result = update_memory_flags(
                    memory["filename"],
                    memory["folder"],
                    memory["status"],
                    new_flags
                )
                
                if result:
                    messages.append(f"Flags updated from '{current_flags}' to '{new_flags}'")
            
            elif action_type == "copy":
                target_folder = action.get("target_folder", "")
                
                # We don't have a copy function, so we'll do this manually
                # This would be implemented in a real system
                messages.append(f"Would copy to {target_folder or 'Inbox'}")
                
        return messages

class FilterManager:
    """
    Manager for memory filters
    """
    
    def __init__(self):
        """Initialize filter manager"""
        self.filters = []
    
    def add_filter(self, filter_obj: MemoryFilter) -> None:
        """Add a filter to the manager"""
        self.filters.append(filter_obj)
    
    def process_memories(self, folders: List[str] = None, statuses: List[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Process all memories with the configured filters
        
        Args:
            folders: List of folders to process (None = all folders)
            statuses: List of statuses to process (None = all statuses)
            dry_run: Whether to actually apply actions or just simulate
            
        Returns:
            Dictionary with action statistics
        """
        # Collect all memories to process
        all_memories = []
        
        # Default to new folder only (incoming memories)
        if statuses is None:
            statuses = ["new"]
        
        # Get default folder list if not specified
        if folders is None:
            from memdir_tools.utils import get_memdir_folders
            folders = get_memdir_folders()
        
        # Collect memories from all specified folders and statuses
        for folder in folders:
            for status in statuses:
                memories = list_memories(folder, status, include_content=True)
                all_memories.extend(memories)
        
        # Process each memory with each filter
        stats = {
            "total_memories": len(all_memories),
            "filters_applied": 0,
            "actions_taken": 0,
            "memories_modified": 0,
            "details": []
        }
        
        modified_memories = set()
        
        for memory in all_memories:
            memory_actions = []
            
            for filter_obj in self.filters:
                if filter_obj.matches(memory):
                    stats["filters_applied"] += 1
                    
                    # Apply actions if not a dry run
                    if not dry_run:
                        actions = filter_obj.apply_actions(memory)
                    else:
                        # Simulate actions
                        actions = [f"Would {action['type']}" for action in filter_obj.actions]
                    
                    stats["actions_taken"] += len(actions)
                    
                    if actions:
                        modified_memories.add(memory["metadata"]["unique_id"])
                        memory_actions.append({
                            "filter": filter_obj.name,
                            "actions": actions
                        })
            
            if memory_actions:
                stats["details"].append({
                    "memory_id": memory["metadata"]["unique_id"],
                    "subject": memory["headers"].get("Subject", "No subject"),
                    "filters_applied": memory_actions
                })
        
        stats["memories_modified"] = len(modified_memories)
        
        return stats

def create_default_filters() -> FilterManager:
    """Create a set of default filters"""
    manager = FilterManager()
    
    # Filter 1: Python-related content goes to Python projects folder
    python_filter = MemoryFilter("Python Content")
    python_filter.add_condition("Tags", r"python")
    python_filter.add_condition("content", r"python|django|flask", negate=True)
    python_filter.add_action("move", target_folder=".Projects/Python", target_status="cur")
    python_filter.add_action("flag", flags="P", mode="add")
    
    # Filter 2: AI-related content goes to AI projects folder
    ai_filter = MemoryFilter("AI Content")
    ai_filter.add_condition("Tags", r"ai|machine[- ]learning|neural|llm")
    ai_filter.add_action("move", target_folder=".Projects/AI", target_status="cur")
    
    # Filter 3: Books and learning content goes to ToDoLater
    learning_filter = MemoryFilter("Learning Content")
    learning_filter.add_condition("Tags", r"books|reading|learning")
    learning_filter.add_condition("Subject", r"books|read|learning")
    learning_filter.add_action("move", target_folder=".ToDoLater/Learning", target_status="cur")
    
    # Filter 4: High priority gets flagged
    priority_filter = MemoryFilter("High Priority")
    priority_filter.add_condition("Priority", r"high")
    priority_filter.add_action("flag", flags="FP", mode="add")
    
    # Filter 5: Move completed items to archive
    done_filter = MemoryFilter("Completed Items")
    done_filter.add_condition("Status", r"completed|done|archived")
    done_filter.add_action("move", target_folder=".Archive/2023", target_status="cur")
    done_filter.add_action("flag", flags="S", mode="add")
    
    # Filter 6: Tag messages with "delete" or "trash" to trash
    trash_filter = MemoryFilter("Trash Items")
    trash_filter.add_condition("Tags", r"trash|delete|remove")
    trash_filter.add_action("move", target_folder=".Trash", target_status="cur")
    
    # Add all filters to manager
    manager.add_filter(python_filter)
    manager.add_filter(ai_filter)
    manager.add_filter(learning_filter)
    manager.add_filter(priority_filter)
    manager.add_filter(done_filter)
    manager.add_filter(trash_filter)
    
    return manager

def run_filters(dry_run: bool = False) -> None:
    """Run the default filters on all memories"""
    manager = create_default_filters()
    
    # Process only new memories by default
    stats = manager.process_memories(statuses=["new"], dry_run=dry_run)
    
    print(f"Processed {stats['total_memories']} memories")
    print(f"Applied {stats['filters_applied']} filters")
    print(f"Took {stats['actions_taken']} actions")
    print(f"Modified {stats['memories_modified']} memories")
    
    if stats['details']:
        print("\nDetails:")
        for detail in stats['details']:
            print(f"- {detail['subject']} ({detail['memory_id']})")
            for filter_applied in detail['filters_applied']:
                print(f"  - {filter_applied['filter']}: {', '.join(filter_applied['actions'])}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Filter System")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without applying them")
    parser.add_argument("--all", action="store_true", help="Process all memories (not just new)")
    
    args = parser.parse_args()
    
    if args.all:
        statuses = STANDARD_FOLDERS
    else:
        statuses = ["new"]
    
    # Create filter manager
    manager = create_default_filters()
    
    # Run filters
    stats = manager.process_memories(statuses=statuses, dry_run=args.dry_run)
    
    print(f"Processed {stats['total_memories']} memories")
    print(f"Applied {stats['filters_applied']} filters")
    print(f"Took {stats['actions_taken']} actions")
    print(f"Modified {stats['memories_modified']} memories")
    
    if stats['details']:
        print("\nDetails:")
        for detail in stats['details']:
            print(f"- {detail['subject']} ({detail['memory_id']})")
            for filter_applied in detail['filters_applied']:
                print(f"  - {filter_applied['filter']}: {', '.join(filter_applied['actions'])}")