#!/usr/bin/env python3
"""
Memory archiving, cleaning, and updating system

This module provides tools for:
1. Archiving old or inactive memories
2. Cleaning up redundant or obsolete memories
3. Automatic updating of memory statuses based on age
4. Managing memory retention policies
"""

import os
import re
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from memdir_tools.utils import (
    list_memories,
    move_memory,
    update_memory_flags,
    get_memdir_folders,
    STANDARD_FOLDERS,
    SPECIAL_FOLDERS,
    MEMDIR_BASE
)

# ANSI color codes
COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "red": "\033[31m",
}

def colorize(text: str, color: str) -> str:
    """Apply color to text"""
    return f"{COLORS.get(color, '')}{text}{COLORS['reset']}"

class MemoryArchiver:
    """
    Memory archiving and cleanup system
    """
    
    def __init__(self):
        """Initialize archiver with default settings"""
        self.archive_age = 90  # days
        self.trash_age = 30  # days
        self.archive_folder = ".Archive"
        self.trash_folder = ".Trash"
        self.archive_rules = []
        self.cleanup_rules = []
        self.retention_policies = {}
        self.tag_based_archiving = {}
        
    def set_archive_age(self, days: int) -> None:
        """Set the age threshold for archiving memories"""
        self.archive_age = days
        
    def set_trash_age(self, days: int) -> None:
        """Set the age threshold for memories in trash"""
        self.trash_age = days
        
    def add_archive_rule(self, folder: str, age_days: int, target_folder: Optional[str] = None) -> None:
        """
        Add an archiving rule
        
        Args:
            folder: Source folder
            age_days: Age in days for archiving
            target_folder: Target archive folder (default: .Archive)
        """
        if target_folder is None:
            target_folder = self.archive_folder
            
        rule = {
            "folder": folder,
            "age_days": age_days,
            "target_folder": target_folder
        }
        
        self.archive_rules.append(rule)
        
    def add_tag_based_archive_rule(self, tag: str, target_folder: str) -> None:
        """
        Add a tag-based archiving rule
        
        Args:
            tag: Tag to match
            target_folder: Target archive folder
        """
        self.tag_based_archiving[tag] = target_folder
        
    def set_retention_policy(self, folder: str, max_count: int, mode: str = "age") -> None:
        """
        Set retention policy for a folder
        
        Args:
            folder: Target folder
            max_count: Maximum number of memories to keep
            mode: How to select memories to remove ("age" or "importance")
        """
        self.retention_policies[folder] = {
            "max_count": max_count,
            "mode": mode
        }
        
    def add_cleanup_rule(self, criteria: Dict[str, Any], action: str = "trash") -> None:
        """
        Add a cleanup rule
        
        Args:
            criteria: Dictionary of criteria to match (e.g., {"status": "completed"})
            action: Action to take ("trash" or "delete")
        """
        rule = {
            "criteria": criteria,
            "action": action
        }
        
        self.cleanup_rules.append(rule)
        
    def _memory_matches_criteria(self, memory: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if a memory matches given criteria"""
        for key, pattern in criteria.items():
            if key in ("age", "min_age", "max_age"):
                # Age-based criteria
                memory_date = memory["metadata"]["date"]
                memory_age = (datetime.now() - memory_date).days
                
                if key == "age" and memory_age < pattern:
                    return False
                elif key == "min_age" and memory_age < pattern:
                    return False
                elif key == "max_age" and memory_age > pattern:
                    return False
            
            elif key == "tag" or key == "tags":
                # Tag-based criteria
                memory_tags = memory["headers"].get("Tags", "").lower().split(",")
                memory_tags = [tag.strip() for tag in memory_tags]
                
                if isinstance(pattern, list):
                    # Check if any tag matches
                    if not any(tag in memory_tags for tag in pattern):
                        return False
                else:
                    # Single tag
                    if pattern.lower() not in memory_tags:
                        return False
            
            elif key in memory["headers"]:
                # Header-based criteria
                value = memory["headers"][key]
                if isinstance(pattern, str):
                    if not re.search(pattern, value, re.IGNORECASE):
                        return False
                else:
                    if value != pattern:
                        return False
            
            elif key == "flags":
                # Flag-based criteria
                memory_flags = "".join(memory["metadata"]["flags"])
                if isinstance(pattern, str):
                    if not re.search(pattern, memory_flags, re.IGNORECASE):
                        return False
                else:
                    if memory_flags != pattern:
                        return False
            
            else:
                # Field not found, criteria fails
                return False
                
        return True
        
    def _create_archive_subfolder_by_date(self, memory_date: datetime) -> str:
        """
        Create an archive subfolder based on date
        
        Args:
            memory_date: The date to use
            
        Returns:
            Archive subfolder path (e.g., ".Archive/2023")
        """
        year = str(memory_date.year)
        month = f"{memory_date.month:02d}"
        
        # Create yearly archive folder
        year_folder = os.path.join(self.archive_folder, year)
        
        # Create folder structure if it doesn't exist
        for status in STANDARD_FOLDERS:
            os.makedirs(os.path.join(MEMDIR_BASE, year_folder, status), exist_ok=True)
            
        return year_folder
        
    def archive_old_memories(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Archive old memories based on age
        
        Args:
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Statistics about archived memories
        """
        stats = {
            "archived": 0,
            "details": []
        }
        
        # Apply default archive rule if no rules defined
        if not self.archive_rules:
            self.add_archive_rule("", self.archive_age, self.archive_folder)
            
        # Process each archive rule
        for rule in self.archive_rules:
            folder = rule["folder"]
            age_days = rule["age_days"]
            target_folder = rule["target_folder"]
            
            # Get memories from the folder
            for status in ["cur"]:  # Only archive from cur
                memories = list_memories(folder, status, include_content=False)
                
                for memory in memories:
                    memory_date = memory["metadata"]["date"]
                    memory_age = (datetime.now() - memory_date).days
                    
                    if memory_age >= age_days:
                        # Create target folder by date if needed
                        target = self._create_archive_subfolder_by_date(memory_date)
                        
                        if not dry_run:
                            # Move to archive
                            move_memory(
                                memory["filename"],
                                folder,
                                target,
                                status,
                                "cur"
                            )
                            
                            # Add Seen flag if not already present
                            if "S" not in memory["metadata"]["flags"]:
                                flags = "".join(memory["metadata"]["flags"]) + "S"
                                update_memory_flags(
                                    memory["filename"],
                                    target,
                                    "cur",
                                    flags
                                )
                                
                        stats["archived"] += 1
                        stats["details"].append({
                            "memory_id": memory["metadata"]["unique_id"],
                            "subject": memory["headers"].get("Subject", "No subject"),
                            "age": memory_age,
                            "action": f"Archived to {target}" if not dry_run else f"Would archive to {target}"
                        })
                        
        # Apply tag-based archiving
        for tag, target_folder in self.tag_based_archiving.items():
            # Get all memories
            for folder in get_memdir_folders():
                for status in ["cur"]:  # Only archive from cur
                    memories = list_memories(folder, status, include_content=False)
                    
                    for memory in memories:
                        memory_tags = memory["headers"].get("Tags", "").lower().split(",")
                        memory_tags = [t.strip() for t in memory_tags]
                        
                        if tag.lower() in memory_tags:
                            # Ensure target folder exists
                            for s in STANDARD_FOLDERS:
                                os.makedirs(os.path.join(MEMDIR_BASE, target_folder, s), exist_ok=True)
                                
                            if not dry_run:
                                # Move to tag-based archive
                                move_memory(
                                    memory["filename"],
                                    folder,
                                    target_folder,
                                    status,
                                    "cur"
                                )
                                
                            stats["archived"] += 1
                            stats["details"].append({
                                "memory_id": memory["metadata"]["unique_id"],
                                "subject": memory["headers"].get("Subject", "No subject"),
                                "tag": tag,
                                "action": f"Archived to {target_folder} based on tag" if not dry_run else f"Would archive to {target_folder} based on tag"
                            })
                            
        return stats
        
    def cleanup_memories(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Clean up memories based on rules
        
        Args:
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Statistics about cleaned up memories
        """
        stats = {
            "trashed": 0,
            "deleted": 0,
            "details": []
        }
        
        # Apply default cleanup rule if no rules defined
        if not self.cleanup_rules:
            self.add_cleanup_rule({"status": "completed|done"}, "trash")
            self.add_cleanup_rule({"status": "obsolete|deprecated"}, "trash")
            
        # Process each cleanup rule
        for rule in self.cleanup_rules:
            criteria = rule["criteria"]
            action = rule["action"]
            
            # Get all memories
            for folder in get_memdir_folders():
                # Skip trash for cleanup
                if folder.startswith(self.trash_folder):
                    continue
                    
                for status in ["cur"]:  # Only clean up from cur
                    memories = list_memories(folder, status, include_content=True)
                    
                    for memory in memories:
                        if self._memory_matches_criteria(memory, criteria):
                            if action == "trash":
                                # Move to trash
                                if not dry_run:
                                    move_memory(
                                        memory["filename"],
                                        folder,
                                        self.trash_folder,
                                        status,
                                        "cur"
                                    )
                                    
                                stats["trashed"] += 1
                                stats["details"].append({
                                    "memory_id": memory["metadata"]["unique_id"],
                                    "subject": memory["headers"].get("Subject", "No subject"),
                                    "action": f"Moved to trash" if not dry_run else f"Would move to trash"
                                })
                                
                            elif action == "delete":
                                # Delete immediately
                                if not dry_run:
                                    file_path = os.path.join(
                                        MEMDIR_BASE, 
                                        folder, 
                                        status, 
                                        memory["filename"]
                                    )
                                    
                                    if os.path.exists(file_path):
                                        os.remove(file_path)
                                        
                                stats["deleted"] += 1
                                stats["details"].append({
                                    "memory_id": memory["metadata"]["unique_id"],
                                    "subject": memory["headers"].get("Subject", "No subject"),
                                    "action": f"Deleted permanently" if not dry_run else f"Would delete permanently"
                                })
                                
        return stats
        
    def empty_trash(self, age_days: Optional[int] = None, dry_run: bool = False) -> Dict[str, Any]:
        """
        Empty trash folder for memories older than age_days
        
        Args:
            age_days: Age threshold in days (default: self.trash_age)
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Statistics about deleted memories
        """
        if age_days is None:
            age_days = self.trash_age
            
        stats = {
            "deleted": 0,
            "details": []
        }
        
        # Get memories from trash
        for status in STANDARD_FOLDERS:
            memories = list_memories(self.trash_folder, status, include_content=False)
            
            for memory in memories:
                memory_date = memory["metadata"]["date"]
                memory_age = (datetime.now() - memory_date).days
                
                if memory_age >= age_days:
                    if not dry_run:
                        # Delete the file
                        file_path = os.path.join(
                            MEMDIR_BASE, 
                            self.trash_folder, 
                            status, 
                            memory["filename"]
                        )
                        
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            
                    stats["deleted"] += 1
                    stats["details"].append({
                        "memory_id": memory["metadata"]["unique_id"],
                        "subject": memory["headers"].get("Subject", "No subject"),
                        "age": memory_age,
                        "action": f"Deleted from trash" if not dry_run else f"Would delete from trash"
                    })
                    
        return stats
        
    def apply_retention_policies(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Apply retention policies to limit the number of memories in folders
        
        Args:
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Statistics about memories affected by retention policies
        """
        stats = {
            "moved": 0,
            "details": []
        }
        
        for folder, policy in self.retention_policies.items():
            max_count = policy["max_count"]
            mode = policy["mode"]
            
            # Get memories from the folder
            memories = list_memories(folder, "cur", include_content=False)
            
            # Skip if under the limit
            if len(memories) <= max_count:
                continue
                
            # Sort by appropriate criteria
            if mode == "age":
                # Sort by date (oldest first)
                memories.sort(key=lambda x: x["metadata"]["timestamp"])
            elif mode == "importance":
                # Sort by priority and flags (least important first)
                def importance_score(memory):
                    # Lower score = less important
                    score = 0
                    
                    # Check priority header
                    priority = memory["headers"].get("Priority", "").lower()
                    if priority == "high":
                        score += 3
                    elif priority == "medium":
                        score += 2
                    elif priority == "low":
                        score += 1
                        
                    # Check flags
                    flags = "".join(memory["metadata"]["flags"])
                    if "F" in flags:  # Flagged
                        score += 2
                    if "P" in flags:  # Priority
                        score += 2
                        
                    return score
                    
                memories.sort(key=importance_score)
            
            # Calculate how many to move
            to_move = memories[:len(memories) - max_count]
            
            for memory in to_move:
                # Create target folder based on date
                memory_date = memory["metadata"]["date"]
                target_folder = self._create_archive_subfolder_by_date(memory_date)
                
                if not dry_run:
                    # Move memory to archive
                    move_memory(
                        memory["filename"],
                        folder,
                        target_folder,
                        "cur",
                        "cur"
                    )
                    
                stats["moved"] += 1
                stats["details"].append({
                    "memory_id": memory["metadata"]["unique_id"],
                    "subject": memory["headers"].get("Subject", "No subject"),
                    "folder": folder,
                    "action": f"Moved to {target_folder} (retention policy)" if not dry_run else f"Would move to {target_folder} (retention policy)"
                })
                
        return stats
        
    def update_memory_statuses(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Update memory statuses based on age and content
        
        Args:
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Statistics about updated memories
        """
        stats = {
            "updated": 0,
            "details": []
        }
        
        # Define status update rules
        status_rules = [
            # Mark memories as completed if they have "completed" in content
            {
                "criteria": {"content": r"\[x\]|\bcompleted\b|\bfinished\b|\bdone\b"},
                "new_status": "completed",
                "flags_add": "S"  # Add Seen flag
            },
            # Mark old memories with no activity as dormant
            {
                "criteria": {"min_age": 60, "flags": r"^[^SR]*$"},  # No S or R flags and at least 60 days old
                "new_status": "dormant",
                "flags_add": "S"  # Add Seen flag
            }
        ]
        
        # Process all memories
        for folder in get_memdir_folders():
            for status in ["cur"]:  # Only update cur
                memories = list_memories(folder, status, include_content=True)
                
                for memory in memories:
                    for rule in status_rules:
                        if self._memory_matches_criteria(memory, rule["criteria"]):
                            # Update status header
                            current_status = memory["headers"].get("Status", "")
                            new_status = rule["new_status"]
                            
                            if current_status != new_status:
                                if not dry_run:
                                    # Read file
                                    file_path = os.path.join(
                                        MEMDIR_BASE, 
                                        folder, 
                                        status, 
                                        memory["filename"]
                                    )
                                    
                                    with open(file_path, "r") as f:
                                        content = f.read()
                                        
                                    # Update status in headers
                                    if "Status: " in content:
                                        content = re.sub(
                                            r"Status: .*$", 
                                            f"Status: {new_status}", 
                                            content, 
                                            flags=re.MULTILINE
                                        )
                                    else:
                                        # Add status header if not present
                                        content = re.sub(
                                            r"^(.*?)---", 
                                            f"\\1Status: {new_status}\n---", 
                                            content, 
                                            flags=re.DOTALL
                                        )
                                        
                                    # Write back
                                    with open(file_path, "w") as f:
                                        f.write(content)
                                        
                                    # Update flags if needed
                                    if "flags_add" in rule:
                                        current_flags = "".join(memory["metadata"]["flags"])
                                        new_flags = current_flags + rule["flags_add"]
                                        # Remove duplicates
                                        new_flags = "".join(sorted(set(new_flags)))
                                        
                                        update_memory_flags(
                                            memory["filename"],
                                            folder,
                                            status,
                                            new_flags
                                        )
                                        
                                stats["updated"] += 1
                                stats["details"].append({
                                    "memory_id": memory["metadata"]["unique_id"],
                                    "subject": memory["headers"].get("Subject", "No subject"),
                                    "folder": folder,
                                    "action": f"Updated status from '{current_status}' to '{new_status}'" if not dry_run else f"Would update status from '{current_status}' to '{new_status}'"
                                })
                                
                                # Only apply the first matching rule
                                break
                                
        return stats
        
    def run_maintenance(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run full maintenance process
        
        Args:
            dry_run: Whether to simulate actions without applying them
            
        Returns:
            Statistics about all maintenance operations
        """
        stats = {
            "archive": self.archive_old_memories(dry_run),
            "cleanup": self.cleanup_memories(dry_run),
            "trash": self.empty_trash(dry_run=dry_run),
            "retention": self.apply_retention_policies(dry_run),
            "status": self.update_memory_statuses(dry_run)
        }
        
        return stats


# Command-line interface
def parse_args():
    """Parse command-line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Archiving and Maintenance System")
    parser.add_argument("--dry-run", action="store_true", help="Simulate actions without applying them")
    
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive old memories")
    archive_parser.add_argument("--age", type=int, default=90, help="Age threshold in days (default: 90)")
    archive_parser.add_argument("--folder", help="Source folder (default: all folders)")
    archive_parser.add_argument("--target", help="Target archive folder (default: .Archive)")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up memories based on criteria")
    cleanup_parser.add_argument("--status", help="Status to match (e.g., 'completed')")
    cleanup_parser.add_argument("--tags", help="Tags to match (comma-separated)")
    cleanup_parser.add_argument("--action", choices=["trash", "delete"], default="trash", help="Action to take")
    
    # Empty trash command
    trash_parser = subparsers.add_parser("empty-trash", help="Empty trash folder")
    trash_parser.add_argument("--age", type=int, default=30, help="Age threshold in days (default: 30)")
    
    # Retention command
    retention_parser = subparsers.add_parser("retention", help="Apply retention policies")
    retention_parser.add_argument("--folder", required=True, help="Target folder")
    retention_parser.add_argument("--max", type=int, required=True, help="Maximum number of memories to keep")
    retention_parser.add_argument("--mode", choices=["age", "importance"], default="age", help="Selection mode")
    
    # Update status command
    status_parser = subparsers.add_parser("update-status", help="Update memory statuses")
    
    # Maintenance command
    maintenance_parser = subparsers.add_parser("maintenance", help="Run full maintenance process")
    
    return parser.parse_args()
    
def main():
    """Main entry point"""
    args = parse_args()
    
    # Create archiver
    archiver = MemoryArchiver()
    
    # Process command
    if args.command == "archive":
        if args.folder:
            archiver.add_archive_rule(args.folder, args.age, args.target)
        else:
            archiver.set_archive_age(args.age)
            if args.target:
                archiver.archive_folder = args.target
                
        stats = archiver.archive_old_memories(args.dry_run)
        
        print(f"Archived {stats['archived']} memories")
        if stats['details']:
            print("\nDetails:")
            for detail in stats['details']:
                print(f"- {detail['subject']} ({detail['memory_id']}): {detail['action']}")
                
    elif args.command == "cleanup":
        criteria = {}
        
        if args.status:
            criteria["status"] = args.status
            
        if args.tags:
            criteria["tags"] = args.tags.split(",")
            
        if criteria:
            archiver.add_cleanup_rule(criteria, args.action)
            
        stats = archiver.cleanup_memories(args.dry_run)
        
        print(f"Trashed {stats['trashed']} memories")
        print(f"Deleted {stats['deleted']} memories")
        if stats['details']:
            print("\nDetails:")
            for detail in stats['details']:
                print(f"- {detail['subject']} ({detail['memory_id']}): {detail['action']}")
                
    elif args.command == "empty-trash":
        stats = archiver.empty_trash(args.age, args.dry_run)
        
        print(f"Deleted {stats['deleted']} memories from trash")
        if stats['details']:
            print("\nDetails:")
            for detail in stats['details']:
                print(f"- {detail['subject']} ({detail['memory_id']}): {detail['action']}")
                
    elif args.command == "retention":
        archiver.set_retention_policy(args.folder, args.max, args.mode)
        stats = archiver.apply_retention_policies(args.dry_run)
        
        print(f"Moved {stats['moved']} memories due to retention policies")
        if stats['details']:
            print("\nDetails:")
            for detail in stats['details']:
                print(f"- {detail['subject']} ({detail['memory_id']}): {detail['action']}")
                
    elif args.command == "update-status":
        stats = archiver.update_memory_statuses(args.dry_run)
        
        print(f"Updated {stats['updated']} memory statuses")
        if stats['details']:
            print("\nDetails:")
            for detail in stats['details']:
                print(f"- {detail['subject']} ({detail['memory_id']}): {detail['action']}")
                
    elif args.command == "maintenance" or not args.command:
        stats = archiver.run_maintenance(args.dry_run)
        
        print(colorize("Memory Maintenance Report", "bold"))
        print(f"Archived: {stats['archive']['archived']} memories")
        print(f"Trashed: {stats['cleanup']['trashed']} memories")
        print(f"Deleted from trash: {stats['trash']['deleted']} memories")
        print(f"Moved by retention: {stats['retention']['moved']} memories")
        print(f"Updated statuses: {stats['status']['updated']} memories")
        
        if args.dry_run:
            print(colorize("\nThis was a dry run. No changes were made.", "yellow"))
        
        total_changes = (
            stats['archive']['archived'] +
            stats['cleanup']['trashed'] +
            stats['cleanup']['deleted'] +
            stats['trash']['deleted'] +
            stats['retention']['moved'] +
            stats['status']['updated']
        )
        
        if total_changes > 0:
            print(colorize(f"\nTotal memories affected: {total_changes}", "green"))
        else:
            print(colorize("\nNo memories were affected.", "cyan"))

if __name__ == "__main__":
    main()