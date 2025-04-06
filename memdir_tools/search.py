#!/usr/bin/env python3
"""
Advanced search capabilities for Memdir memories
"""

import os
import re
import json
import argparse
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta
import dateutil.parser

from memdir_tools.utils import (
    list_memories,
    get_memdir_folders,
    STANDARD_FOLDERS,
    FLAGS
)

class SearchQuery:
    """
    A search query for memories with advanced filtering capabilities
    """
    
    def __init__(self):
        """Initialize an empty search query"""
        self.conditions = []
        self.sort_by = None
        self.sort_reverse = False
        self.limit = None
        self.offset = 0
        self.include_content = False
    
    def add_condition(self, field: str, operator: str, value: Any) -> 'SearchQuery':
        """
        Add a search condition
        
        Args:
            field: Field to search in (e.g., "Subject", "Tags", "content", "flags", "date")
            operator: Comparison operator (e.g., "contains", "matches", "=", ">", "<", etc.)
            value: Value to compare against
            
        Returns:
            Self for chaining
        """
        self.conditions.append({
            "field": field,
            "operator": operator,
            "value": value
        })
        return self
    
    def set_sort(self, field: str, reverse: bool = False) -> 'SearchQuery':
        """
        Set the sort order for results
        
        Args:
            field: Field to sort by (e.g., "date", "subject")
            reverse: Whether to sort in reverse order
            
        Returns:
            Self for chaining
        """
        self.sort_by = field
        self.sort_reverse = reverse
        return self
    
    def set_pagination(self, limit: Optional[int] = None, offset: int = 0) -> 'SearchQuery':
        """
        Set pagination parameters
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            Self for chaining
        """
        self.limit = limit
        self.offset = offset
        return self
    
    def with_content(self, include: bool = True) -> 'SearchQuery':
        """
        Include memory content in results
        
        Args:
            include: Whether to include content
            
        Returns:
            Self for chaining
        """
        self.include_content = include
        return self

def _get_field_value(memory: Dict[str, Any], field: str) -> Any:
    """Get the value of a field from a memory"""
    # Case insensitive field matching
    field_lower = field.lower()
    
    # Special fields
    if field_lower == "content":
        return memory.get("content", "")
    elif field_lower == "flags":
        return "".join(memory["metadata"]["flags"])
    elif field_lower == "date":
        return memory["metadata"]["date"]
    elif field_lower == "id":
        return memory["metadata"]["unique_id"]
    elif field_lower == "filename":
        return memory["filename"]
    elif field_lower == "folder":
        return memory["folder"]
    elif field_lower == "status" or field_lower == "maildir_status":
        return memory["status"]
    elif field == "Status" or field_lower == "status_value" or field_lower == "state":
        return memory["headers"].get("Status", "")
    
    # Check headers (case-insensitive)
    for header_key in memory["headers"]:
        if header_key.lower() == field_lower:
            value = memory["headers"][header_key]
            
            # Try to parse date fields
            if field_lower in ["date", "due", "created", "modified", "deleteddate"]:
                try:
                    return dateutil.parser.parse(value)
                except (ValueError, TypeError):
                    return value
                    
            return value
    
    # Check metadata (case-insensitive)
    for meta_key in memory["metadata"]:
        if meta_key.lower() == field_lower:
            return memory["metadata"][meta_key]
    
    return None

def _compare_values(value1: Any, operator: str, value2: Any) -> bool:
    """Compare two values with the given operator"""
    # Handle None values
    if value1 is None:
        return False
        
    # Text comparison operators
    if operator == "contains":
        return str(value2).lower() in str(value1).lower()
    elif operator == "matches":
        try:
            return bool(re.search(str(value2), str(value1), re.IGNORECASE))
        except re.error:
            return False
    elif operator == "startswith":
        return str(value1).lower().startswith(str(value2).lower())
    elif operator == "endswith":
        return str(value1).lower().endswith(str(value2).lower())
    
    # Tag-specific operator
    elif operator == "has_tag":
        tags = str(value1).lower().split(",")
        return str(value2).lower() in [tag.strip() for tag in tags]
    
    # Equality operators
    elif operator == "=":
        if isinstance(value1, datetime) and not isinstance(value2, datetime):
            try:
                value2 = dateutil.parser.parse(str(value2))
            except (ValueError, TypeError):
                return False
        
        # Case-insensitive string comparison
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.lower() == value2.lower()
            
        return value1 == value2
    elif operator == "!=":
        if isinstance(value1, datetime) and not isinstance(value2, datetime):
            try:
                value2 = dateutil.parser.parse(str(value2))
            except (ValueError, TypeError):
                return False
                
        # Case-insensitive string comparison
        if isinstance(value1, str) and isinstance(value2, str):
            return value1.lower() != value2.lower()
            
        return value1 != value2
    
    # Date and numeric comparison
    elif operator in [">", "<", ">=", "<="]:
        # Handle dates
        if isinstance(value1, datetime) and not isinstance(value2, datetime):
            try:
                value2 = dateutil.parser.parse(str(value2))
            except (ValueError, TypeError):
                return False
        
        # Handle relative dates
        if isinstance(value2, str) and value2.startswith("now"):
            now = datetime.now()
            
            if value2 == "now":
                value2 = now
            else:
                # Parse "now-1d", "now+1w", etc.
                match = re.match(r"now([+-])(\d+)([dwmy])", value2)
                if match:
                    op, num, unit = match.groups()
                    num = int(num)
                    
                    if op == "-":
                        num = -num
                        
                    if unit == "d":
                        value2 = now + timedelta(days=num)
                    elif unit == "w":
                        value2 = now + timedelta(weeks=num)
                    elif unit == "m":
                        # Approximate month as 30 days
                        value2 = now + timedelta(days=num * 30)
                    elif unit == "y":
                        # Approximate year as 365 days
                        value2 = now + timedelta(days=num * 365)
        
        if operator == ">":
            return value1 > value2
        elif operator == "<":
            return value1 < value2
        elif operator == ">=":
            return value1 >= value2
        elif operator == "<=":
            return value1 <= value2
    
    # Flag-specific operator
    elif operator == "has_flag":
        flags = str(value1)
        return str(value2).upper() in flags
    
    # Default
    return False

def _memory_matches_query(memory: Dict[str, Any], query: SearchQuery, debug: bool = False) -> bool:
    """Check if a memory matches a search query"""
    if debug:
        print(f"\nDEBUG: Checking memory {memory['metadata']['unique_id']} against query...")
        print(f"  Memory Headers: {memory.get('headers', {})}")
        print(f"  Memory Content: {memory.get('content', '')[:50]}...") # Show preview
        print(f"  Query Conditions: {query.conditions}")

    if not query.conditions:
        return True

    # Separate keyword conditions (pairs of Subject/content) from others
    keyword_values = set()
    keyword_conditions = []
    other_conditions = []

    # Identify potential keyword values (values used in both Subject and content 'contains' conditions)
    subject_contains = {c['value'] for c in query.conditions if c['field'] == 'Subject' and c['operator'] == 'contains'}
    content_contains = {c['value'] for c in query.conditions if c['field'] == 'content' and c['operator'] == 'contains'}
    keyword_values = subject_contains.intersection(content_contains)

    for condition in query.conditions:
        is_keyword_part = (
            condition['operator'] == 'contains' and
            condition['value'] in keyword_values and
            (condition['field'] == 'Subject' or condition['field'] == 'content')
        )
        if is_keyword_part:
            keyword_conditions.append(condition)
        else:
            other_conditions.append(condition)

    # Evaluate keyword conditions (OR logic for each keyword value)
    if keyword_values:
        if debug: print("DEBUG: Evaluating keyword conditions...")
        for keyword in keyword_values:
            subject_match = False
            content_match = False
            subject_value = _get_field_value(memory, "Subject")
            content_value = _get_field_value(memory, "content") # Assumes content is loaded if needed

            subject_match = _compare_values(subject_value, "contains", keyword)
            content_match = _compare_values(content_value, "contains", keyword)

            if not (subject_match or content_match):
                if debug: print(f"DEBUG: Keyword '{keyword}' NOT found in Subject OR content. Failing match.")
                return False # Must match at least one keyword condition group if present
            if debug: print(f"DEBUG: Keyword '{keyword}' found in Subject or content.")

    # Evaluate other conditions (AND logic)
    if debug and other_conditions: print("DEBUG: Evaluating other conditions...")
    for condition in other_conditions:
        field = condition["field"]
        operator = condition["operator"]
        target_value = condition["value"]

        # Special handling for Status header vs status field
        if field == "Status" or field == "status_value" or field == "state":
            field_value = memory["headers"].get("Status", "")
        else:
            field_value = _get_field_value(memory, field)

        if debug:
            print(f"DEBUG: Condition Check - Field: {field}, Operator: {operator}, Target: '{target_value}', Actual Value: '{field_value}'")

        match = _compare_values(field_value, operator, target_value)
        if not match:
            if debug: print(f"DEBUG: Condition failed. Failing match.")
            return False

    if debug: print(f"DEBUG: Memory MATCHED all conditions.")
    return True

def search_memories(base_dir: str, query: SearchQuery, folders: Optional[List[str]] = None, statuses: Optional[List[str]] = None, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Search memories with an advanced query
    
    Args:
        query: The search query
        folders: List of folders to search (None = all folders)
        statuses: List of statuses to search (None = all statuses)
        debug: Whether to print debug information
        
    Returns:
        List of matching memories
    """
    results = []
    
    # Default to all folders if not specified
    if folders is None:
        folders = get_memdir_folders(base_dir) # Pass base_dir
    
    # Default to all standard folders if not specified
    if statuses is None:
        statuses = STANDARD_FOLDERS
    
    # Determine if content is needed for matching
    needs_content_for_match = any(
        cond["field"] == "content" for cond in query.conditions
    )

    # Search in each folder and status
    for folder in folders:
        for status in statuses:
            # Always include content if needed for matching, otherwise respect query setting
            include_content_for_list = needs_content_for_match or query.include_content
            memories = list_memories(base_dir, folder, status, include_content=include_content_for_list) # Pass base_dir

            for memory in memories:
                if _memory_matches_query(memory, query, debug):
                    # If content was included only for matching, remove it before adding to results
                    if needs_content_for_match and not query.include_content and "content" in memory:
                         del memory["content"] # Remove content if not requested in output
                    results.append(memory)
    
    # Sort results if specified
    if query.sort_by:
        try:
            results.sort(
                key=lambda x: _get_field_value(x, query.sort_by) or "", 
                reverse=query.sort_reverse
            )
        except Exception as e:
            print(f"Warning: Unable to sort results: {e}")
            # Fall back to sorting by date
            results.sort(
                key=lambda x: x["metadata"]["timestamp"],
                reverse=True
            )
    
    # Apply pagination
    if query.offset or query.limit:
        start = query.offset
        end = None if query.limit is None else start + query.limit
        results = results[start:end]
    
    return results

def parse_search_args(args_str: str) -> SearchQuery:
    """
    Parse a search string into a SearchQuery
    
    Format examples:
    - "subject:python tags:learning"
    - "content:/regex pattern/ date>2023-01-01"
    - "priority:high status!=completed"
    
    Special operators:
    - fieldname:value => field contains value
    - fieldname=/regex/ => field matches regex
    - fieldname=value => field equals value
    - fieldname>value, fieldname<value => greater/less than
    - fieldname!=value => not equals
    - flags:S => has Seen flag
    - tags:python => has python tag
    
    Special fields:
    - "status:" refers to file status (cur, new, tmp)
    - "status_value:" or "state:" refers to the Status header field
    
    Args:
        args_str: Search string
        
    Returns:
        SearchQuery object
    """
    query = SearchQuery()
    
    # Split the string, respecting quoted strings
    pattern = r'([^\s"]+)|"([^"]*)"'
    matches = re.findall(pattern, args_str)
    tokens = [m[0] or m[1] for m in matches]
    
    # Process tokens
    keyword_tokens = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        
        # Tag shorthand (e.g. #python is equivalent to tags:python)
        if token.startswith("#") and len(token) > 1:
            tag = token[1:]
            query.add_condition("Tags", "has_tag", tag)
            i += 1
            continue
            
        # Flag shorthand (e.g. +F is equivalent to flags:F)
        if token.startswith("+") and len(token) > 1 and all(c in "FRSP" for c in token[1:]):
            for flag in token[1:]:
                query.add_condition("flags", "has_flag", flag)
            i += 1
            continue
        
        # Field-specific queries
        field_match = re.match(r"([a-zA-Z_]+)(:|=|!=|>|<|>=|<=)(.+)", token)
        if field_match:
            field, operator, value = field_match.groups()
            
            # Special handling for status vs Status fields
            if field.lower() == "status_value" or field.lower() == "state":
                # This refers to the Status header field
                field = "Status"  # Proper capitalization for header field
            elif field.lower() == "status" and value.lower() in ["active", "pending", "completed", "in-progress", "blocked", "deferred"]:
                # If the value looks like a status value not a status folder, interpret as Status header
                field = "Status"
            
            # Convert operator to internal format
            if operator == ":":
                if field.lower() == "tags":
                    operator = "has_tag"
                elif field.lower() == "flags":
                    operator = "has_flag"
                else:
                    operator = "contains"
            
            # Handle regex
            if value.startswith("/") and value.endswith("/") and len(value) > 2:
                value = value[1:-1]
                operator = "matches"
            
            # Handle tag lists for tag search
            if field.lower() == "tags" and operator == "has_tag" and "," in value:
                # Split by comma and add each tag as a separate condition
                for tag in value.split(","):
                    tag = tag.strip()
                    if tag:
                        query.add_condition("Tags", "has_tag", tag)
            else:
                # Regular condition
                query.add_condition(field, operator, value)
        
        # Sort option
        elif token.startswith("sort:"):
            sort_field = token[5:]
            sort_reverse = sort_field.startswith("-")
            
            if sort_reverse:
                sort_field = sort_field[1:]
                
            query.set_sort(sort_field, sort_reverse)
        
        # Limit option
        elif token.startswith("limit:"):
            try:
                limit = int(token[6:])
                query.set_pagination(limit=limit)
            except ValueError:
                pass
        
        # Content option
        elif token == "with_content":
            query.with_content(True)
        
        # Plain keyword = search in subject and content
        else:
            keyword_tokens.append(token)
            
        i += 1
    
    # Add keyword search if any keywords were found
    if keyword_tokens:
        keyword = " ".join(keyword_tokens)
        query.add_condition("Subject", "contains", keyword)
        query.add_condition("content", "contains", keyword)
    
    return query

def print_search_results(results: List[Dict[str, Any]], output_format: str = "text") -> None:
    """
    Print search results in the specified format
    
    Args:
        results: Search results
        output_format: Output format ("text", "json", "csv", "compact")
    """
    if not results:
        print("No matching memories found.")
        return
        
    if output_format == "json":
        # Convert datetime objects to strings for JSON serialization
        json_results = []
        for memory in results:
            json_memory = memory.copy()
            
            # Convert date
            json_memory["metadata"] = memory["metadata"].copy()
            if isinstance(json_memory["metadata"]["date"], datetime):
                json_memory["metadata"]["date"] = json_memory["metadata"]["date"].isoformat()
                
            # Convert dates in headers
            json_memory["headers"] = memory["headers"].copy()
            for key, value in json_memory["headers"].items():
                if isinstance(value, datetime):
                    json_memory["headers"][key] = value.isoformat()
                    
            json_results.append(json_memory)
            
        print(json.dumps(json_results, indent=2))
        
    elif output_format == "csv":
        # Headers
        headers = ["ID", "Subject", "Date", "Folder", "Tags", "Priority", "Status", "Flags"]
        print(",".join(f'"{h}"' for h in headers))
        
        # Rows
        for memory in results:
            row = [
                memory["metadata"]["unique_id"],
                memory["headers"].get("Subject", "").replace('"', '""'),
                memory["metadata"]["date"].isoformat() if isinstance(memory["metadata"]["date"], datetime) else str(memory["metadata"]["date"]),
                memory["folder"] or "Inbox",
                memory["headers"].get("Tags", "").replace('"', '""'),
                memory["headers"].get("Priority", ""),
                memory["headers"].get("Status", ""),
                "".join(memory["metadata"]["flags"])
            ]
            print(",".join(f'"{str(r)}"' for r in row))
            
    elif output_format == "compact":
        # Simple line format for each memory
        for memory in results:
            subject = memory["headers"].get("Subject", "No subject")
            date = memory["metadata"]["date"]
            date_str = date.strftime("%Y-%m-%d") if isinstance(date, datetime) else str(date)
            folder = memory["folder"] or "Inbox"
            flags = "".join(memory["metadata"]["flags"])
            
            print(f"{memory['metadata']['unique_id']} - {date_str} - {subject} ({folder}) {flags}")
            
    else:  # text (default)
        from memdir_tools.cli import print_memory, COLORS, colorize
        
        # Print summary
        print(colorize(f"Found {len(results)} matching memories", "bold"))
        print("")
        
        # Print each memory
        for memory in results:
            print_memory(memory, "content" in memory)
            print("----------------------------------")

def main() -> None:
    """Main entry point"""
    # Create parser with detailed description
    parser = argparse.ArgumentParser(
        description="Advanced memory search with powerful filtering capabilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUERY SYNTAX EXAMPLES:
  "python learning"         Search for "python" and "learning" in subject/content
  "subject:python"          Search for "python" in subject field
  "tags:python,learning"    Search for memories with both tags
  "#python #learning"       Same as above using tag shortcuts
  "priority:high"           Filter by high priority
  "Status=active"           Filter by Status header value
  "status=cur"              Filter by Maildir status folder
  "date>2023-01-01"         Filter by date after Jan 1, 2023
  "date>now-7d"             Filter by date in the last 7 days
  "sort:date"               Sort by date (newest first)
  "sort:-date"              Sort by date (oldest first)
  "limit:10"                Show only 10 results

FIELD OPERATORS:
  field:value               Field contains value (case-insensitive)
  field=/regex/             Field matches regex pattern
  field=value               Field equals value exactly 
  field!=value              Field does not equal value
  field>value               Field is greater than value
  field<value               Field is less than value
  field>=value              Field is greater than or equal to value
  field<=value              Field is less than or equal to value

SPECIAL SHORTCUTS:
  #tag                      Shorthand for "tags:tag"
  +F                        Shorthand for "flags:F" (Flagged)
  +S                        Shorthand for "flags:S" (Seen)
  +P                        Shorthand for "flags:P" (Priority)
  +R                        Shorthand for "flags:R" (Replied)

RELATIVE DATES:
  now                       Current time
  now-7d                    7 days ago
  now+1w                    1 week from now
  now-1m                    1 month ago
  now-1y                    1 year ago

For more detailed documentation, see: /root/hacks/memdir_tools/SEARCH_README.md
"""
    )
    
    # Create argument groups for better organization
    query_group = parser.add_argument_group('Query Options')
    query_group.add_argument("query", nargs="*", help="Search query string with field operators and shortcuts")
    
    filter_group = parser.add_argument_group('Filter Options')
    filter_group.add_argument("-f", "--folder", help="Folder to search (default: all folders)")
    filter_group.add_argument("-s", "--status", choices=STANDARD_FOLDERS, help="Status folder (default: all statuses)")
    filter_group.add_argument("--subject", help="Search in subject field")
    filter_group.add_argument("--content", help="Search in content field")
    filter_group.add_argument("--tags", help="Search for tags (comma-separated)")
    filter_group.add_argument("--priority", choices=["high", "medium", "low"], help="Filter by priority")
    filter_group.add_argument("--status-filter", help="Filter by Status header value (e.g., active, completed)")
    filter_group.add_argument("--date-from", help="Filter by date range start (YYYY-MM-DD or relative)")
    filter_group.add_argument("--date-to", help="Filter by date range end (YYYY-MM-DD or relative)")
    filter_group.add_argument("--flags", help="Filter by flags (S=Seen, R=Replied, F=Flagged, P=Priority)")
    
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument("--format", choices=["text", "json", "csv", "compact"], default="text", 
                              help="Output format (text=full details, compact=one line per memory)")
    output_group.add_argument("--sort", help="Sort results by field (e.g., date, subject, priority)")
    output_group.add_argument("--reverse", action="store_true", help="Reverse sort order")
    output_group.add_argument("--limit", type=int, help="Limit number of results")
    output_group.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    output_group.add_argument("--with-content", action="store_true", help="Include memory content in results")
    output_group.add_argument("--debug", action="store_true", help="Show debug information")
    
    args = parser.parse_args()
    
    # Build query from command line arguments
    query = SearchQuery()
    
    # Add conditions from arguments
    if args.subject:
        query.add_condition("Subject", "contains", args.subject)
    
    if args.content:
        query.add_condition("content", "contains", args.content)
    
    if args.tags:
        for tag in args.tags.split(","):
            query.add_condition("Tags", "has_tag", tag.strip())
    
    if args.priority:
        query.add_condition("Priority", "=", args.priority)
    
    if args.status_filter:
        query.add_condition("Status", "=", args.status_filter)
    
    if args.date_from:
        query.add_condition("Date", ">=", args.date_from)
    
    if args.date_to:
        query.add_condition("Date", "<=", args.date_to)
    
    if args.flags:
        for flag in args.flags:
            query.add_condition("flags", "has_flag", flag)
    
    # Set sort parameters
    if args.sort:
        query.set_sort(args.sort, args.reverse)
    
    # Set pagination
    query.set_pagination(args.limit, args.offset)
    
    # Set content inclusion
    query.with_content(args.with_content)
    
    # Parse positional query arguments and add them to the query
    if args.query:
        parsed_query = parse_search_args(" ".join(args.query))
        query.conditions.extend(parsed_query.conditions)
        
        # Use parsed query sort if not set explicitly
        if args.sort is None and parsed_query.sort_by:
            query.sort_by = parsed_query.sort_by
            query.sort_reverse = parsed_query.sort_reverse
            
        # Use parsed query pagination if not set explicitly
        if args.limit is None and parsed_query.limit:
            query.limit = parsed_query.limit
            
        # Use parsed query content setting if not set explicitly
        if not args.with_content and parsed_query.include_content:
            query.include_content = True
    
    # Set folders and statuses
    folders = [args.folder] if args.folder else None
    statuses = [args.status] if args.status else None
    
    # Show debug info if requested
    if args.debug:
        print("Search query:")
        for condition in query.conditions:
            print(f"  - {condition['field']} {condition['operator']} {condition['value']}")
        print(f"Sort by: {query.sort_by or 'None'} {'(reverse)' if query.sort_reverse else ''}")
        print(f"Limit: {query.limit or 'None'}, Offset: {query.offset}")
        print(f"Include content: {query.include_content}")
        print(f"Folders: {folders or 'All'}")
        print(f"Statuses: {statuses or 'All'}")
        print("")
    
    # Perform search
    results = search_memories(query, folders, statuses, debug=args.debug)
    
    # Print results
    print_search_results(results, args.format)

if __name__ == "__main__":
    main()