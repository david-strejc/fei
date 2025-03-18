#!/usr/bin/env python3
"""
Integration example for FEI and Memdir
This script demonstrates how to use FEI with Memdir for memory management
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.core.assistant import Assistant
from fei.tools.memdir_connector import MemdirConnector

class MemdirFEIAssistant:
    """FEI assistant with Memdir integration"""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229", provider: str = "anthropic"):
        """
        Initialize the assistant
        
        Args:
            model: LLM model to use
            provider: LLM provider to use
        """
        # Initialize FEI assistant
        self.assistant = Assistant(model=model, provider=provider)
        
        # Initialize Memdir connector
        self.memdir = MemdirConnector()
        
        # Check Memdir connection
        if not self.memdir.check_connection():
            print("Warning: Cannot connect to Memdir server. Memory features will be disabled.")
            self.memdir_available = False
        else:
            self.memdir_available = True
    
    def chat(self):
        """Run interactive chat with memory integration"""
        history = []
        
        print("FEI Assistant with Memdir Integration")
        print("Type 'exit' to quit, 'save' to save the conversation as a memory,")
        print("'search <query>' to search memories, 'list' to list recent memories")
        print("-" * 50)
        
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for special commands
            if user_input.lower() == "exit":
                break
                
            elif user_input.lower() == "save":
                self._save_conversation(history)
                continue
                
            elif user_input.lower().startswith("search "):
                query = user_input[7:].strip()
                self._search_memories(query)
                continue
                
            elif user_input.lower() == "list":
                self._list_recent_memories()
                continue
            
            # Check if the input references a memory to include context
            memory_references = self._extract_memory_references(user_input)
            if memory_references:
                context = self._get_memory_context(memory_references)
                if context:
                    # Add the context to the user message
                    user_input = f"{user_input}\n\nRelevant context from memories:\n{context}"
            
            # Get assistant response
            response = self.assistant.chat(user_input)
            print(f"\nAssistant: {response}")
            
            # Save message pair to history
            history.append({"user": user_input, "assistant": response})
    
    def _save_conversation(self, history: List[Dict[str, str]]):
        """Save the conversation as a memory"""
        if not self.memdir_available or not history:
            print("No conversation to save or Memdir is not available.")
            return
        
        # Get memory details from user
        subject = input("Enter a subject for this memory: ")
        tags = input("Enter tags (comma-separated): ")
        priority = input("Enter priority (high/medium/low) [medium]: ") or "medium"
        folder = input("Enter target folder (leave empty for Inbox): ")
        
        # Format conversation as content
        content = "# Conversation\n\n"
        for msg in history:
            content += f"## User\n{msg['user']}\n\n## Assistant\n{msg['assistant']}\n\n"
        
        # Save to Memdir
        try:
            result = self.memdir.create_memory_from_conversation(
                subject, content, tags, priority, folder
            )
            print(f"Memory saved: {result['filename']}")
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def _search_memories(self, query: str):
        """Search memories and print results"""
        if not self.memdir_available:
            print("Memdir is not available.")
            return
        
        try:
            results = self.memdir.search(query, limit=5, with_content=True)
            
            print(f"\nFound {results['count']} memories matching '{query}':")
            print("-" * 50)
            
            for memory in results["results"]:
                print(f"ID: {memory['metadata']['unique_id']}")
                print(f"Subject: {memory['headers'].get('Subject', 'No subject')}")
                print(f"Date: {memory['metadata']['date']}")
                print(f"Tags: {memory['headers'].get('Tags', 'None')}")
                print(f"Folder: {memory['folder'] or 'Inbox'}")
                
                # Show truncated content if available
                if "content" in memory:
                    content_preview = memory["content"][:200] + "..." if len(memory["content"]) > 200 else memory["content"]
                    print(f"\nContent Preview:\n{content_preview}")
                
                print("-" * 50)
        except Exception as e:
            print(f"Error searching memories: {e}")
    
    def _list_recent_memories(self):
        """List recent memories"""
        if not self.memdir_available:
            print("Memdir is not available.")
            return
        
        try:
            # List memories in new and cur folders
            new_memories = self.memdir.list_memories(status="new")
            cur_memories = self.memdir.list_memories(status="cur")
            
            all_memories = new_memories + cur_memories
            # Sort by date (newest first)
            all_memories.sort(key=lambda x: x["metadata"]["timestamp"], reverse=True)
            
            print("\nRecent memories:")
            print("-" * 50)
            
            for memory in all_memories[:10]:  # Show only 10 most recent
                print(f"ID: {memory['metadata']['unique_id']}")
                print(f"Subject: {memory['headers'].get('Subject', 'No subject')}")
                print(f"Date: {memory['metadata']['date']}")
                print(f"Tags: {memory['headers'].get('Tags', 'None')}")
                print(f"Folder: {memory['folder'] or 'Inbox'}")
                print("-" * 50)
        except Exception as e:
            print(f"Error listing memories: {e}")
    
    def _extract_memory_references(self, text: str) -> List[str]:
        """
        Extract memory references from text (e.g., #mem:abc123)
        
        Args:
            text: Input text
            
        Returns:
            List of memory IDs
        """
        import re
        
        # Look for patterns like #mem:abc123 or {mem:abc123}
        pattern = r'(?:#mem:|{mem:)([a-z0-9]+)(?:})?'
        matches = re.findall(pattern, text)
        
        return matches
    
    def _get_memory_context(self, memory_ids: List[str]) -> str:
        """
        Get content from referenced memories
        
        Args:
            memory_ids: List of memory IDs
            
        Returns:
            Formatted context string
        """
        if not self.memdir_available:
            return ""
        
        context_parts = []
        
        for memory_id in memory_ids:
            try:
                memory = self.memdir.get_memory(memory_id)
                
                if memory:
                    subject = memory['headers'].get('Subject', 'No subject')
                    content = memory.get('content', '')
                    
                    context_parts.append(f"Memory: {subject} (ID: {memory_id})\n\n{content}\n")
            except:
                # Skip failed retrievals
                pass
        
        return "\n".join(context_parts)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FEI Assistant with Memdir Integration")
    parser.add_argument("--model", default="claude-3-sonnet-20240229", help="LLM model to use")
    parser.add_argument("--provider", default="anthropic", help="LLM provider to use")
    
    args = parser.parse_args()
    
    # Create and run the assistant
    assistant = MemdirFEIAssistant(model=args.model, provider=args.provider)
    assistant.chat()

if __name__ == "__main__":
    main()