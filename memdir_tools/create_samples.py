#!/usr/bin/env python3
"""
Create sample memories to demonstrate the Memdir system
"""

import os
import sys
import random
from datetime import datetime, timedelta
import textwrap

from memdir_tools.utils import (
    ensure_memdir_structure,
    save_memory,
    move_memory,
    update_memory_flags
)

# Data for random memory generation
SUBJECTS = [
    "Weekly Planning Session", "Meeting Notes: Product Team", "Book Review: {book}",
    "Research on {topic}", "Project Idea: {project}", "Learning Notes: {topic}",
    "Conference Notes: {event}", "Bug Report: {issue}", "Feature Request: {feature}",
    "Technical Design: {project}", "Interview with {person}", "Analysis of {topic}",
    "Quick Thoughts on {topic}", "Tutorial: How to {action}", "Summary of {event}",
    "Reflections on {topic}", "Brainstorming Session: {topic}", "Debugging Notes: {issue}",
    "Code Review: {project}", "Recipe: {food}"
]

TOPICS = [
    "Machine Learning", "Data Structures", "Algorithms", "Python", "JavaScript",
    "Rust", "Go", "Databases", "Cloud Computing", "Web Development", "DevOps",
    "Security", "Blockchain", "UI/UX Design", "Mobile Development", "Testing",
    "Big Data", "Microservices", "Docker", "Kubernetes", "React", "Angular",
    "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Natural Language Processing",
    "Computer Vision", "Reinforcement Learning", "Neural Networks", "Git", "CI/CD"
]

BOOKS = [
    "Clean Code", "The Pragmatic Programmer", "Design Patterns", "Refactoring",
    "Domain-Driven Design", "The Mythical Man-Month", "Soft Skills", "Code Complete",
    "Working Effectively with Legacy Code", "The Phoenix Project", "Accelerate",
    "Building Microservices", "Site Reliability Engineering", "The DevOps Handbook",
    "Continuous Delivery", "Patterns of Enterprise Application Architecture"
]

PROJECTS = [
    "Knowledge Management System", "Task Tracker", "Personal Finance App",
    "Social Network", "E-commerce Platform", "Content Management System",
    "API Gateway", "Authentication Service", "Data Pipeline", "Analytics Dashboard",
    "Search Engine", "Chat Application", "Recommendation System", "Mobile Game",
    "Productivity Tool", "Browser Extension", "Desktop Application"
]

EVENTS = [
    "PyCon 2024", "KubeCon", "AWS Summit", "Google I/O", "Apple WWDC",
    "GitHub Universe", "Docker Con", "React Conf", "DevOps Days", "Rust Conf",
    "Node Congress", "JS Conf", "MongoDB World", "PostgreSQL Conference",
    "Kafka Summit", "TensorFlow Dev Summit", "MLOps Summit"
]

PEOPLE = [
    "John Doe", "Jane Smith", "Elon Musk", "Satya Nadella", "Sundar Pichai",
    "Mark Zuckerberg", "Jensen Huang", "Sam Altman", "Andrew Ng", "Yann LeCun",
    "Martin Fowler", "Kent Beck", "Robert C. Martin", "Linus Torvalds", "Guido van Rossum"
]

TAGS = [
    "work", "personal", "research", "learning", "project", "idea", "meeting",
    "conference", "book", "coding", "design", "planning", "review", "tutorial",
    "howto", "bug", "feature", "documentation", "testing", "production",
    "development", "performance", "security", "code", "architecture", "database",
    "frontend", "backend", "devops", "ui", "ux", "mobile", "web", "desktop",
    "algorithm", "datastructure", "python", "javascript", "rust", "go", "react",
    "angular", "vue", "node", "django", "flask", "spring", "docker", "kubernetes",
    "aws", "azure", "gcp", "terraform", "ansible", "git", "cicd", "cloud", "agile"
]

PRIORITIES = ["high", "medium", "low"]
STATUSES = ["active", "pending", "completed", "in-progress", "blocked", "deferred"]

def generate_random_content(subject):
    """Generate random content based on the subject"""
    paragraphs = random.randint(3, 6)
    sections = random.randint(2, 4)
    
    content = [f"# {subject}"]
    content.append("")
    
    # Introduction
    intro_sentences = random.randint(2, 4)
    intro = []
    for _ in range(intro_sentences):
        words = random.randint(10, 20)
        sentence = " ".join(random.sample(TAGS + TOPICS, words))
        sentence = sentence.capitalize() + "."
        intro.append(sentence)
    content.append(" ".join(intro))
    content.append("")
    
    # Generate sections
    for i in range(sections):
        section_title = random.choice([
            "Overview", "Details", "Implementation", "Next Steps", "Background",
            "Summary", "Discussion", "Key Points", "Analysis", "Observations",
            "Questions", "Decisions", "Action Items", "Resources", "References"
        ])
        
        content.append(f"## {section_title}")
        content.append("")
        
        # Generate paragraphs for this section
        for _ in range(random.randint(1, 3)):
            paragraph_sentences = random.randint(3, 6)
            paragraph = []
            for _ in range(paragraph_sentences):
                words = random.randint(8, 16)
                sentence = " ".join(random.sample(TAGS + TOPICS, words))
                sentence = sentence.capitalize() + "."
                paragraph.append(sentence)
            content.append(" ".join(paragraph))
            content.append("")
            
        # Sometimes add a list
        if random.random() > 0.5:
            list_items = random.randint(3, 6)
            content.append("")
            for j in range(list_items):
                item = random.choice(TOPICS + PROJECTS + BOOKS)
                content.append(f"- {item}")
            content.append("")
    
    return "\n".join(content)

def generate_random_headers():
    """Generate random headers for a memory"""
    # Pick random subject template
    subject_template = random.choice(SUBJECTS)
    
    # Fill in template
    subject = subject_template
    if "{topic}" in subject:
        subject = subject.replace("{topic}", random.choice(TOPICS))
    if "{book}" in subject:
        subject = subject.replace("{book}", random.choice(BOOKS))
    if "{project}" in subject:
        subject = subject.replace("{project}", random.choice(PROJECTS))
    if "{event}" in subject:
        subject = subject.replace("{event}", random.choice(EVENTS))
    if "{person}" in subject:
        subject = subject.replace("{person}", random.choice(PEOPLE))
    if "{issue}" in subject:
        subject = subject.replace("{issue}", f"Issue #{random.randint(100, 999)}")
    if "{feature}" in subject:
        subject = subject.replace("{feature}", f"{random.choice(PROJECTS)} {random.choice(['Integration', 'Export', 'Import', 'View', 'Editor', 'Dashboard'])}")
    if "{action}" in subject:
        subject = subject.replace("{action}", f"{random.choice(['Build', 'Deploy', 'Configure', 'Optimize', 'Debug', 'Test', 'Design', 'Implement'])} {random.choice(PROJECTS)}")
    if "{food}" in subject:
        subject = subject.replace("{food}", random.choice(["Pasta", "Pizza", "Salad", "Soup", "Sandwich", "Curry", "Stir-fry"]))
    
    # Select random tags (2-5 tags)
    num_tags = random.randint(2, 5)
    tags = ",".join(random.sample(TAGS, num_tags))
    
    # Other headers
    priority = random.choice(PRIORITIES)
    status = random.choice(STATUSES)
    
    # Dates
    today = datetime.now()
    random_days = random.randint(-30, 30)
    random_date = today + timedelta(days=random_days)
    
    headers = {
        "Subject": subject,
        "Tags": tags,
        "Priority": priority,
        "Status": status,
        "Date": random_date.isoformat()
    }
    
    # Sometimes add due date
    if random.random() > 0.7:
        due_days = random.randint(1, 90)
        due_date = today + timedelta(days=due_days)
        headers["Due"] = due_date.isoformat()
    
    # Sometimes add other headers
    if random.random() > 0.8:
        headers["Author"] = random.choice(PEOPLE)
        
    if random.random() > 0.9:
        headers["Version"] = f"{random.randint(0, 2)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    
    return headers, subject

def create_samples(count=20):
    """Create sample memories in different folders with various metadata"""
    # Ensure the directory structure exists
    ensure_memdir_structure()
    
    # Create sample folders
    for folder in [".Projects/Python", ".Projects/AI", ".ToDoLater/Learning", ".Archive/2023"]:
        folder_path = os.path.join(os.getcwd(), "Memdir", folder)
        for status in ["cur", "new", "tmp"]:
            os.makedirs(os.path.join(folder_path, status), exist_ok=True)
    
    # Create standard examples first
    create_standard_samples()
    
    # Create random memories
    created_count = 7  # We already created 7 standard examples
    memory_ids = []
    
    # Create remaining random memories
    for _ in range(count - created_count):
        # Decide where to put this memory
        folder_choices = ["", ".Projects/Python", ".Projects/AI", ".ToDoLater/Learning"]
        folder = random.choice(folder_choices)
        
        # Generate random content
        headers, subject = generate_random_headers()
        content = generate_random_content(subject)
        
        # Decide on flags
        flags = ""
        if random.random() > 0.7:
            flag_choices = list("FRSP")
            num_flags = random.randint(0, 3)
            if num_flags > 0:
                flags = "".join(random.sample(flag_choices, num_flags))
        
        # Save the memory
        memory_id = save_memory(folder, content, headers, flags)
        memory_ids.append(memory_id)
        
        # Move most to cur, but leave some in new
        if random.random() > 0.2:
            move_memory(memory_id, folder, folder, "new", "cur")
    
    print(f"Created {count} sample memories in various folders")
    return memory_ids

def create_standard_samples():
    """Create the standard example memories"""
    # Sample 1: Python learning memory
    python_memory = """
# Python Learning Notes

## Key Concepts
- Everything in Python is an object
- Functions are first-class citizens
- Dynamic typing with strong type enforcement
- Comprehensive standard library

## Code Examples
```python
# List comprehension
squares = [x**2 for x in range(10)]

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}

# Generator expression
sum_of_squares = sum(x**2 for x in range(100))
```

## Resources
- Official Python docs: https://docs.python.org
- Real Python: https://realpython.com
- Python Cookbook by David Beazley
"""
    
    python_headers = {
        "Subject": "Python Learning Notes",
        "Tags": "python,learning,programming,notes",
        "Priority": "high",
        "Status": "active",
        "References": ""
    }
    
    python_id = save_memory(".Projects/Python", python_memory, python_headers, "F")
    # Move to cur folder
    move_memory(python_id, ".Projects/Python", ".Projects/Python", "new", "cur")
    
    # Sample 2: AI research memory
    ai_memory = """
# Transformer Architecture Research

## Key Components
- Self-attention mechanism
- Positional encodings
- Layer normalization
- Residual connections
- Feed-forward networks

## Recent Developments
- Mixture of Experts (MoE) for scaling
- FlashAttention for efficiency
- Sparse attention patterns
- Retrieval-augmented generation

## Papers to Read
- "Attention is All You Need" (original Transformer paper)
- "GPT-4 Technical Report"
- "Scaling Laws for Neural Language Models"
- "Training language models to follow instructions"
"""
    
    ai_headers = {
        "Subject": "Transformer Architecture Research",
        "Tags": "ai,transformers,research,llm",
        "Priority": "medium",
        "Status": "active",
        "References": ""
    }
    
    ai_id = save_memory(".Projects/AI", ai_memory, ai_headers, "S")
    # Move to cur folder
    move_memory(ai_id, ".Projects/AI", ".Projects/AI", "new", "cur")
    
    # Sample 3: To-do later memory
    todo_memory = """
# Books to Read

## Technical
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Clean Code" by Robert C. Martin
- "The Pragmatic Programmer" by Andrew Hunt and David Thomas

## Fiction
- "Project Hail Mary" by Andy Weir
- "The Three-Body Problem" by Liu Cixin
- "Dune" by Frank Herbert

## Philosophy
- "Thinking, Fast and Slow" by Daniel Kahneman
- "Gödel, Escher, Bach" by Douglas Hofstadter
"""
    
    todo_headers = {
        "Subject": "Books to Read",
        "Tags": "books,reading,learning,todo",
        "Priority": "low",
        "Status": "pending",
        "Due": (datetime.now() + timedelta(days=90)).isoformat()
    }
    
    todo_id = save_memory(".ToDoLater/Learning", todo_memory, todo_headers)
    # Move to cur folder
    move_memory(todo_id, ".ToDoLater/Learning", ".ToDoLater/Learning", "new", "cur")
    
    # Sample 4: Archived memory
    archive_memory = """
# 2023 Learning Goals

## Completed
- [x] Learn Python basics
- [x] Complete SQL course
- [x] Build a simple web application
- [x] Create a personal portfolio

## Partially Completed
- [~] Read 20 technical books (15/20)
- [~] Contribute to open source (2/5 PRs)

## Not Started
- [ ] Learn Rust programming
- [ ] Complete ML certification
"""
    
    archive_headers = {
        "Subject": "2023 Learning Goals",
        "Tags": "goals,2023,learning,completed",
        "Priority": "low",
        "Status": "archived",
        "Completion": "75%"
    }
    
    archive_id = save_memory(".Archive/2023", archive_memory, archive_headers, "SR")
    # Move to cur folder
    move_memory(archive_id, ".Archive/2023", ".Archive/2023", "new", "cur")
    
    # Sample 5: Regular inbox memory
    inbox_memory = """
# Project Ideas for 2025

## AI Tools
- Memory management system with Maildir-like structure
- Code assistant with repository understanding
- Personal knowledge graph with automatic connections

## Data Processing
- Streaming data pipeline with real-time analytics
- Document processing system with semantic search
- Multi-modal content analyzer

## Web Applications
- Personal dashboard for productivity tracking
- Knowledge management system with bi-directional links
- API aggregator with unified interface
"""
    
    inbox_headers = {
        "Subject": "Project Ideas for 2025",
        "Tags": "projects,ideas,planning,2025",
        "Priority": "medium",
        "Status": "active"
    }
    
    inbox_id = save_memory("", inbox_memory, inbox_headers, "F")
    # Move to cur folder
    move_memory(inbox_id, "", "", "new", "cur")
    
    # Sample 6: Memory with references
    ref_memory = """
# Memory System Implementation Notes

## Key Components
- Maildir-compatible structure
- Header-based metadata
- Flag system for status tracking
- Hierarchical organization
- Cross-reference support

## Implementation Tasks
- Create core utility functions ✅
- Implement CLI interface ✅
- Add search capabilities ✅
- Set up filtering rules ⏳
- Create automatic organization system ⏳

## Integration with Existing Tools
- Consider using mu/mu4e as a reference
- Look at notmuch for tag-based organization
- maildir-utils package provides good abstractions
"""
    
    ref_headers = {
        "Subject": "Memory System Implementation Notes",
        "Tags": "memory,maildir,implementation,notes",
        "Priority": "high",
        "Status": "in-progress",
        "References": f"<{inbox_id}>"
    }
    
    ref_id = save_memory("", ref_memory, ref_headers, "FR")
    # Move to cur folder
    move_memory(ref_id, "", "", "new", "cur")
    
    # Create a memory in .Trash to demonstrate
    trash_memory = """
# Outdated Information

This memory contains information that is no longer relevant or has been superseded by newer memories.

Please refer to the latest documentation for up-to-date information.
"""
    
    trash_headers = {
        "Subject": "Outdated Information",
        "Tags": "outdated,obsolete",
        "Priority": "low",
        "Status": "deleted",
        "DeletedDate": datetime.now().isoformat()
    }
    
    trash_id = save_memory(".Trash", trash_memory, trash_headers, "S")
    # Move to cur folder
    move_memory(trash_id, ".Trash", ".Trash", "new", "cur")

if __name__ == "__main__":
    # Get count from command line if provided
    count = 20
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            pass
            
    create_samples(count)