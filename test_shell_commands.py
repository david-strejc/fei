#!/usr/bin/env python3
"""
Test script to verify the updated shell command validation logic.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the ShellRunner class
from fei.tools.code import ShellRunner

def test_command(shell_runner, command):
    """Test if a command is allowed and print the result."""
    is_allowed, reason = shell_runner._is_allowed_command(command)
    result = "ALLOWED" if is_allowed else "DENIED"
    print(f"Command: '{command}' -> {result}")
    if not is_allowed:
        print(f"  Reason: {reason}")
    return is_allowed

def main():
    """Main function to test various commands."""
    # Create a ShellRunner instance
    shell_runner = ShellRunner()
    
    # Test commands that should be allowed
    allowed_commands = [
        "cd fei_test_dir",
        "cd fei_test_dir && python data_processor.py",
        "python fei_test_dir/data_processor.py",
        "cat data.txt",
        "ls -la",
        "mkdir -p test_dir",
        "rm -rf test_dir",
        "grep 'pattern' file.txt",
        "find . -name '*.py'",
        "python -m http.server",
        "python3 -c 'print(\"Hello, world!\")'",
        "echo 'Hello, world!'",
        "wget https://example.com",
        "curl https://example.com",
    ]
    
    # Test commands that should be denied
    denied_commands = [
        "rm -rf /",
        "sudo apt-get install package",
        "ssh user@host",
        "eval 'echo dangerous'",
        "cat file.txt | grep pattern",  # Contains pipe
        "echo 'Hello' > file.txt",      # Contains redirect
        "wget --post-data='data' https://example.com",
        "curl --data 'data' https://example.com",
    ]
    
    # Test allowed commands
    print("=== Testing commands that should be allowed ===")
    allowed_results = [test_command(shell_runner, cmd) for cmd in allowed_commands]
    
    # Test denied commands
    print("\n=== Testing commands that should be denied ===")
    denied_results = [test_command(shell_runner, cmd) for cmd in denied_commands]
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Allowed commands: {sum(allowed_results)}/{len(allowed_commands)}")
    print(f"Denied commands: {len(denied_commands) - sum(denied_results)}/{len(denied_commands)}")
    
    # Check for unexpected results
    unexpected_allowed = [cmd for cmd, result in zip(allowed_commands, allowed_results) if not result]
    unexpected_denied = [cmd for cmd, result in zip(denied_commands, denied_results) if result]
    
    if unexpected_allowed:
        print("\nUnexpected denied commands:")
        for cmd in unexpected_allowed:
            print(f"  - {cmd}")
    
    if unexpected_denied:
        print("\nUnexpected allowed commands:")
        for cmd in unexpected_denied:
            print(f"  - {cmd}")

if __name__ == "__main__":
    main()
