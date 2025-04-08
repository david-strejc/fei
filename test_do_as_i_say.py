#!/usr/bin/env python3
"""
Test script to verify the --do-as-i-say option for bypassing command checks.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the ShellRunner class
from fei.tools.code import ShellRunner

def test_command(shell_runner, command, do_as_i_say=False):
    """Test if a command is allowed and print the result."""
    is_allowed, reason = shell_runner._is_allowed_command(command, do_as_i_say)
    result = "ALLOWED" if is_allowed else "DENIED"
    print(f"Command: '{command}' (do_as_i_say={do_as_i_say}) -> {result}")
    if not is_allowed:
        print(f"  Reason: {reason}")
    return is_allowed

def main():
    """Main function to test the --do-as-i-say option."""
    # Create a ShellRunner instance
    shell_runner = ShellRunner()
    
    # Test commands that should normally be denied
    denied_commands = [
        "rm -rf /",
        "sudo apt-get install package",
        "ssh user@host",
        "eval 'echo dangerous'",
        "cat file.txt | grep pattern",  # Contains pipe
        "echo 'Hello' > file.txt",      # Contains redirect
        "cd fei_test_dir && python data_processor.py",  # Contains pipe
    ]
    
    # Test without do_as_i_say
    print("=== Testing commands WITHOUT do_as_i_say ===")
    without_results = [test_command(shell_runner, cmd, False) for cmd in denied_commands]
    
    # Test with do_as_i_say
    print("\n=== Testing commands WITH do_as_i_say ===")
    with_results = [test_command(shell_runner, cmd, True) for cmd in denied_commands]
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Without do_as_i_say: {sum(without_results)}/{len(denied_commands)} allowed")
    print(f"With do_as_i_say: {sum(with_results)}/{len(denied_commands)} allowed")
    
    # Test with ShellRunner instance having do_as_i_say=True
    print("\n=== Testing with ShellRunner(do_as_i_say=True) ===")
    shell_runner_bypass = ShellRunner(do_as_i_say=True)
    bypass_results = [test_command(shell_runner_bypass, cmd) for cmd in denied_commands]
    print(f"With ShellRunner(do_as_i_say=True): {sum(bypass_results)}/{len(denied_commands)} allowed")

if __name__ == "__main__":
    main()
