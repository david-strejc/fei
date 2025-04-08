#!/usr/bin/env python3
"""
Test script to verify the --do-as-i-say command-line option.
"""

import sys
import os
import argparse

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

# Import the necessary modules
from fei.tools.code import shell_runner, initialize_shell_runner

def test_command(command, do_as_i_say=False):
    """Test if a command is allowed and print the result."""
    # Initialize the shell_runner with the do_as_i_say flag
    initialize_shell_runner(do_as_i_say=do_as_i_say)

    # Check if the command is allowed
    # Note: We need to explicitly pass do_as_i_say to the method
    is_allowed, reason = shell_runner._is_allowed_command(command, do_as_i_say)
    result = "ALLOWED" if is_allowed else "DENIED"
    print(f"Command: '{command}' (do_as_i_say={do_as_i_say}) -> {result}")
    if not is_allowed:
        print(f"  Reason: {reason}")
    return is_allowed

def main():
    """Main function to test the --do-as-i-say command-line option."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Test the --do-as-i-say command-line option")
    parser.add_argument("--do-as-i-say", action="store_true", help="Bypass all command checks")
    args = parser.parse_args()

    # Test commands that should normally be denied
    denied_commands = [
        "rm -rf /",
        "sudo apt-get install package",
        "ssh user@host",
        "eval 'echo dangerous'",
        "cat file.txt | grep pattern",  # Contains pipe
        "echo 'Hello' > file.txt",      # Contains redirect
    ]

    # Test with the command-line option
    print(f"=== Testing commands with --do-as-i-say={args.do_as_i_say} ===")
    results = [test_command(cmd, args.do_as_i_say) for cmd in denied_commands]

    # Print summary
    print("\n=== Summary ===")
    print(f"Commands allowed: {sum(results)}/{len(denied_commands)}")

if __name__ == "__main__":
    main()
