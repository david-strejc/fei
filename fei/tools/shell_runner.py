"""
Shell runner for Fei.

This module provides a secure shell command execution system with support for
interactive commands, long-running tasks, and proper process management.
"""

import os
import re
import signal
import time
import threading
import subprocess
from typing import Dict, List, Optional, Union, Tuple, Any, Set, Callable

from fei.utils.logging import get_logger
from fei.core.process_manager import process_manager

logger = get_logger(__name__)

# Constants
DEFAULT_COMMAND_TIMEOUT = 60  # seconds
MAX_OUTPUT_SIZE = 50000


class ShellRunner:
    """
    Tool for executing shell commands with advanced security and process management.
    
    This class provides a secure shell command execution system with support for
    interactive commands, long-running tasks, and proper process management.
    """

    # Global allowlist of safe commands
    ALLOWED_COMMANDS = {
        # File system (non-destructive)
        "ls", "find", "cat", "head", "tail", "less", "more", "grep", "tree", "stat", "du",
        "file", "whereis", "which", "locate", "pwd", "dirname", "basename", "realpath",
        "cd", # Allow changing directories

        # File management (potentially destructive but needed)
        "mkdir", "touch", "rm", "cp", "mv", "ln", "chmod", "chown", "tar", "zip", "unzip",
        "gzip", "gunzip", "bzip2", "bunzip2", "rsync",

        # Process management
        "ps", "top", "htop", "kill", "pkill", "pgrep", "nice", "renice", "time",

        # Network (read-only)
        "ping", "traceroute", "dig", "host", "nslookup", "netstat", "ss", "ifconfig",
        "ip", "arp", "route", "wget", "curl",

        # Text processing
        "awk", "sed", "cut", "paste", "tr", "sort", "uniq", "wc", "diff", "patch",
        "grep", "egrep", "fgrep", "tee",

        # Development
        "git", "svn", "hg", "make", "gcc", "g++", "clang", "javac", "python", "python3",
        "pip", "pip3", "npm", "node", "ruby", "perl", "php", "go", "rust", "cargo",
        "mvn", "gradle", "ant", "cmake", "docker", "kubectl", "helm", "terraform",

        # System info
        "uname", "hostname", "uptime", "free", "df", "mount", "lsblk", "lsusb", "lspci",
        "lsmod", "lshw", "dmidecode", "dmesg", "journalctl", "systemctl",

        # Utilities
        "xargs", "watch", "yes", "sleep", "timeout", "printf", "open", "bc", "hexdump", "xxd"
    }

    # Denylist of dangerous commands
    DENIED_COMMANDS = {
        # Dangerous system commands
        "rm -rf /", "rm -fr /", "dd if=", "mkfs", "shutdown", "reboot", "poweroff",
        "passwd", "sudo", "su", "chroot", "crontab",

        # Network attacks
        "nc ", "ncat ", "telnet ", "ssh ", "nmap ", "tcpdump ", "wireshark ",

        # Dangerous scripting
        "eval ", "exec ", "`", "$(",

        # Web access (allow these with proper arguments)
        "wget --post", "curl --data", "lynx --dump", "w3m -dump",

        # File system manipulation (add spaces to avoid false positives)
        " > ", " >> ", " | "
    }

    def __init__(
        self,
        default_timeout: int = DEFAULT_COMMAND_TIMEOUT,
        max_output_size: int = MAX_OUTPUT_SIZE,
        enforce_allowlist: bool = True,
        sandbox: bool = True,
        do_as_i_say: bool = False
    ):
        """
        Initialize shell runner.
        
        Args:
            default_timeout: Default command timeout in seconds
            max_output_size: Maximum output size in characters
            enforce_allowlist: Whether to enforce the command allowlist
            sandbox: Whether to use sandboxing
            do_as_i_say: Whether to bypass all command checks (DANGEROUS!)
        """
        self.default_timeout = default_timeout
        self.max_output_size = max_output_size
        self.enforce_allowlist = enforce_allowlist
        self.sandbox = sandbox
        self.do_as_i_say = do_as_i_say

    def _is_allowed_command(self, command: str, do_as_i_say: bool = False) -> Tuple[bool, str]:
        """
        Check if a command is allowed.
        
        Args:
            command: Command to check
            do_as_i_say: Whether to bypass all command checks
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        # Bypass checks if do_as_i_say is True
        if do_as_i_say or self.do_as_i_say:
            logger.warning(f"DANGEROUS: Bypassing command checks for: {command}")
            return True, "Checks bypassed with do_as_i_say"

        # Check for empty command
        if not command or not command.strip():
            return False, "Empty command"

        # Check for dangerous web commands
        if "wget --post" in command or "curl --data" in command:
            return False, "Command contains potentially dangerous web request with data"

        # Split command by common command separators to handle compound commands
        command_parts = re.split(r'\s*(?:&&|\|\||;)\s*', command)

        for part in command_parts:
            part = part.strip()
            if not part:
                continue

            # Extract the main command (before any arguments)
            main_command = part.split()[0] if part.split() else ""

            # Handle command paths (e.g., /usr/bin/ls)
            if "/" in main_command:
                main_command = os.path.basename(main_command)

            # Check if main command is allowed
            if main_command not in self.ALLOWED_COMMANDS:
                return False, f"Command not in allowlist: {main_command}"

            # Check for denied patterns
            for denied in self.DENIED_COMMANDS:
                if denied in part:
                    return False, f"Command contains denied pattern: {denied}"

            # Check for shell redirections and pipes
            if re.search(r'[^\\](?:>|>>|\|)', part):
                return False, "Command contains redirection or pipe"

        return True, "Command allowed"

    def is_interactive_command(self, command: str) -> bool:
        """
        Check if a command is likely to be interactive.
        
        Args:
            command: Command to check
            
        Returns:
            Whether the command is likely to be interactive
        """
        # List of keywords that suggest interactive applications
        interactive_keywords = [
            # GUI applications
            "gui", "gtk", "gnome", "x11", "wayland", "qt", "tkinter", "curses",
            # Interactive tools
            "vim", "nano", "emacs", "less", "more", "top", "htop",
            # Interpreters and REPLs
            "python", "ipython", "jupyter", "node", "irb", "pry",
            # Development servers
            "npm start", "npm run", "flask run", "rails server",
            # Games
            "game", "pygame", "play",
            # Interactive shells
            "bash", "zsh", "fish", "sh", "ksh", "csh",
            # Interactive applications
            "watch", "dialog", "whiptail", "ncurses", "tmux", "screen",
            # Interactive debuggers
            "gdb", "pdb", "lldb", "jdb", "dbx",
            # Interactive database clients
            "mysql", "psql", "sqlite3", "mongo", "redis-cli",
            # Interactive network tools
            "ssh", "telnet", "nc", "netcat",
        ]

        # Check if any keyword is in the command
        command_lower = command.lower()
        for keyword in interactive_keywords:
            if keyword in command_lower:
                return True
        
        # Check for specific patterns that indicate interactive commands
        interactive_patterns = [
            # Python with -i flag
            r'python\s+(?:-i|--interactive)',
            # Node.js with -i flag
            r'node\s+(?:-i|--interactive)',
            # Commands with -i flag
            r'\s-i\s',
            # Commands with --interactive flag
            r'\s--interactive\s',
        ]
        
        for pattern in interactive_patterns:
            if re.search(pattern, command):
                return True
        
        return False

    def run_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        background: Optional[bool] = None,
        capture_stderr: bool = True,
        env: Optional[Dict[str, str]] = None,
        do_as_i_say: bool = False
    ) -> Dict[str, Any]:
        """
        Run a shell command with extensive safety features.
        
        Args:
            command: Shell command to execute
            timeout: Command timeout in seconds (overrides default)
            background: Force running in background (for interactive commands)
            capture_stderr: Whether to capture stderr
            env: Environment variables
            do_as_i_say: Whether to bypass all command checks (DANGEROUS!)
            
        Returns:
            Dict with stdout, stderr, exit_code, and error (if any)
        """
        if not command:
            return {
                "success": False,
                "error": "Command is required",
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }

        # Check if command is allowed (bypass checks if do_as_i_say is True)
        is_allowed, reason = self._is_allowed_command(command, do_as_i_say)
        if not is_allowed:
            return {
                "success": False,
                "error": f"Command not allowed: {reason}",
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }

        # Set timeout
        cmd_timeout = timeout or self.default_timeout

        # Auto-detect if command should run in background
        if background is None:
            background = self.is_interactive_command(command)

        # For interactive or background commands, use process_manager
        if background:
            logger.info(f"Running command in background mode: {command}")

            try:
                # Spawn process using process_manager
                process_info = process_manager.spawn_process(
                    command=command,
                    env=env,
                    timeout=cmd_timeout,
                    is_interactive=True
                )

                # Return immediately with process information
                return {
                    "success": True,
                    "stdout": f"Process started in background (PID: {process_info['pid']}).",
                    "stderr": "",
                    "exit_code": None,
                    "background": True,
                    "pid": process_info["pid"],
                    "note": f"Process running in background. Will be terminated after {cmd_timeout} seconds if not finished."
                }
            except Exception as e:
                logger.error(f"Error executing background command: {e}", exc_info=True)
                return {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": "",
                    "exit_code": -1
                }

        # For non-interactive commands, use subprocess.run
        try:
            # Run the command with timeout
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=cmd_timeout,
                env=env or os.environ.copy()
            )

            # Get output
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode

            # Truncate output if too large
            if len(stdout) > self.max_output_size:
                stdout = stdout[:self.max_output_size] + "\n... [output truncated]"
            if len(stderr) > self.max_output_size:
                stderr = stderr[:self.max_output_size] + "\n... [output truncated]"

            return {
                "success": exit_code == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Command timed out after {cmd_timeout} seconds",
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }

        except Exception as e:
            logger.error(f"Error executing command: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": "",
                "exit_code": -1
            }


# Global instance
shell_runner = ShellRunner()


def initialize_shell_runner(do_as_i_say: bool = False) -> None:
    """
    Initialize or update the shell_runner with command-line arguments.
    
    Args:
        do_as_i_say: Whether to bypass all command checks (DANGEROUS!)
    """
    global shell_runner
    # Create a new ShellRunner instance with the do_as_i_say flag
    shell_runner = ShellRunner(do_as_i_say=do_as_i_say)
    if do_as_i_say:
        logger.warning("DANGEROUS: Command security checks are bypassed with --do-as-i-say flag!")
