# Shell Command Execution System Implementation Plan

## Overview

This document outlines the plan for implementing a comprehensive shell command execution system in Fei. The system will handle various scenarios including interactive commands, long-running tasks, and proper process management.

## Current Limitations

1. **Output Retrieval**: The current system doesn't properly capture output from background processes.
2. **Interactive Commands**: While there's detection for interactive commands, the implementation for handling them is incomplete.
3. **Process Management**: The process management tools are mostly placeholders.
4. **Input Handling**: There's limited support for sending input to running processes.

## Design Goals

1. **Robust Process Management**: Properly track and manage all spawned processes.
2. **Output Capture**: Capture stdout/stderr from background processes in real-time.
3. **Interactive Input**: Allow sending input to interactive processes.
4. **Graceful Termination**: Properly terminate processes when requested or on system exit.
5. **Security**: Maintain the existing security features while improving functionality.

## Implementation Plan

### Phase 1: Core Process Management

1. ✅ Analyze current implementation
2. ✅ Design new architecture
3. ⬜ Create `ProcessManager` class
   - Implement process tracking
   - Implement output capture via temporary files
   - Implement input handling
   - Implement process termination
4. ⬜ Update `ShellRunner` to use the new `ProcessManager`

### Phase 2: Tool Integration

1. ⬜ Update process management tools
   - `view_process_output`
   - `send_process_input`
   - `check_process_status`
   - `terminate_process`
2. ⬜ Add new tools if needed
   - `list_processes`
   - `attach_to_process`

### Phase 3: UI Integration

1. ⬜ Add UI components for process management
   - Process list view
   - Process detail view
   - Process input/output view
2. ⬜ Implement process management in CLI and Textual interfaces

### Phase 4: Testing and Documentation

1. ⬜ Create test cases for various scenarios
   - Interactive commands
   - Long-running tasks
   - Process termination
   - Error handling
2. ⬜ Document the new system
   - API documentation
   - User guide
   - Examples

## Technical Design

### ProcessManager Class

```python
class ProcessManager:
    """Manages processes spawned by Fei."""
    
    def __init__(self, temp_dir=None, max_output_size=1024*1024):
        """Initialize the process manager."""
        self.processes = {}  # pid -> process info
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.max_output_size = max_output_size
        self._lock = threading.RLock()
        
    def spawn_process(self, command, env=None, cwd=None, timeout=None):
        """Spawn a new process and track it."""
        # Implementation details...
        
    def get_process_output(self, pid, max_lines=100):
        """Get the output of a process."""
        # Implementation details...
        
    def send_process_input(self, pid, input_str):
        """Send input to a process."""
        # Implementation details...
        
    def check_process_status(self, pid):
        """Check if a process is running."""
        # Implementation details...
        
    def terminate_process(self, pid, force=False, timeout=3.0):
        """Terminate a process."""
        # Implementation details...
        
    def list_processes(self):
        """List all tracked processes."""
        # Implementation details...
        
    def cleanup(self):
        """Clean up all tracked processes."""
        # Implementation details...
```

### Process Information Structure

```python
{
    "pid": 1234,
    "command": "python -m http.server",
    "start_time": 1617123456.789,
    "process": <subprocess.Popen object>,
    "stdout_file": "/tmp/fei_process_1234_stdout.txt",
    "stderr_file": "/tmp/fei_process_1234_stderr.txt",
    "input_file": "/tmp/fei_process_1234_input.txt",
    "timeout": 3600,
    "status": "running",  # running, terminated, exited
    "exit_code": None,
    "is_interactive": True
}
```

## Progress Tracking

- [x] Analyze current implementation
- [x] Design new architecture
- [ ] Implement ProcessManager class
- [ ] Update ShellRunner
- [ ] Update process management tools
- [ ] Add UI components
- [ ] Create test cases
- [ ] Document the new system
