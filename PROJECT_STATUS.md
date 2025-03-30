# Fei Agent Project Status - 2025-03-30

## Summary

This document outlines the current status of the Fei agent project, recent debugging efforts, known issues, and the plan for future development, focusing on achieving autonomous task execution and self-evolution capabilities.

## Recent Debugging & Findings (Snake Game Task)

We tested the agent's ability to autonomously generate a Snake game using Pygame via the command-line interface (`--task` mode) with the Google Gemini provider. This involved several debugging steps:

1.  **API Key Verification:** Confirmed the `GOOGLE_API_KEY` was valid using `curl` but initially failed within the agent.
2.  **LiteLLM Integration:** Fixed `litellm` authentication errors by explicitly passing the `api_key` in `fei/core/assistant.py`.
3.  **Model Compatibility:** Switched from potentially unstable experimental models (`gemini/gemini-2.5-pro`, `gemini/gemini-2.5-pro-exp-03-25`) back to `gemini/gemini-1.5-pro-latest` due to rate limits and inconsistent behavior.
4.  **Dependency Issues:** Installed missing dependencies (`textual-autocomplete`) and resolved import errors (`Dropdown` import removed from `fei/ui/textual_chat.py`).
5.  **Tool Call Parsing (Google Gemini):** Identified that the Gemini model often describes the tool action in text rather than generating the correct structured tool call format expected by `litellm`. This required significant prompt refinement to explicitly name the tool (`Replace`) and its arguments (`file_path`, `content`).
6.  **Tool Handler Bug:** Fixed a `ValueError: too many values to unpack (expected 2)` in `fei/tools/handlers.py`'s `replace_handler` because it wasn't correctly handling the 3 return values (`success`, `message`, `backup_path`) from `code_editor.replace_file`.
7.  **Task Executor Logic:** Refactored the `TaskExecutor` loop in `fei/core/task_executor.py` to correctly handle iterations involving tool calls and continuation messages. Also corrected the initial prompt used in `--task` mode to instruct autonomous execution rather than waiting for user input.

## Current State

*   The agent can be invoked via the CLI (`python -m fei --task "..."`).
*   It can use the Google Gemini provider (`gemini/gemini-1.5-pro-latest`).
*   With explicit prompting, it can generate code iteratively and *attempt* to use the `Replace` tool to write/overwrite files.
*   The `Replace` tool handler is now correctly implemented.
*   Basic logging (`fei_agent.log`) and debug logging (`--debug`) are functional.
*   The agent successfully generated the first ~5 steps of the Snake game code and called the `Replace` tool multiple times in the last run, creating `snake_game_final.py`.

## Known Issues & Bugs

1.  **Incomplete Task Execution:** In the last run, the agent stopped after 5 iterations despite the task requiring ~11 steps and the `[TASK_COMPLETE]` signal not being present. The `TaskExecutor` might still have issues with its loop termination or iteration logic.
2.  **LLM Tool Use Reliability (Gemini):** The Gemini model still seems unreliable in consistently generating correctly formatted tool calls, even with detailed prompts. It often defaults to text descriptions.
3.  **Interactive Shell Commands:** The agent currently lacks a mechanism to handle long-running or interactive shell commands effectively (e.g., running the generated Pygame). The `Shell` tool executes commands, but feedback/control for interactive sessions is missing.
4.  **MCP Cleanup Errors (Tests):** Console logs during `pytest` runs show `RuntimeError: no running event loop` and `ValueError: I/O operation on closed file` originating from `fei/core/mcp.py`'s cleanup function. This needs investigation, although it doesn't seem to affect CLI execution directly.
5.  **Tool Call Error Handling:** The agent got stuck asking for a corrected tool when the `Replace` tool handler failed due to the unpacking error. Error handling within the `TaskExecutor` could be more robust.

## TODOs / Next Steps

1.  **Complete Snake Game Task:** Re-run the agent with increased `--max-iterations` (e.g., 15) to allow it to finish all 11 steps of the Snake game generation. Verify the final `snake_game_final.py` is complete and functional.
2.  **Debug TaskExecutor Loop:** Investigate why the `TaskExecutor` stopped prematurely in the last successful run (reported completion after 1 iteration despite multiple tool calls logged) and in the limited iteration run (stopped at max iterations without finishing).
3.  **Improve Tool Call Reliability:**
    *   Further refine prompts for Gemini tool use.
    *   Consider adding logic to the `Assistant` or `TaskExecutor` to parse code/actions from text responses as a fallback if structured tool calls fail.
    *   Evaluate if other models (GPT-4o, Claude 3) exhibit more reliable tool use behavior for complex tasks.
4.  **Handle Interactive Shell Commands:** Implement a strategy for managing interactive processes spawned by the `Shell` tool. (See Plan below).
5.  **Fix MCP Cleanup Errors:** Debug the `fei/core/mcp.py` cleanup logic to resolve the event loop/logging errors during test teardown.
6.  **Integrate Self-Evolution Framework:** Begin implementing the self-evolution stages using Memdir.

## Plan: Handling Interactive Shell Commands

The current `Shell` tool executes commands but doesn't provide a mechanism for interaction or managing long-running processes effectively (like a game).

**Proposed Solution:**

1.  **Background Execution:** Modify the `Shell` tool handler (`fei/tools/handlers.py` -> `shell_handler`) and the underlying `ShellRunner.run_command` in `fei/tools/code.py`:
    *   **Detect Interactive:** Enhance `is_interactive_command` or add logic to detect commands likely requiring interaction (e.g., running a `pygame` script).
    *   **Force Background:** When an interactive command is detected, *always* run it in the background using `subprocess.Popen` with `start_new_session=True` (as partially implemented).
    *   **Return PID:** The tool result *must* reliably return the process ID (PID) of the background process.
    *   **No Blocking:** Ensure the handler returns immediately after launching the background process, providing the PID and status information.
2.  **Process Management Tool(s):** Introduce new tools for managing these background processes:
    *   `view_process_output <pid>`: Reads recent stdout/stderr from the background process's pipes (might require temporary file redirection or more advanced IPC).
    *   `send_process_input <pid> <input_string>`: Sends input to the process's stdin.
    *   `check_process_status <pid>`: Checks if the process is still running.
    *   `terminate_process <pid>`: Sends SIGTERM (and potentially SIGKILL) to the process group.
3.  **Agent Logic:** The agent needs to be prompted or learn to:
    *   Recognize when a command should be run interactively.
    *   Use the `Shell` tool to start it in the background.
    *   Use the process management tools (`view_process_output`, `check_process_status`, `terminate_process`) to monitor and control the background task based on the overall goal.

## Plan: Self-Evolution Framework (using Memdir)

Integrate the proposed self-evolution framework, adapting it to use the project's existing Memdir system.

1.  **Long-Term Memory (Memdir):**
    *   **Storage:** Utilize the existing `memdir_tools` capabilities. Define specific Memdir folders (e.g., `.Evolution`, `.Knowledge`, `.History`) for storing:
        *   Evolution stage progress (`stage_id`, `timestamp`, `status`).
        *   Learned behaviors (code patterns, successful tool sequences, error resolutions) tagged appropriately (e.g., `#learning`, `#pattern`).
        *   Operational history (tasks attempted, tools used, errors, outcomes) tagged `#history`.
    *   **Schema:** Use structured JSON within Memdir entries. Implement versioning if schema evolves significantly. Leverage Memdir's tagging and search for retrieval.
2.  **Stage Management:**
    *   **Immutable Stages:** Store core stage logic in `fei/evolution/stages/stage_N.py`. Each stage module could define modifications or additions to the `Assistant`, `ToolRegistry`, or `TaskExecutor`.
    *   **Metadata:** Use a `fei/evolution/manifest.json` (or store in Memdir) to define stage details, validation criteria, and allowed transitions.
3.  **Evolution Process:**
    *   **Stage Transition:**
        *   **Checkpointing:** Before attempting transition, use `memory_create_handler` to save critical state (current stage, key variables, relevant memory IDs) to a checkpoint memory entry in Memdir (e.g., folder `.Evolution/Checkpoints`).
        *   **Parallel Execution:** Modify the agent's main entry point or a dedicated orchestrator script to spawn a new instance for the next stage (`subprocess.Popen(["python", "-m", "fei", "--stage", str(next_stage), ...])`). Both instances would need access to Memdir.
        *   **Validation:** The new stage instance runs predefined validation tasks (could be specific prompts directed at itself, running unit tests via the `Shell` tool, comparing outputs on benchmark tasks stored in Memdir). Validation results are logged to Memdir.
        *   **Rollback:** If validation fails, the orchestrator (or the main instance upon restart) reads the latest valid checkpoint from Memdir and reverts relevant state/code. Code rollback might involve checking out a previous Git commit hash stored in the checkpoint or replacing specific modules.
    *   **Safe Switch:**
        *   Use Memdir entries (e.g., in `.Evolution/Status`) to manage the "Traffic Light System" (`current_stage`, `next_stage_candidate`, `health_status`).
        *   The switch logic resides in the orchestrator or the agent's boot process, reading status from Memdir.
4.  **Operational Workflow:**
    *   **Boot Process:** Agent reads `current_stage` from Memdir status. Loads corresponding stage module (`fei/evolution/stages/stage_{current_stage}.py`). Loads relevant knowledge/state from Memdir based on stage.
    *   **Evolution Triggers:** Implement logic to check Memdir for manual trigger entries or evaluate automatic triggers based on operational history stored in Memdir (e.g., count successful tasks tagged `#history`).
5.  **Security & Safety:**
    *   **Root Access:** Use the existing `Shell` tool's allowlist/denylist. Enhance logging for sensitive commands.
    *   **Sandboxing:** Explore running stage candidates in Docker containers if feasible, mapping necessary volumes for code access and Memdir.
6.  **Monitoring & Logging:**
    *   Log all significant events (task start/end, tool calls, errors, stage transitions, validation results) to both the file log (`fei_agent.log`) and Memdir (tagged `#history` or `#evolution_log`).
    *   Implement alerting based on Memdir queries (e.g., check for recent `#evolution_log` entries with `status: failed`).

This provides a comprehensive overview and a path forward.
