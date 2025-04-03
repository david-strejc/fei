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

## Architecture & Code Review (2025-04-03)

Based on a review of core files (`assistant.py`, `task_executor.py`, `registry.py`, `mcp.py`) and MCP documentation:

*   **Strengths:**
    *   Modular design separating core logic, tools, MCP handling, and UI.
    *   Multi-LLM support via LiteLLM.
    *   Comprehensive built-in tool suite (`fei/tools/code.py`).
    *   Flexible configuration loading (config file, env vars).
    *   Basic MCP client structure supporting stdio/HTTP and service abstractions.
*   **Weaknesses/Flaws:**
    *   **Critical MCP Security Gap:** The `fei/core/mcp.py` client lacks explicit user consent checks before accessing resources or executing tools via MCP, deviating significantly from MCP specification security guidelines. This poses a major risk for autonomous operation and self-evolution.
    *   **Fragile Task Execution:** The `TaskExecutor` loop relies heavily on the LLM to maintain task state via conversation history ("Continue..."), which could be unreliable for complex, long-running tasks.
    *   **Problematic Process Cleanup:** The use of `atexit` for cleaning up asynchronous MCP server processes (`fei/core/mcp.py`) is known to be unreliable and likely causes the observed test errors.
    *   **Basic Server Readiness Check:** Stdio MCP server startup check relies on a fixed `asyncio.sleep`, which isn't robust.
    *   **LLM Tool Reliability:** Consistent and accurate tool call generation by LLMs (especially Gemini) remains a challenge.

## Known Issues & Bugs

1.  **Critical - Missing MCP Security/Consent Flows:** The `fei/core/mcp.py` client currently lacks explicit user consent checks before accessing resources or executing tools via MCP. This is a critical deviation from the MCP specification's Security and Trust & Safety guidelines and must be addressed before enabling broader MCP interactions, especially for self-evolution.
2.  **Incomplete Task Execution:** In the last run, the agent stopped after 5 iterations despite the task requiring ~11 steps and the `[TASK_COMPLETE]` signal not being present. The `TaskExecutor` loop might still have issues with its loop termination or iteration logic, potentially related to its reliance on the LLM for state management.
3.  **LLM Tool Use Reliability (Gemini):** The Gemini model still seems unreliable in consistently generating correctly formatted tool calls, even with detailed prompts. It often defaults to text descriptions.
4.  **Interactive Shell Commands:** The agent currently lacks a mechanism to handle long-running or interactive shell commands effectively (e.g., running the generated Pygame). The `Shell` tool executes commands, but feedback/control for interactive sessions is missing.
5.  **MCP Cleanup Errors (Tests):** Console logs during `pytest` runs show `RuntimeError: no running event loop` and `ValueError: I/O operation on closed file`. This likely originates from the problematic use of `atexit` for cleaning up asynchronous MCP server processes in `fei/core/mcp.py`'s `ProcessManager`.
6.  **Tool Call Error Handling:** The agent got stuck asking for a corrected tool when the `Replace` tool handler failed due to the unpacking error. Error handling within the `TaskExecutor` could be more robust.

## TODOs / Next Steps

1.  **Implement MCP Consent/Security Flows (Highest Priority):** Design and integrate explicit user confirmation steps (potentially configurable) within the agent's workflow before accessing MCP resources or executing MCP tools. Ensure adherence to MCP specification security guidelines. This is critical before enabling broader autonomous actions or self-evolution involving external data/tools.
2.  **Refactor `ProcessManager` Cleanup:** Investigate and implement alternative strategies to `atexit` for reliably cleaning up asynchronous MCP server processes on agent exit (e.g., using signal handlers or a dedicated management process).
3.  **Improve MCP Server Readiness Check:** Implement a more robust handshake or health check mechanism for stdio servers instead of relying solely on `asyncio.sleep`.
4.  **Complete Snake Game Task:** Re-run the agent with increased `--max-iterations` (e.g., 15) to allow it to finish all 11 steps of the Snake game generation. Verify the final `snake_game_final.py` is complete and functional. *(Note: This was partially completed, but a full run is still needed for verification)*.
5.  **Debug TaskExecutor Loop:** Investigate why the `TaskExecutor` stopped prematurely in past runs and assess the fragility of its state management approach.
6.  **Improve Tool Call Reliability:**
    *   Further refine prompts for Gemini tool use.
    *   Consider adding logic to the `Assistant` or `TaskExecutor` to parse code/actions from text responses as a fallback if structured tool calls fail.
    *   Evaluate if other models (GPT-4o, Claude 3) exhibit more reliable tool use behavior for complex tasks.
7.  **Handle Interactive Shell Commands:** Implement a strategy for managing interactive processes spawned by the `Shell` tool. (See Plan below).
8.  **Integrate Self-Evolution Framework:** Begin implementing the self-evolution stages using Memdir, ensuring MCP security flows are addressed first.

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

## Plan: LLM Evaluation & Cost Optimization Workflow

To manage operational costs effectively, especially as the agent evolves and potentially spawns sub-agents for specific tasks, Fei needs a mechanism to dynamically select the most appropriate LLM based on cost and capability.

**Proposed Strategy:**

1.  **Model Benchmarking:**
    *   Define a suite of representative benchmark tasks relevant to Fei's core functions (e.g., code generation, file manipulation, tool use accuracy, planning, self-correction).
    *   Run these benchmarks periodically across various available LLMs (e.g., different versions of GPT, Claude, Gemini, Groq Llama, Mistral, etc.) supported by LiteLLM.
    *   Record performance metrics (e.g., task success rate, code quality score, execution time, tool use success rate) and associated costs (token usage, API call cost) for each model on each task.
    *   Store benchmark results and up-to-date cost data in Memdir (e.g., folder `.Knowledge/LLM_Benchmarks`).
2.  **Dynamic Model Routing:**
    *   Implement a "Model Router" component within the `Assistant` or `TaskExecutor`.
    *   Before initiating an LLM call, the router analyzes the current task/sub-task context (e.g., complexity, required capabilities like tool use, conversation history length).
    *   The router queries the benchmark data in Memdir to identify suitable models that meet the task requirements within potential cost constraints.
    *   It selects the most cost-effective model that meets the performance threshold for the specific task context. For example:
        *   Use cheaper/faster models (e.g., Claude 3 Haiku, Gemini Flash, Groq Llama-8b) for simple queries, planning, summarization, or initial drafts.
        *   Use more capable models (e.g., GPT-4o, Claude 3 Opus/Sonnet, Gemini 1.5 Pro) for complex code generation, refactoring, debugging, or critical decision-making steps.
3.  **Cost Tracking & Budgeting:**
    *   Utilize LiteLLM's cost tracking capabilities for each API call.
    *   Log cost information alongside operational history in Memdir.
    *   Implement optional budget constraints per task or time period, influencing the model router's decisions.
4.  **Feedback Loop:**
    *   Analyze task success/failure rates correlated with the chosen model (using Memdir history).
    *   Use this data to refine the model selection criteria and potentially trigger re-benchmarking if a model's performance degrades.
5.  **Sub-Agent Specialization:**
    *   As the agent evolves (potentially using the Self-Evolution Framework), allow different stages or specialized sub-agents (spawned for specific complex tasks) to have different default model preferences or routing logic based on their function.
