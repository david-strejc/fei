# Fei 2.0 Codebase Evaluation Plan

## 1. Overall Goal

Evaluate the `fei_2.0` codebase for implementation quality, data flow correctness, adherence to purpose, documentation accuracy, and test coverage. Identify areas for improvement and update documentation and tests accordingly.

## 2. Analysis Phases

The evaluation will be conducted in phases, focusing on different modules of the application:

### Phase 1: Core Components (`fei/core/`)
*   **Files:** `assistant.py`, `conversation_manager.py`, `enhanced_model_optimizer.py`, `mcp.py`, `mediator.py`, `model_optimizer.py`, `orchestrator.py`, `process_manager.py`, `provider_client.py`, `task_executor.py`, `tool_executor.py`
*   **Evaluation Aspects:**
    *   Code clarity, maintainability, and adherence to Python best practices.
    *   Design patterns used (e.g., Mediator, Orchestrator).
    *   Interaction and data flow between core components.
    *   Error handling and resilience.
    *   Efficiency and potential bottlenecks.
*   **Tasks:**
    *   [x] Analyze `assistant.py` for main interaction logic.
    *   [x] Analyze `conversation_manager.py` for state management.
    *   [x] Analyze `orchestrator.py` and `mediator.py` for control flow.
    *   [x] Analyze `task_executor.py` and `tool_executor.py` for execution logic.
    *   [x] Analyze `process_manager.py` for external process handling.
    *   [x] Analyze model optimization components.
    *   [x] Map data flow through the core system.
    *   [x] Identify potential improvements in core logic.

### Phase 2: Memory System (`fei/memory/`, `memdir_tools/`)
*   **Files:** `fei/memory/base.py`, `compression.py`, `embeddings.py`, `manager.py`, `memdir.py`, `vector_store.py`, `memdir_tools/*`
*   **Evaluation Aspects:**
    *   Memory abstraction (`base.py`).
    *   Integration with Memdir (`memdir.py`, `memdir_tools`).
    *   Vector store implementation and usage.
    *   Embedding generation and handling.
    *   Memory compression techniques.
    *   Data flow for storing and retrieving information.
    *   API design and usage (`memdir_tools/server.py`).
*   **Tasks:**
    *   [x] Analyze `fei/memory/manager.py` for overall memory orchestration.
    *   [x] Analyze `memdir.py` connector.
    *   [x] Analyze `vector_store.py` implementation.
    *   [x] Review `memdir_tools` CLI and server implementation.
    *   [x] Verify data consistency and retrieval logic.
    *   [x] Suggest improvements for memory efficiency or structure.

### Phase 3: Tool System (`fei/tools/`)
*   **Files:** `category.py`, `code.py`, `definitions.py`, `enhanced_registry.py`, `handlers.py`, `lifecycle.py`, `memdir_connector.py`, `memory_tools.py`, `memorychain_connector.py`
*   **Evaluation Aspects:**
    *   Tool definition and registration (`definitions.py`, `enhanced_registry.py`).
    *   Tool execution lifecycle (`lifecycle.py`, `handlers.py`).
    *   Integration with memory (`memory_tools.py`, `memdir_connector.py`).
    *   Code generation/analysis tools (`code.py`).
    *   Extensibility and modularity of the tool system.
*   **Tasks:**
    *   [x] Analyze tool registration and discovery.
    *   [x] Analyze tool execution flow and error handling.
    *   [x] Review specific tool implementations (e.g., memory, code).
    *   [x] Evaluate the design of tool categories and lifecycle.
    *   [x] Identify potential new tools or improvements to existing ones.

### Phase 4: Model Management (`fei/models/`, `fei/providers/`)
*   **Files:** `fei/models/*`, `fei/providers/*`
*   **Evaluation Aspects:**
    *   Model abstraction and configuration (`fei/models/base.py`, `config.py`).
    *   Model selection logic (`selector.py`, `registry.py`).
    *   Provider abstraction and integration (`fei/providers/base.py`, `litellm_provider.py`, `manager.py`).
    *   Fallback mechanisms (`fei/models/fallback.py`).
    *   API key management and security considerations.
*   **Tasks:**
    *   [x] Analyze the model registry and selection process.
    *   [x] Review provider implementations, especially LiteLLM integration.
    *   [x] Evaluate the configuration and management of different models.
    *   [x] Check error handling for provider API calls.
    *   [x] Suggest improvements for model/provider flexibility or robustness.

### Phase 5: Evolution System (`fei/evolution/`)
*   **Files:** `loader.py`, `manifest.json`, `path_manager.py`, `state_machine.py`, `state.py`, `stages/*`
*   **Evaluation Aspects:**
    *   State machine design and implementation.
    *   Loading and management of evolution stages.
    *   Manifest usage for defining evolution paths.
    *   Clarity and purpose of different evolution stages.
*   **Tasks:**
    *   [x] Analyze the state machine logic (`state_machine.py`, `state.py`).
    *   [x] Review the loading mechanism (`loader.py`, `manifest.json`).
    *   [x] Evaluate the design and implementation of individual stages.
    *   [x] Assess the overall purpose and effectiveness of the evolution system.

### Phase 6: MCP System (`fei/mcp/`)
*   **Files:** `base.py`, `manager.py`
*   **Evaluation Aspects:**
    *   MCP base definition and manager logic.
    *   Integration with core components (Tool Executor, Task Executor).
    *   Handling of MCP tool calls and responses.
*   **Tasks:**
    *   [x] Analyze `manager.py` for MCP server communication.
    *   [x] Review the `base.py` abstraction.
    *   [x] Verify correct integration with the tool execution system.

### Phase 7: UI (`fei/ui/`)
*   **Files:** (Currently seems empty or minimal, needs confirmation)
*   **Evaluation Aspects:**
    *   If UI components exist (e.g., Textual), evaluate their implementation.
    *   Interaction with the core Fei system.
*   **Tasks:**
    *   [x] List files in `fei/ui/` to confirm contents.
    *   [x] Analyze its structure and connection to the backend.

### Phase 8: Utilities (`fei/utils/`)
*   **Files:** (Needs listing)
*   **Evaluation Aspects:**
    *   Purpose and quality of utility functions.
    *   Potential for reuse or refactoring.
*   **Tasks:**
    *   [x] List files in `fei/utils/`
    *   [x] Review utility functions for clarity and correctness.

### Phase 9: Testing (`tests/`, `fei/tests/`, `fei_test_dir/`, `memdir_tools/tests/`)
*   **Files:** All files within test directories.
*   **Evaluation Aspects:**
    *   Test coverage for different modules.
    *   Quality and effectiveness of unit, integration, and end-to-end tests.
    *   Use of fixtures and mocking.
    *   Maintainability of the test suite.
*   **Tasks:**
    *   [ ] Run test suite to check current status (`pytest`).
    *   [ ] Analyze test coverage reports (if available/generatable).
    *   [ ] Review tests for key components (core, memory, tools).
    *   [ ] Identify gaps in test coverage.
    *   [ ] **Revise/add tests** based on code analysis findings.

### Phase 10: Documentation (`docs/`, `*.md`)
*   **Files:** All files in `docs/` and top-level Markdown files.
*   **Evaluation Aspects:**
    *   Accuracy and completeness of documentation.
    *   Clarity and organization.
    *   Consistency with the current codebase.
    *   Presence of diagrams and examples.
*   **Tasks:**
    *   [ ] Review `README.md`.
    *   [ ] Review `docs/architecture.md` and other key documents.
    *   [ ] Compare documentation against actual implementation from previous phases.
    *   [ ] **Update documentation** to reflect the current state and findings.

### Phase 11: Configuration and Setup (`config/`, `requirements.txt`, `setup.py`, `*.sh`)
*   **Files:** `.env.example`, `requirements.txt`, `setup.py`, `manage_fei_keys.sh`, `run_fei.sh`, `test_env_key.sh`, `test_fei_shell_task.sh`
*   **Evaluation Aspects:**
    *   Clarity of configuration (`.env.example`).
    *   Correctness of dependencies (`requirements.txt`).
    *   Package setup (`setup.py`).
    *   Usefulness and correctness of shell scripts.
*   **Tasks:**
    *   [ ] Review `requirements.txt` for pinned/unpinned versions.
    *   [ ] Analyze `setup.py`.
    *   [ ] Review shell scripts for clarity and function.

## 3. Execution

1.  Follow the phases outlined above.
2.  For each file/module, read the code, understand its purpose and interactions.
3.  Document findings, potential issues, and suggested improvements directly in this plan or separate files (e.g., `IMPROVEMENTS.md`, `FLAWS.md`).
4.  Mark tasks as complete [x] as they are finished.
5.  After analysis phases, consolidate findings.
6.  Implement necessary code improvements (if approved/required).
7.  Update documentation (Phase 10).
8.  Revise tests (Phase 9).

## 4. Findings and Improvements Log

*(This section will be populated during the analysis)*

*   **Core:**
    *   **`assistant.py`:** Contains redundant local definitions for `ProviderManager`, `ToolManager`, and `ConversationManager`. These conflict/overlap with dedicated core components and should be removed/refactored.
    *   **`assistant.py`:** Orchestration logic (LLM calls, tool processing, continuation) is present in `Assistant` class methods (`_send_message_to_llm`, `process_tool_calls`, etc.). This logic should be moved to `AssistantOrchestrator` for better separation of concerns, simplifying the `Assistant` facade.
    *   **`assistant.py`:** Hardcoded model switching logic for memory tools in `process_tool_calls` could be abstracted or made configurable.
    *   **`conversation_manager.py`:** Well-structured for managing conversation state and formatting messages. Correctly handles provider differences for tool results.
    *   **`conversation_manager.py`:** Suggestion: Improve encapsulation of `last_message_id` by adding a setter method instead of allowing external direct attribute access.
    *   **`orchestrator.py`:** Effectively centralizes the core chat interaction flow (LLM calls, tool processing, continuation). Good use of delegation to other components.
    *   **`mediator.py`:** Solid implementation of the Mediator pattern, facilitating decoupled communication between components. Includes specific handlers and a general broadcast mechanism.
    *   **`orchestrator.py` / `assistant.py`:** Duplicate logic for model switching during memory tool execution exists in both files. This should be consolidated, likely within the `Orchestrator` or potentially abstracted further (e.g., into `ToolExecutor`).
    *   **`tool_executor.py`:** Well-designed component for handling tool execution. Correctly differentiates between registry and MCP tools, interacts with `MCPManager`, and includes good error handling and mediator notifications.
    *   **`task_executor.py`:** The autonomous `execute_task` mode has **major issues**. It bypasses the public `assistant.chat()` interface, directly calls private assistant methods, and duplicates orchestration logic. It needs refactoring to use only `assistant.chat()` for each step. The interactive mode (`execute_interactive`) seems correctly implemented.
    *   **`process_manager.py`:** Robust implementation for spawning, tracking, interacting with, and terminating external processes. Uses threads for non-blocking I/O to temp files, handles process groups, timeouts, and cleanup via `atexit`.
    *   **`process_manager.py`:** Caution: Uses `shell=True` which requires careful command handling. Suggestion: Implement output file size capping during writing in the reader thread to prevent disk exhaustion by verbose processes. Consider dependency injection instead of global instance.
    *   **`model_optimizer.py`:** Implements dynamic model selection via live evaluation of candidate models based on cost/latency. Uses caching. Relies on `litellm` for evaluation and cost data. Model discovery is basic.
    *   **`enhanced_model_optimizer.py`:** Implements an alternative optimization approach by delegating selection to `ModelSwitcher` based on task type (basic keyword detection) and strategy (cost/performance). Faster (no live evaluation) but relies entirely on `ModelSwitcher`'s data/logic.
    *   **Model Optimizers:** Coexistence/relationship between `ModelOptimizer` and `EnhancedModelOptimizer` is unclear. `Assistant` currently uses `ModelOptimizer`. The enhanced version depends heavily on the unanalyzed `ModelSwitcher`. Task type detection in enhanced version is rudimentary.
    *   **`mcp.py`:** Provides `MCPClient` for low-level communication (stdio/http, consent) and `MCPManager` facade with service classes (`MCPMemoryService`, etc.). Handles server config/discovery and lifecycle for stdio servers via an internal `asyncio`-based `ProcessManager`.
    *   **`mcp.py` / `process_manager.py`:** Contains a duplicate `ProcessManager` concept using `asyncio`, distinct from the `threading`-based one in `fei/core/process_manager.py`. This needs consolidation.
    *   **Core Data Flow:** User Input -> Assistant -> Orchestrator -> ConversationManager (add msg) -> ModelOptimizer -> ToolExecutor (get tools) -> ProviderClient (send msg) -> LLM -> ProviderClient (recv response) -> Orchestrator -> [Tool Loop if needed: ToolExecutor -> MCPManager/ToolRegistry -> ProviderClient (continuation)] -> ConversationManager (add response) -> Assistant -> User. (Mediator used for side notifications). `TaskExecutor` autonomous mode currently breaks this flow.
    *   **Core Improvements Identified:**
        *   Refactor `assistant.py` (remove local classes, move orchestration logic).
        *   Refactor `task_executor.py` (use public `assistant.chat()` only).
        *   Consolidate the two `ProcessManager` implementations.
        *   Consolidate memory tool model-switching logic.
        *   Clarify `ModelOptimizer` vs `EnhancedModelOptimizer` strategy.
        *   Improve `ConversationManager.last_message_id` encapsulation.
        *   Review `shell=True` usage for security.
        *   Enhance process output capping (during write).
        *   Use dependency injection for global `process_manager`.
        *   Review late `litellm` imports in `ProviderClient`.
*   **Memory:**
    *   **`manager.py`:** Provides a good abstraction layer using a Factory pattern (`MemoryFactory`) to manage and interact with different memory system backends (`MemoryInterface`). Lazily instantiates systems and delegates operations (CRUD, search) to the active instance.
    *   **`manager.py`:** Suggestion: Consider raising an error during initialization if the configured default memory system fails to load, instead of just logging.
    *   **`memdir.py`:** Implements `MemoryInterface` using HTTP calls (`requests`) to a Memdir server. Includes logic to auto-start/stop a local server via `subprocess.Popen` using class-level state.
    *   **`memdir.py`:** **Major Issue:** Uses synchronous `requests`, blocking I/O in an async application. Should use `httpx` or similar async client.
    *   **`memdir.py`:** Uses a different, synchronous process management approach (`subprocess`, class variable) compared to the `asyncio`-based one in `mcp.py`. Needs consolidation.
    *   **`memdir.py`:** Class-level server management is fragile. `atexit` cleanup is potentially unreliable. Update logic (delete+create) is complex and potentially risky.
    *   **`vector_store.py`:** Implements a very basic in-memory vector store with JSON file persistence. Uses an `EmbeddingProvider` for vector generation and cosine similarity for search.
    *   **`vector_store.py`:** **Major Scalability Issue:** Loads all vectors into memory and uses sequential search. Inefficient JSON persistence (saves all on each change). Suitable only for small demos/tests. Needs replacement with a proper vector DB (ChromaDB, FAISS, etc.) for real use. Synchronous file I/O within async methods.
    *   **`memdir_tools/cli.py`:** Provides a direct command-line interface to Memdir file operations (CRUD, search, flag, mkdir) using utility functions. Separate from the HTTP API. Suffers from potentially inefficient memory lookups by ID.
    *   **`memdir_tools/server.py`:** Implements a Flask-based HTTP REST API for Memdir, using API key auth. Wraps the same utility functions as the CLI. **Major Issue:** Synchronous (Flask + file I/O), limiting concurrency. Also has inefficient memory lookups by ID.
    *   **Data Consistency:** Relies on shared utility functions (`memdir_tools/utils.py`, etc.) used by both CLI and server. Potential inconsistencies arise from: synchronous server operations (concurrency risks), inefficient ID lookups, complex update logic in `fei/memory/memdir.py` connector (delete+create), and potential bugs in shared utils. True verification requires analyzing utils and integration testing.
    *   **Memory Improvements Identified:**
        *   Replace `VectorStore` with a proper vector database integration.
        *   Make `MemdirMemory` connector asynchronous (`httpx`).
        *   Use consolidated process manager for Memdir server auto-start.
        *   Refactor Memdir server management (remove class state).
        *   Make Memdir server asynchronous (FastAPI, `aiofiles`).
        *   Implement efficient Memdir ID lookups (indexing).
        *   Improve Memdir update logic (atomic server API or safer connector logic).
        *   Use explicit cleanup for Memdir server process, not `atexit`.
        *   Make `MemoryManager` initialization more robust on default system failure.
*   **Tools:**
    *   **`registry.py`:** Provides core `ToolRegistry` for managing local tools (registration, discovery). Includes helper `register_class_methods` for convenience. Handles basic argument validation and sync/async handler execution.
    *   **`registry.py`:** **CRITICAL Issue:** Duplicates MCP tool handling logic (`_handle_mcp_tool`) also found in `ToolExecutor`. This logic should be removed from the registry; `ToolExecutor` should be solely responsible for routing between the registry (for local tools) and `MCPManager` (for MCP tools).
    *   **`registry.py`:** Argument validation is basic; consider using `jsonschema` library. Async execution handling (`_execute_async_tool`) is complex and might be simplified if the caller manages the event loop correctly. Schema generation from class methods is basic.
    *   **`handlers.py`:** Contains the concrete implementation functions for tools. They parse arguments, call underlying logic (from `code.py`, `repomap.py`, `shell_runner`), handle errors with try/except blocks, and format results. Process management handlers are complex and tightly coupled to `shell_runner` state.
    *   **`code.py`:** Implements core logic for file/code tools (GlobFinder, GrepTool, CodeEditor, FileViewer, DirectoryExplorer, SystemInfo, ShellRunner). Uses threading for some parallel operations (glob ignore, grep). `CodeEditor` includes validation (AST, linters) and backups. `ShellRunner` has allow/denylists, background process handling (via `subprocess`/`threading`), and timeout logic.
    *   **`code.py` / Process Management:** `ShellRunner` introduces a *third* distinct approach to process management (`subprocess`/`threading`), separate from `fei/core/process_manager.py` and `fei/core/mcp.py`. Needs urgent consolidation. `atexit` cleanup used.
    *   **`code.py` / Security:** `ShellRunner`'s security relies on allow/denylists and basic parsing, which can be fragile with `shell=True`. Sandboxing mentioned but not implemented. `do_as_i_say` flag is dangerous.
    *   **`category.py`:** Defines `ToolCategoryRegistry` for grouping tools. Creates default categories. Overlaps conceptually with `tags` in `ToolRegistry`. Integration/usage within the system is unclear.
    *   **`lifecycle.py`:** Defines `ToolLifecycleManager` for registering tool initializer/cleanup functions and managing basic state. Integration into the application's startup/shutdown or tool execution flow is unclear.
    *   **Tool System Improvements Identified:**
        *   Remove MCP logic from `ToolRegistry`.
        *   Consolidate the three Process Management implementations.
        *   Enhance `ShellRunner` security (parsing, `shell=True`, sandboxing).
        *   Improve argument validation (`jsonschema`).
        *   Clarify/Integrate/Simplify `ToolCategoryRegistry` & `ToolLifecycleManager`.
        *   Simplify async tool execution handling in `ToolRegistry`.
        *   Document validation tool dependencies.
    *   **Potential New Tools:** `RefactorCodeTool`, `RunTestsTool`, `DependencyAnalysisTool`, `GitTool`, `DatabaseTool`, `DockerTool`, `EnhancedMemoryTool`.
*   **Model Management:**
    *   **`registry.py`:** Provides `ModelRegistry` to store static `ModelInfo` (capabilities, cost, etc.) loaded from config. Allows finding models meeting basic `ModelRequirement`.
    *   **`selector.py`:** Defines `BaseModelSelector` and strategy subclasses (Cost, Performance, Balanced, Random) to choose the "best" model from a list provided by the registry based on stored metadata. Includes basic fallback logic.
    *   **Selection/Optimization:** The relationship/usage pattern between `ModelRegistry`/`ModelSelector` (metadata-based selection), `ModelOptimizer` (live evaluation), and `EnhancedModelOptimizer` (delegation to `ModelSwitcher`) is unclear. Effectiveness depends heavily on accurate `ModelInfo` data. Selection metrics are basic.
    *   **`providers/base.py`:** Defines a clear `LLMProviderInterface` abstract base class and a `ProviderFactory` for managing different provider implementations. Good abstraction design.
    *   **`providers/litellm_provider.py`:** Implements the `LLMProviderInterface` using `litellm` library. **However,** its logic (config, API keys, `litellm.completion` call, response parsing) is almost entirely duplicated in `core/provider_client.py`.
    *   **`models/config.py`:** Manages model-specific configuration (metadata defaults, selector/fallback settings, default provider/model) within the main `Config` object.
    *   **`providers/manager.py`:** Correctly uses the `ProviderFactory` and `LLMProviderInterface` to get configured provider instances.
    *   **Configuration/Management Summary:** Model metadata is stored statically (loaded via `models/config.py` into `models/registry.py`). Active provider/model selection seems primarily handled by `core/provider_client.py` using `Config` values (set via `models/config.py`). An unused provider abstraction layer (`providers/base.py`, `providers/litellm_provider.py`, `providers/manager.py`) exists alongside the direct `litellm` integration in `ProviderClient`.
    *   **Redundancy/Usage:** The `ProviderClient` appears to bypass the `ProviderFactory` and `LLMProviderInterface` abstraction, calling `litellm` directly. This makes the `litellm_provider.py` implementation likely unused and the abstraction layer ineffective in the current flow. This needs architectural clarification/refactoring.
    *   **Provider Error Handling:** `ProviderClient` uses broad `try...except Exception` around `litellm` calls. It logs errors but returns user-facing error strings as content, rather than raising specific exceptions. This prevents crashes but limits the ability of the caller (`Orchestrator`) to implement nuanced error handling (e.g., retries, fallbacks based on error type).
    *   **Model/Provider Improvements Identified:**
        *   Resolve provider abstraction inconsistency (use `LLMProviderInterface` or remove it).
        *   Implement specific exception handling for `litellm` API calls.
        *   Clarify and unify model selection/optimization strategy (`ModelSelector` vs `ModelOptimizer` vs `EnhancedModelOptimizer`).
        *   Improve model metadata management (dynamic updates?).
        *   Refine model fallback logic.
        *   Consider context-aware model selection.
        *   Ensure secure and consistent API key handling.
*   **Evolution System:**
    *   **`state.py`:** Defines `EvolutionState` (data class) and `EvolutionStateManager` (manages state persistence to JSON, validates transitions based on manifest). Simple single-path state management.
    *   **`state_machine.py`:** Defines `EvolutionPath` and `EvolutionStateMachine`. Seems to enhance/replace `EvolutionStateManager` by supporting multiple named evolution paths loaded from the manifest. Manages the same state file but interprets transitions within the context of the current path.
    *   **Redundancy:** The coexistence of `EvolutionStateManager` and `EvolutionStateMachine` is confusing. `EvolutionStateMachine` appears more capable. Needs clarification/consolidation. Both load manifest repeatedly (caching needed) and use sync file I/O.
    *   **`manifest.json`:** Defines stages (name, description, module path, validation criteria) and allowed transitions between them. Currently defines stages 0-3 in a mostly linear path.
    *   **`loader.py`:** Defines `StageLoader`. Uses `EvolutionStateManager` (the simpler one) to get the current stage ID. Reads `manifest.json` to find the stage's module path. Dynamically imports the stage module and calls `apply_stage_modifications(assistant, tool_registry)` within it. Handles transitions and fallback to stage_0. Uses sync file I/O and loads manifest repeatedly.
    *   **`stages/stage_0.py`:** Implements the baseline stage (Stage 0). The `apply_stage_modifications` function correctly performs no actions.
    *   **Stage Implementation:** Only Stage 0 is implemented. Stages 1-3 defined in the manifest lack corresponding modules, indicating the evolution feature is incomplete beyond the baseline.
    *   **Overall Assessment:** The system aims to manage agent progression through defined stages using a manifest and dynamic loading. The design concept is sound but currently ineffective due to incomplete implementation (only Stage 0 exists) and architectural issues (redundant state managers, unclear transition triggers). Needs significant work to be functional.
*   **MCP System:**
    *   **`core/mcp.py` (`MCPManager`, `MCPClient`):** Handles MCP server config, discovery, stdio process lifecycle, consent, and communication (HTTP/stdio JSON-RPC). Provides service client facades (`MCPMemoryService`, etc.). Contains an `asyncio`-based `ProcessManager`.
    *   **`mcp/base.py`:** Defines abstract interfaces (`MCPServiceInterface`, `MCPServerInterface`) and registries (`MCPServiceRegistry`, `MCPServerRegistry`) for MCP components. **However, this abstraction layer appears unused.** The implementation in `core/mcp.py` does not inherit from or use these base classes/registries.
    *   **Integration:** `ToolExecutor` correctly identifies MCP tools (by name convention) and delegates calls to `MCPManager`, which uses `MCPClient` for communication, consent handling, and stdio server management. The flow seems correct, but error propagation could be more specific, and the duplicate MCP logic in `ToolRegistry` bypasses this flow.
*   **UI:**
    *   **`textual_chat.py`:** Implements a TUI using Textual. Interacts correctly with `Assistant.chat` for general messages. Includes autocomplete for memory commands.
    *   **`textual_chat.py` / Tool Flow Bypass:** **Major Issue:** Handles `/mem` commands by directly importing and calling tool handlers (e.g., `memory_list_handler`) instead of going through the `Assistant`'s tool execution flow. This breaks encapsulation and consistency. Memory handlers might also block the UI's event loop if they perform sync I/O.
*   **Utilities:**
    *   **`config.py`:** Provides robust `Config` class managing settings from INI, .env, and environment variables. Includes schema validation, type conversion, and secure path handling. Uses a singleton pattern via `get_config`.
    *   **`logging.py`:** Provides `setup_logging` (for root logger) and `get_logger` (for named loggers). Configurable via env vars. Uses standard `logging` module. Potential issue with handler duplication if both functions add handlers.
*   ...etc.
