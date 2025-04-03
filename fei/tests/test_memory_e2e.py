import pytest
import asyncio
from pathlib import Path
import shutil

from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
from fei.tools.code import create_code_tools
from fei.tools.memory_tools import create_memory_tools, MemdirConnector
from fei.utils.config import get_config, reset_config

# Define a temporary directory for Memdir server data
MEMDIR_TEST_DIR = Path("/tmp/fei_test_memdir_e2e")
MEMDIR_PORT = 8766 # Use a different port for testing

# --- Test Fixtures ---

@pytest.fixture(scope="module")
def tool_registry_with_memory():
    """Creates a ToolRegistry with code and memory tools."""
    registry = ToolRegistry()
    create_code_tools(registry)
    # Configure memory tools to use the test Memdir server
    create_memory_tools(registry)
    # Override MemdirConnector settings for testing
    MemdirConnector.DEFAULT_SERVER_URL = f"http://localhost:{MEMDIR_PORT}"
    MemdirConnector.DEFAULT_DATA_DIR = str(MEMDIR_TEST_DIR)
    # MemdirConnector defaults are set globally here for the module
    MemdirConnector.DEFAULT_SERVER_URL = f"http://localhost:{MEMDIR_PORT}"
    MemdirConnector.DEFAULT_DATA_DIR = str(MEMDIR_TEST_DIR)
    MemdirConnector.DEFAULT_PORT = MEMDIR_PORT
    return registry

@pytest.fixture(scope="module", autouse=True)
async def setup_memdir_server(tool_registry_with_memory):
    """Starts and stops the Memdir server for the test module."""
    # API Key should be set via environment variable BEFORE running pytest

    # Ensure Memdir test directory is clean
    if MEMDIR_TEST_DIR.exists():
        shutil.rmtree(MEMDIR_TEST_DIR)
    MEMDIR_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # --- Configure API Key ---
    test_api_key = "test-key-for-e2e-memdir-789" # Use a unique key for the test
    config = get_config()
    original_key = config.get("memdir.api_key") # Store original key if exists
    config.set("memdir.api_key", test_api_key)
    # Ensure the environment variable doesn't interfere or matches
    # os.environ["MEMDIR_API_KEY"] = test_api_key # Setting config should be sufficient
    # --- End Configure API Key ---

    # Start the server using the tool (it should now pick up the configured key)
    start_result = tool_registry_with_memory.execute_tool("memdir_server_start_handler", {})
    print(f"Memdir start result: {start_result}")
    assert start_result.get("success"), "Failed to start Memdir server for testing"

    # Wait a moment for the server to be fully ready
    await asyncio.sleep(2)

    # Check connection
    connector = MemdirConnector()
    is_connected = connector.check_connection()
    print(f"Memdir connection check: {is_connected}")
    assert is_connected, "Memdir server did not start or is not connectable"

    yield # Run tests

    # Stop the server using the tool
    stop_result = tool_registry_with_memory.execute_tool("memdir_server_stop_handler", {})
    print(f"Memdir stop result: {stop_result}")
    # Clean up directory
    # shutil.rmtree(MEMDIR_TEST_DIR) # Keep data for inspection if needed

    # --- Restore Environment Variable (if it was set before test) ---
    # No longer needed here as we assume it's set externally for the test run
    # --- End Restore Environment Variable ---


# --- Test Function ---

@pytest.mark.asyncio
async def test_memory_creation_and_search(tool_registry_with_memory):
    """
    Test Fei's ability to create and search memories using Memdir tools.
    """
    # Instantiate the assistant with tools
    try:
        # Use Google provider as it's configured with API key
        assistant = Assistant(
            provider="google",
            model="gemini/gemini-1.5-pro-latest", # Use the standard model for main interaction
            tool_registry=tool_registry_with_memory
        )
        # Inject the memory system prompt (or ensure it's part of the default system prompt)
        memory_system_prompt = """
You are Fei, an AI assistant capable of complex tasks and self-evolution. You have access to a long-term memory system (Memdir) via tools.

**Memory Creation:**
When you learn something important, solve a problem, complete a significant part of a task, or encounter a useful pattern/error resolution, use the `memory_create_handler` tool to save this information.
- Provide a concise, descriptive `subject` (like a title).
- Include detailed `content` explaining the information or context.
- Add relevant `tags` (comma-separated, e.g., `#learning, #python, #error_fix, #evolution_stage_1`). Use `#core` for immutable, foundational knowledge (like your core purpose, evolution stage definitions, critical safety rules).
- Specify the `folder` (e.g., `.Knowledge`, `.History`, `.Evolution/Checkpoints`). Use `.Core` for immutable memories tagged `#core`.

**Memory Retrieval:**
When you need information to perform a task, recall past experiences, or understand context, use the `memory_search_handler` tool.
- Provide a specific `query` describing what you need.
- Optionally specify `folder`, `tags`, or `limit`.

**Core Memories:**
Memories tagged `#core` and stored in the `.Core` folder are immutable and represent foundational knowledge. Do not attempt to modify or delete them. Refer to them when necessary for understanding your purpose, evolution state, or safety guidelines.
"""
    except ValueError as e:
        pytest.fail(f"Failed to initialize Assistant, likely missing API key: {e}")

    # --- Step 1: Create Memories ---
    prompt_create = """
    Please store the following two pieces of information in memory:
    1. Subject: 'Test Knowledge Memory', Content: 'This is a test memory stored in the knowledge base.', Tags: '#test, #knowledge', Folder: '.Knowledge'
    2. Subject: 'Core Agent Purpose', Content: 'My core purpose is to assist users with code-related tasks and evolve my capabilities safely.', Tags: '#core, #purpose', Folder: '.Core'
    Use the memory_create_handler tool for each. Confirm when done.
    """
    print("\n--- Sending Create Prompt ---")
    response_create = await assistant.chat(prompt_create, system_prompt=memory_system_prompt)
    print(f"\n--- Create Response ---\n{response_create}\n-----------------------")

    # Basic check on response
    assert "stored" in response_create.lower() or "created" in response_create.lower()

    # Verify memories were created (optional, requires tool call inspection or direct Memdir check)
    # For simplicity, we rely on the search step for verification.

    # --- Step 2: Search Memories ---
    # Reset conversation slightly to avoid confusion, or just continue
    # assistant.reset_conversation()

    prompt_search = """
    Now, please search for memories related to 'test knowledge' and also search for memories tagged '#core'. Use the memory_search_handler tool.
    """
    print("\n--- Sending Search Prompt ---")
    response_search = await assistant.chat(prompt_search, system_prompt=memory_system_prompt)
    print(f"\n--- Search Response ---\n{response_search}\n-----------------------")

    # Assertions based on the LLM's response summarizing the search results
    assert "Test Knowledge Memory" in response_search
    assert "Core Agent Purpose" in response_search
    assert "test knowledge" in response_search.lower() # Check if it mentions the first search
    assert "core memory" in response_search.lower() or "core purpose" in response_search.lower() # Check if it mentions the second search

    # More robust check: Inspect the conversation history for tool calls and results
    # This requires accessing assistant.conversation and parsing tool messages
    create_calls = 0
    search_calls = 0
    found_knowledge_result = False
    found_core_result = False

    for msg in assistant.conversation:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function", {}).get("name") == "memory_create_handler":
                    create_calls += 1
                if tc.get("function", {}).get("name") == "memory_search_handler":
                    search_calls += 1
        # Check tool results (assuming they are added as 'tool' role messages)
        if msg.get("role") == "tool" and msg.get("name") == "memory_search_handler":
             try:
                 content_data = json.loads(msg.get("content", "{}"))
                 if isinstance(content_data, dict) and "results" in content_data:
                     for result in content_data["results"]:
                         if "Test Knowledge Memory" in result.get("subject", ""):
                             found_knowledge_result = True
                         if "Core Agent Purpose" in result.get("subject", ""):
                             found_core_result = True
             except json.JSONDecodeError:
                 pass # Ignore errors parsing content for this check

    assert create_calls >= 2, "Expected at least 2 calls to memory_create_handler"
    assert search_calls >= 1, "Expected at least 1 call to memory_search_handler" # Could be 1 or 2 calls
    assert found_knowledge_result, "Search results in conversation history did not contain the knowledge memory"
    assert found_core_result, "Search results in conversation history did not contain the core memory"
