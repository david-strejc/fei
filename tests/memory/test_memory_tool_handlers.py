# Integration tests for fei.tools.memory_tools handlers
import pytest
import subprocess
import time
import os
import signal
import uuid
import requests
import socket
import sys
from contextlib import closing
from pathlib import Path
from typing import Tuple, Generator
from dotenv import load_dotenv

# Load environment variables for potential API keys or configs used by handlers
load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / 'config' / '.env')

from fei.tools.memdir_connector import MemdirConnector
from fei.tools.memory_tools import (
    memory_create_handler,
    memory_search_handler,
    memory_delete_handler,
    memdir_server_start,
    memdir_server_stop,
    memdir_server_status,
)

# --- Constants ---
TEST_API_KEY = "test-handler-secret-key" # Use a distinct key for handler tests
SERVER_STARTUP_TIMEOUT = 15 # seconds, increased slightly just in case

# --- Helper Functions ---
def find_free_port():
   """Finds an available port on the local machine."""
   with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
       s.bind(('', 0))
       s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
       return s.getsockname()[1]

# --- Fixtures ---

@pytest.fixture(scope="session")
def memdir_data_dir(tmp_path_factory) -> Path:
   """Creates a temporary directory for Memdir data for the test session."""
   return tmp_path_factory.mktemp("memdir_handler_data")

@pytest.fixture(scope="session")
def memdir_server(memdir_data_dir: Path) -> Generator[Tuple[str, str, int, Path], None, None]:
   """
   Starts a Memdir server instance as a subprocess for the test session.

   Yields:
       Tuple[str, str, int, Path]: Base URL, API key, PID, and data directory path.
   """
   port = find_free_port()
   base_url = f"http://127.0.0.1:{port}"
   api_key = TEST_API_KEY
   data_dir = memdir_data_dir

   # Command to start the server
   python_executable = sys.executable
   # Adjust path relative to this test file's location
   server_script = Path(__file__).parent.parent.parent / "memdir_tools" / "server.py"

   command = [
       python_executable,
       str(server_script),
       "--port", str(port),
       "--data-dir", str(data_dir),
       "--api-key", api_key,
   ]

   print(f"\n[Fixture] Starting Memdir server: {' '.join(command)}")
   project_root = Path(__file__).parent.parent.parent
   env = os.environ.copy()
   # Ensure PYTHONPATH includes project root AND memdir_tools parent if needed
   env['PYTHONPATH'] = f"{str(project_root)}{os.pathsep}{str(project_root / 'memdir_tools')}{os.pathsep}{env.get('PYTHONPATH', '')}"

   # Use preexec_fn=os.setsid on Unix-like systems to create a process group
   # This helps ensure all child processes are terminated properly.
   preexec_fn = os.setsid if os.name != 'nt' else None

   process = subprocess.Popen(
       command,
       stdout=subprocess.PIPE,
       stderr=subprocess.PIPE,
       text=True,
       env=env,
       preexec_fn=preexec_fn # Set process group leader (Unix)
   )

   # --- Wait for server to be ready ---
   start_time = time.time()
   server_ready = False
   ping_url = f"{base_url}/ping"
   while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
       # Check if process terminated unexpectedly
       if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"[Fixture] Server process terminated prematurely (PID: {process.pid}). Exit code: {process.returncode}")
            print("--- Server STDOUT ---")
            print(stdout)
            print("--- Server STDERR ---")
            print(stderr)
            pytest.fail(f"Memdir server process terminated unexpectedly during startup.")
            break # Exit loop

       try:
           response = requests.get(ping_url, timeout=0.5)
           # Memdir server /ping returns 200 OK with "pong"
           if response.status_code == 200 and response.text == "pong":
                print(f"[Fixture] Memdir server responded on {base_url}")
                server_ready = True
                break
           else:
               print(f"[Fixture] Ping to {ping_url} returned status {response.status_code}, text: '{response.text}'. Retrying...")
       except requests.ConnectionError:
           # print(f"[Fixture] Connection error to {ping_url}. Retrying...") # Verbose
           pass # Server not up yet
       except requests.Timeout:
            print(f"[Fixture] Connection to {ping_url} timed out. Retrying...")
       time.sleep(0.2) # Wait before retrying


   if not server_ready:
       # Cleanup and raise error if server didn't start
       print(f"[Fixture] Server failed to become ready within {SERVER_STARTUP_TIMEOUT} seconds.")
       stdout, stderr = process.communicate()
       print("--- Server STDOUT ---")
       print(stdout)
       print("--- Server STDERR ---")
       print(stderr)
       # Terminate process group on Unix, terminate process on Windows
       try:
           if os.name != 'nt':
               os.killpg(os.getpgid(process.pid), signal.SIGTERM)
           else:
               process.terminate()
           process.wait(timeout=5)
       except Exception as e:
           print(f"[Fixture] Error during cleanup of failed server start: {e}")
           process.kill() # Force kill if terminate fails
           process.wait()
       pytest.fail(f"Memdir server failed to start within {SERVER_STARTUP_TIMEOUT} seconds.")

   # --- Yield URL, API Key, PID, Data Dir ---
   yield base_url, api_key, process.pid, data_dir

   # --- Teardown: Stop the server ---
   print(f"\n[Fixture] Stopping Memdir server (PID: {process.pid}, Group: {os.getpgid(process.pid) if os.name != 'nt' else 'N/A'})...")
   # Terminate process group on Unix, terminate process on Windows
   try:
       if process.poll() is None: # Only terminate if still running
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            stdout, stderr = process.communicate(timeout=5) # Wait for termination
            print("[Fixture] --- Server STDOUT (Shutdown) ---")
            print(stdout)
            print("[Fixture] --- Server STDERR (Shutdown) ---")
            print(stderr)
   except subprocess.TimeoutExpired:
       print(f"[Fixture] Server (PID: {process.pid}) did not terminate gracefully, killing.")
       if process.poll() is None:
            if os.name != 'nt':
                os.killpg(os.getpgid(process.pid), signal.SIGKILL) # Use SIGKILL
            else:
                process.kill()
            process.wait() # Wait for kill
   except ProcessLookupError:
        print(f"[Fixture] Server process (PID: {process.pid}) already terminated.")
   except Exception as e:
        print(f"[Fixture] Error during server stop: {e}")
        if process.poll() is None:
            process.kill() # Final attempt to kill
            process.wait()
   print("[Fixture] Memdir server stopped.")


@pytest.fixture
def memdir_connector(memdir_server: Tuple[str, str, int, Path]) -> MemdirConnector:
   """Provides a MemdirConnector instance configured for the test server."""
   base_url, api_key, _, _ = memdir_server
   # Initialize without auto-starting server, as the fixture handles it
   connector = MemdirConnector(
       server_url=base_url,
       api_key=api_key,
       auto_start=False
   )
   return connector

# --- Test Functions ---

# TODO: Add tests for handlers:
# - memory_create_handler
# - memory_search_handler
# - memory_delete_handler
# - memory_create_handler
# - memory_search_handler
# - memory_delete_handler
# - memdir_server_start
# - memdir_server_stop
# - memdir_server_status

def test_fixture_runs(memdir_server):
    """Basic test to ensure the server fixture starts and stops."""
    base_url, api_key, pid, data_dir = memdir_server
    assert isinstance(base_url, str) and base_url.startswith("http://")
    assert api_key == TEST_API_KEY
    assert isinstance(pid, int)
    assert isinstance(data_dir, Path) and data_dir.exists()
    # Check if server is responding
    try:
        response = requests.get(f"{base_url}/ping", timeout=1)
        assert response.status_code == 200
        assert response.text == "pong"
    except requests.RequestException as e:
        pytest.fail(f"Server ping failed after fixture setup: {e}")


def test_memory_create_handler(memdir_connector: MemdirConnector):
    """Test the memory_create_handler function."""
    test_content = f"Test content for create handler {uuid.uuid4()}"
    test_folder = "handler_test_folder"
    test_tags = ["handler", "test", "create"]

    # Call the handler function directly
    result_str = memory_create_handler(
        connector=memdir_connector,
        content=test_content,
        folder=test_folder,
        tags=test_tags
    )

    # Assert the expected string format for success
    assert isinstance(result_str, str)
    assert "Memory created successfully" in result_str
    assert f"Folder: {test_folder}" in result_str
    assert "Filename: " in result_str # Check for the presence of the filename key
    # We don't know the exact filename, but we know it should be reported.


def test_memory_search_handler(memdir_connector: MemdirConnector):
    """Test the memory_search_handler function."""
    unique_content = f"Searchable content {uuid.uuid4()}"
    search_folder = "search_handler_tests"
    search_tags = ["search", "handler"]

    # 1. Create a memory to search for
    create_result = memory_create_handler(
        connector=memdir_connector,
        content=unique_content,
        folder=search_folder,
        tags=search_tags
    )
    assert "Memory created successfully" in create_result
    # Extract filename (simple parsing, might need refinement if format changes)
    try:
        filename = create_result.split("Filename: ")[1].split("\n")[0]
    except IndexError:
        pytest.fail(f"Could not parse filename from create result: {create_result}")


    # 2. Call the search handler
    search_query = f'"{unique_content}"' # Search for the exact phrase
    result_str = memory_search_handler(
        connector=memdir_connector,
        query=search_query,
        folder=search_folder, # Search within the specific folder
        limit=5
    )

    # 3. Assert the expected string format for search results
    assert isinstance(result_str, str)
    assert f"Found 1 result(s) for query" in result_str # Expecting one result
    assert filename in result_str # The filename of the created memory should be in the results
    assert unique_content[:50] in result_str # A snippet of the content should be present
    assert f"Folder: {search_folder}" in result_str # Folder context should be there


def test_memory_delete_handler(memdir_connector: MemdirConnector):
    """Test the memory_delete_handler function."""
    delete_content = f"Content to be deleted {uuid.uuid4()}"
    delete_folder = "delete_handler_tests"
    delete_tags = ["delete", "handler"]

    # 1. Create a memory to delete
    create_result = memory_create_handler(
        connector=memdir_connector,
        content=delete_content,
        folder=delete_folder,
        tags=delete_tags
    )
    assert "Memory created successfully" in create_result
    try:
        # Extract filename (ensure consistent parsing logic)
        filename = create_result.split("Filename: ")[1].split("\n")[0].strip()
    except IndexError:
        pytest.fail(f"Could not parse filename from create result: {create_result}")

    # 2. Call the delete handler
    delete_result_str = memory_delete_handler(
        connector=memdir_connector,
        filename=filename,
        folder=delete_folder
    )

    # 3. Assert the expected string format for delete success
    assert isinstance(delete_result_str, str)
    assert "Memory deleted successfully" in delete_result_str
    assert f"Filename: {filename}" in delete_result_str
    assert f"Folder: {delete_folder}" in delete_result_str

    # 4. Verify the memory is actually gone by searching
    time.sleep(0.1) # Give a moment for potential filesystem changes
    search_query = f'"{delete_content}"'
    search_result_str = memory_search_handler(
        connector=memdir_connector,
        query=search_query,
        folder=delete_folder,
        limit=1
    )
    assert "Found 0 result(s)" in search_result_str


def test_memdir_server_status_handler_running(memdir_server: Tuple[str, str, int, Path]):
    """Test the memdir_server_status_handler when the server is running."""
    base_url, api_key, pid, data_dir = memdir_server

    # Call the status handler function
    # Note: The handler uses MemdirConnector.get_server_status(), a class method.
    # This method might not know about the *specific* instance run by the fixture
    # unless it reads from a shared state or environment variables.
    # Let's see what the default MemdirConnector() inside the handler finds.
    # We might need to adjust the handler or test setup if it doesn't pick up
    # the fixture's server details correctly.
    status_result = memdir_server_status_handler(args={}) # Pass empty args

    assert isinstance(status_result, dict)
    # Check the status reported by the handler against the fixture details
    assert status_result.get("status") == "running" # Expecting it to find the running server
    assert status_result.get("pid") == pid
    # The handler's connector might default to localhost:5001 unless configured.
    # Let's assert based on the fixture's known URL for now.
    # This might fail if the handler doesn't use the fixture's config.
    assert status_result.get("url") == base_url
    assert status_result.get("data_dir") == str(data_dir)
    assert status_result.get("api_key_set") is True # API key is set in the fixture


def test_memdir_server_stop_handler(memdir_server: Tuple[str, str, int, Path]):
    """Test the memdir_server_stop_handler function."""
    base_url, api_key, pid, data_dir = memdir_server

    # 1. Verify server is running initially using the status handler
    initial_status = memdir_server_status_handler(args={})
    assert initial_status.get("status") == "running"
    assert initial_status.get("pid") == pid

    # 2. Call the stop handler
    # This handler internally creates a MemdirConnector instance.
    # It needs to find the correct PID/process to stop. It might rely on
    # finding a .pid file in the default data directory, or reading config.
    # This could be fragile if the handler's default config doesn't match the fixture.
    stop_result = memdir_server_stop_handler(args={})

    # 3. Assert the handler reported success
    assert isinstance(stop_result, dict)
    # Check for possible success messages, e.g., "stopped" or "not_running" if already stopped
    assert stop_result.get("status") in ["stopped", "not_running"], \
        f"Stop handler returned unexpected status: {stop_result.get('status')}"
    assert "Server stopped successfully" in stop_result.get("message", "") or \
           "Server process not found" in stop_result.get("message", "") or \
           "Server not running" in stop_result.get("message", "")


    # 4. Verify server is stopped using the status handler again
    # Allow a moment for the process to terminate fully
    time.sleep(0.5)
    final_status = memdir_server_status_handler(args={})
    assert final_status.get("status") == "not_running"
    assert final_status.get("pid") is None

    # Note: The memdir_server fixture's teardown will run after this test.
    # It should ideally handle the case where the server is already stopped gracefully.
    # We added checks in the fixture teardown (process.poll()) to help with this.


def test_memdir_server_start_handler(memdir_handler_config):
    """Test the memdir_server_start_handler function in isolation."""
    port, data_dir, api_key, base_url, pid_file = memdir_handler_config

    # Store original env vars
    original_env = {}
    env_vars_to_set = {
        "MEMDIR_PORT": str(port),
        "MEMDIR_DATA_DIR": str(data_dir),
        "MEMDIR_API_KEY": api_key
    }
    for key, value in env_vars_to_set.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
        print(f"[Test Start Handler] Set env: {key}={value}")

    try:
        # 1. Verify server is NOT running initially using status handler
        # The status handler also reads env vars implicitly via MemdirConnector
        initial_status = memdir_server_status_handler(args={})
        print(f"[Test Start Handler] Initial status check result: {initial_status}")
        # It might report 'unknown' or 'not_running' depending on implementation
        assert initial_status.get("status") in ["not_running", "unknown"], \
            f"Expected server to be not running initially, but status was: {initial_status.get('status')}"
        assert initial_status.get("pid") is None

        # 2. Call the start handler
        print("[Test Start Handler] Calling start handler...")
        start_result = memdir_server_start_handler(args={})
        print(f"[Test Start Handler] Start handler result: {start_result}")

        # 3. Assert the handler reported success
        assert isinstance(start_result, dict)
        assert start_result.get("status") == "started", \
            f"Start handler failed. Status: {start_result.get('status')}, Message: {start_result.get('message')}"
        assert "Server started successfully" in start_result.get("message", "")
        assert start_result.get("pid") is not None
        started_pid = start_result.get("pid")

        # 4. Verify server IS running using status handler again
        # Allow time for server process to fully initialize
        time.sleep(1.0)
        print("[Test Start Handler] Calling status handler after start...")
        final_status = memdir_server_status_handler(args={})
        print(f"[Test Start Handler] Final status check result: {final_status}")
        assert final_status.get("status") == "running"
        assert final_status.get("pid") == started_pid
        assert final_status.get("port") == port
        assert final_status.get("url") == base_url
        assert final_status.get("data_dir") == str(data_dir)
        assert final_status.get("api_key_set") is True

        # 5. Check connectivity directly
        try:
            response = requests.get(f"{base_url}/ping", timeout=1)
            assert response.status_code == 200
            assert response.text == "pong"
            print("[Test Start Handler] Ping successful.")
        except requests.RequestException as e:
            pytest.fail(f"Server ping failed after start handler success: {e}")

    finally:
        # Restore original environment variables
        print("[Test Start Handler] Restoring environment variables...")
        for key, value in original_env.items():
            if value is None:
                if key in os.environ:
                    del os.environ[key]
                    print(f"[Test Start Handler] Unset env: {key}")
            else:
                os.environ[key] = value
                print(f"[Test Start Handler] Restored env: {key}={value}")
        # Teardown of the fixture will handle stopping the server