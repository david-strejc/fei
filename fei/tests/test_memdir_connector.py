# fei/tests/test_memdir_connector.py
# Integration tests for MemdirConnector

import pytest
import subprocess
import time
import os # Keep os import if needed elsewhere, or remove if uuid is the only reason
import uuid # Added import
import requests # Added import
import socket
import sys
from contextlib import closing
from pathlib import Path
from typing import Tuple, Generator

from fei.tools.memdir_connector import MemdirConnector

# --- Constants ---
TEST_API_KEY = "test-secret-key"
SERVER_STARTUP_TIMEOUT = 10 # seconds

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
   return tmp_path_factory.mktemp("memdir_data")

@pytest.fixture(scope="session")
def memdir_server(memdir_data_dir: Path, request) -> Generator[Tuple[str, str], None, None]:
   """
   Starts a Memdir server instance as a subprocess for the test session.

   Yields:
       Tuple[str, str]: The base URL of the server and the API key.
   """
   port = find_free_port()
   base_url = f"http://127.0.0.1:{port}"
   api_key = TEST_API_KEY
   data_dir = str(memdir_data_dir)

   # Command to start the server
   # Ensure we use the same Python executable that's running pytest
   python_executable = sys.executable
   server_script = Path(__file__).parent.parent.parent / "memdir_tools" / "server.py"

   command = [
       python_executable,
       str(server_script),
       "--port", str(port),
       "--data-dir", data_dir,
       "--api-key", api_key,
   ]

   print(f"\nStarting Memdir server: {' '.join(command)}")
   # Use Popen for non-blocking execution
   # Set PYTHONPATH to include the project root directory
   project_root = Path(__file__).parent.parent.parent
   env = os.environ.copy()
   env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')

   process = subprocess.Popen(
       command,
       stdout=subprocess.PIPE,
       stderr=subprocess.PIPE,
       text=True,
       env=env # Pass the modified environment
   )

   # --- Wait for server to be ready ---
   start_time = time.time()
   server_ready = False
   while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
       try:
           # Use a simple endpoint like /ping or /status if available, otherwise root
           response = requests.get(f"{base_url}/ping", timeout=0.5)
           # Check for a specific success status code if /ping is implemented
           # If /ping isn't implemented, check if *any* response comes back
           if response.status_code == 200 or response.status_code == 404: # Adjust based on actual /ping behavior
                print(f"Memdir server responded on {base_url}")
                server_ready = True
                break
       except requests.ConnectionError:
           time.sleep(0.2) # Wait before retrying
       except requests.Timeout:
            print(f"Memdir server connection timed out on {base_url}, retrying...")
            time.sleep(0.2)


   if not server_ready:
       # Cleanup and raise error if server didn't start
       stdout, stderr = process.communicate()
       print("--- Server STDOUT ---")
       print(stdout)
       print("--- Server STDERR ---")
       print(stderr)
       process.terminate()
       process.wait()
       pytest.fail(f"Memdir server failed to start within {SERVER_STARTUP_TIMEOUT} seconds.")

   # --- Yield URL and API Key ---
   yield base_url, api_key

   # --- Teardown: Stop the server ---
   print(f"\nStopping Memdir server (PID: {process.pid})...")
   process.terminate()
   try:
       stdout, stderr = process.communicate(timeout=5) # Wait for termination
       print("--- Server STDOUT (Shutdown) ---")
       print(stdout)
       print("--- Server STDERR (Shutdown) ---")
       print(stderr)
   except subprocess.TimeoutExpired:
       print(f"Server (PID: {process.pid}) did not terminate gracefully, killing.")
       process.kill()
       process.wait() # Wait for kill
   print("Memdir server stopped.")


@pytest.fixture
def memdir_connector(memdir_server: Tuple[str, str]) -> MemdirConnector:
   """Provides a MemdirConnector instance configured for the test server."""
   base_url, api_key = memdir_server
   # Initialize without auto-starting server, as the fixture handles it
   connector = MemdirConnector(
       server_url=base_url, # Corrected argument name
       api_key=api_key,
       auto_start=False # Corrected argument name and value logic
   )
   return connector


# --- Initial Failing Test ---

def test_connector_initialization(memdir_connector: MemdirConnector, memdir_server: Tuple[str, str]):
   """Verify the connector is initialized with the correct URL and API key."""
   base_url, api_key = memdir_server
   assert memdir_connector.server_url == base_url # Corrected attribute name
   assert memdir_connector.api_key == api_key
   assert memdir_connector.auto_start is False # Corrected attribute name
   # Add a basic check to see if the server is reachable via the connector
   # This assumes MemdirConnector might have a status check method or similar
   # If not, we'll add a test for a specific API call next.
   # For now, just check initialization.

# --- API Method Tests ---

def test_create_memory(memdir_connector: MemdirConnector):
    """Test creating a memory via the connector and live server."""
    test_text = "This is the content of the test memory."
    test_headers = {"source": "pytest", "test_id": str(uuid.uuid4())} # Renamed variable

    # Call the method under test
    response = memdir_connector.create_memory(content=test_text, headers=test_headers) # Corrected argument names

    # Assertions for a successful creation based on actual server response
    assert isinstance(response, dict)
    assert response.get("success") is True, f"Expected success=True, got response: {response}"
    assert "filename" in response, f"Expected 'filename' in response: {response}"
    filename = response["filename"]
    assert isinstance(filename, str)
    assert len(filename) > 0 # Basic check that filename is not empty
    # We could parse the filename format here if needed, but let's keep it simple for now.
    # Example: assert re.match(r"\d+\.[a-f0-9]+\.[^:]+:2,", filename)

    # Check other expected fields (optional but good)
    assert response.get("message") == "Memory created successfully"
    assert response.get("folder") == "root" # Assuming default folder "" maps to "root" in response

    # We could add a subsequent 'get_memory' call here to verify,
    # but let's keep tests focused for now. We'll test 'get_memory' separately.


def test_get_memory(memdir_connector: MemdirConnector):
    """Test retrieving a specific memory via the connector."""
    test_text = f"Content for get_memory test - {uuid.uuid4()}"
    test_headers = {"Subject": "Get Test", "X-Test-ID": str(uuid.uuid4())}

    # 1. Create a memory to retrieve
    create_response = memdir_connector.create_memory(
        content=test_text,
        headers=test_headers,
        folder="" # Create in root
    )
    assert create_response.get("success") is True
    filename = create_response.get("filename")
    assert filename is not None

    # 2. Attempt to retrieve the memory using the connector
    # Pass filename as the memory_id positional argument
    retrieved_memory = memdir_connector.get_memory(filename, folder="") # Corrected call signature

    # 3. Assertions
    assert isinstance(retrieved_memory, dict)
    # Check structure based on expected server response for GET /memories/<id>
    # (Assuming it returns content and headers similar to list_memories)
    assert retrieved_memory.get("filename") == filename
    assert retrieved_memory.get("folder") == "" # Check folder
    assert retrieved_memory.get("content") == test_text # Check content
    assert isinstance(retrieved_memory.get("headers"), dict)
    # Check if original headers are present (server might add others like Date)
    for key, value in test_headers.items():
        assert retrieved_memory["headers"].get(key) == value


def test_search_memory(memdir_connector: MemdirConnector):
    """Test searching for memories via the connector."""
    unique_tag = f"search-tag-{uuid.uuid4()}"
    test_text1 = f"First memory for search test {unique_tag}"
    test_text2 = f"Second memory for search test {unique_tag}"
    test_headers1 = {"Subject": "Search Test 1"}
    test_headers2 = {"Subject": "Search Test 2"}

    # 1. Create memories
    memdir_connector.create_memory(content=test_text1, headers=test_headers1)
    memdir_connector.create_memory(content=test_text2, headers=test_headers2)

    # Allow some time for indexing if the server does it asynchronously
    # (Adjust if necessary, though likely not needed for simple file storage)
    # time.sleep(0.1)

    # 2. Search for one of the memories using a unique part of its content
    search_term = "Second memory"
    # Assuming search takes a query string and request content explicitly
    search_results = memdir_connector.search(query=search_term, with_content=True)

    # 3. Assertions
    assert isinstance(search_results, list), f"Expected list, got {type(search_results)}"
    assert len(search_results) == 1, f"Expected 1 result for '{search_term}', got {len(search_results)}"

    result = search_results[0]
    assert isinstance(result, dict)
    # Check structure based on expected server response for GET /search
    # (Assuming it returns similar info to list_memories, maybe without full content by default)
    assert "filename" in result
    assert "folder" in result
    assert "headers" in result
    assert isinstance(result["headers"], dict)
    assert result["headers"].get("Subject") == "Search Test 2"
    # Check if content preview or full content is present depending on server default
    assert search_term in result.get("content", "") or search_term in result.get("content_preview", "")



def test_delete_memory(memdir_connector: MemdirConnector):
    """Test deleting a memory via the connector."""
    test_text = f"Content for delete_memory test - {uuid.uuid4()}"
    test_headers = {"Subject": "Delete Test"}

    # 1. Create a memory
    create_response = memdir_connector.create_memory(content=test_text, headers=test_headers)
    assert create_response.get("success") is True
    filename = create_response.get("filename")
    assert filename is not None

    # 2. Verify it exists initially
    try:
        memdir_connector.get_memory(filename)
    except Exception as e:
        pytest.fail(f"Failed to get newly created memory before delete: {e}")

    # 3. Delete the memory
    # Assuming delete_memory takes filename or memory_id
    delete_response = memdir_connector.delete_memory(filename)
    assert isinstance(delete_response, dict)
    assert delete_response.get("success") is True

    # 4. Verify it's gone
    with pytest.raises(Exception) as excinfo:
        memdir_connector.get_memory(filename)

    # Check if the exception message indicates "not found" or similar
    # The exact exception type and message depend on how _make_request handles 404
    assert "not found" in str(excinfo.value).lower() or "404" in str(excinfo.value)


    assert "not found" in str(excinfo.value).lower() or "404" in str(excinfo.value)


def test_folder_operations(memdir_connector: MemdirConnector):
    """Test listing, creating, and deleting folders via the connector."""
    # 1. List initial folders
    initial_folders = memdir_connector.list_folders()
    assert isinstance(initial_folders, list)
    # Check for expected default folders (server might create these)
    # Note: The server implementation might vary slightly on defaults.
    # Check for folders known to be created by default by ensure_memdir_structure
    assert ".Trash" in initial_folders # Check if Trash is created by default
    # assert "" not in initial_folders # Verify root isn't explicitly listed

    # 2. Create a new folder
    new_folder_name = f"test-folder-{uuid.uuid4()}"
    create_response = memdir_connector.create_folder(new_folder_name)
    assert isinstance(create_response, dict)
    assert create_response.get("success") is True

    # 3. Verify new folder exists in list
    folders_after_create = memdir_connector.list_folders()
    # Check for the folder name with the leading dot added by the server/manager
    assert f".{new_folder_name}" in folders_after_create

    # 4. Delete the folder
    # Delete the folder using the name with the leading dot
    delete_response = memdir_connector.delete_folder(f".{new_folder_name}")
    assert isinstance(delete_response, dict)
    assert delete_response.get("success") is True

    # 5. Verify folder is gone
    folders_after_delete = memdir_connector.list_folders()
    assert new_folder_name not in folders_after_delete

import signal # Needed for _stop_server check on Unix

# --- Server Management Tests ---

@pytest.fixture
def managed_connector_details(tmp_path):
    """Provides details for a connector managing its own server."""
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    api_key = f"managed-key-{uuid.uuid4()}"
    # Create a unique data dir for this managed instance
    data_dir = tmp_path / f"managed_memdir_data_{port}"
    data_dir.mkdir()
    return {
        "port": port,
        "server_url": base_url,
        "api_key": api_key,
        "data_dir": str(data_dir)
    }

@pytest.fixture
def managed_connector(managed_connector_details):
    """Provides a connector configured to manage its own server process."""
    details = managed_connector_details
    # Initialize with auto_start=False so we can test _start_server explicitly.
    connector = MemdirConnector(
        server_url=details["server_url"],
        api_key=details["api_key"],
        auto_start=False
    )
    # HACK: Directly set the port and data_dir the connector *should* use internally.
    # This highlights that MemdirConnector needs refactoring for better testability.
    connector._port = details["port"]
    # Store data_dir for tests to pass to _start_server if needed
    connector._test_data_dir = details["data_dir"]

    yield connector

    # Teardown: ensure server started by the connector is stopped
    if hasattr(connector, '_server_process') and connector._server_process:
        print(f"\n[Teardown managed_connector] Stopping server process {connector._server_process.pid}")
        try:
            # Use the connector's own stop method if available and reliable
            connector._stop_server()
        except Exception as e:
            print(f"[Teardown managed_connector] Error during _stop_server: {e}. Attempting direct kill.")
            try:
                if connector._server_process.poll() is None: # Check if still running
                     if os.name == 'nt':
                         connector._server_process.terminate()
                     else:
                         os.killpg(os.getpgid(connector._server_process.pid), signal.SIGTERM)
                     connector._server_process.wait(timeout=2)
            except Exception as kill_e:
                 print(f"[Teardown managed_connector] Error during direct kill: {kill_e}")
        finally:
             connector._server_process = None # Ensure it's cleared


def test_get_server_status_not_running(managed_connector: MemdirConnector):
    """Test _get_server_status when server is not running."""
    connector = managed_connector
    # Assuming _get_server_status exists and returns a dict
    status = connector.get_server_status() # Corrected method name
    assert isinstance(status, dict)
    assert status.get("running") is False
    assert status.get("pid") is None
    assert status.get("port") == connector._port # Should know configured port
    assert status.get("url") == connector.server_url
    assert status.get("data_dir") is None # Doesn't know data_dir until started


def test_start_server_and_status(managed_connector: MemdirConnector):
    """Test starting the server via the connector and checking status."""
    connector = managed_connector
    assert connector._server_process is None # Should not be running initially

    # 1. Start the server
    # Pass the specific data_dir for this test
    started = connector._start_server(data_dir=connector._test_data_dir)
    assert started is True, "Connector._start_server() failed to return True"
    assert hasattr(connector, '_server_process'), "Connector lacks _server_process attribute after start"
    assert connector._server_process is not None, "Connector._server_process is None after start"
    # Give the process a moment to potentially crash
    time.sleep(0.2)
    assert connector._server_process.poll() is None, f"Server process exited unexpectedly (PID: {connector._server_process.pid})"

    # 2. Check status (assuming _get_server_status exists)
    status = connector.get_server_status() # Corrected method name
    assert status.get("running") is True
    assert status.get("pid") == connector._server_process.pid
    assert status.get("port") == connector._port
    assert status.get("url") == connector.server_url
    # Assuming _get_server_status can retrieve the data_dir used by the running process
    assert status.get("data_dir") == connector._test_data_dir

    # 3. Try starting again (should likely return False or do nothing gracefully)
    started_again = connector._start_server(data_dir=connector._test_data_dir)
    assert started_again is False, "Calling _start_server again should return False"
    assert status.get("pid") == connector._server_process.pid # PID should not change


def test_stop_server(managed_connector: MemdirConnector):
    """Test stopping the server via the connector."""
    connector = managed_connector

    # 1. Start the server first
    started = connector._start_server(data_dir=connector._test_data_dir)
    assert started is True
    assert connector._server_process is not None
    pid = connector._server_process.pid

    # 2. Stop the server
    connector._stop_server()
    assert connector._server_process is None # Check internal handle is cleared

    # 3. Verify process is actually stopped (allow time)
    time.sleep(0.5)
    try:
        # On Unix, sending signal 0 checks if process exists without killing
        if os.name != 'nt':
            os.kill(pid, 0)
        else:
            # On Windows, check if process is still active (less direct)
            # This might require psutil or similar for a robust check.
            # For now, we rely on _server_process being None and check status.
            pass
        # If kill(pid, 0) didn't raise an error, the process still exists
        process_exists_after_stop = True
    except OSError:
        process_exists_after_stop = False # Process does not exist (good!)
    except Exception as e:
         pytest.fail(f"Error checking process existence after stop: {e}")

    assert not process_exists_after_stop, f"Server process (PID: {pid}) still exists after _stop_server()"

    # 4. Check status after stopping
    status_after_stop = connector.get_server_status() # Corrected method name
    assert status_after_stop.get("running") is False
    assert status_after_stop.get("pid") is None


def test_stop_server_when_not_running(managed_connector: MemdirConnector):
    """Test stopping the server when it's not running."""
    connector = managed_connector
    assert connector._server_process is None # Ensure not running

    # Call stop (should not raise error)
    connector._stop_server()

    assert connector._server_process is None # Still None