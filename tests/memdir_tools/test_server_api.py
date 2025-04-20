# memdir_tools/tests/test_server_api.py
import pytest
import pytest
import os
import json
import urllib.parse # Needed for URL encoding search parameters
from flask import Flask
from werkzeug.utils import secure_filename

# Assuming server.py is structured to allow importing the app factory or app instance
# We might need to adjust this based on the actual structure of server.py
# Let's assume there's a create_app function for now
from memdir_tools.server import create_app
from memdir_tools.utils import MEMDIR_CONFIG_ENV_VAR, MEMDIR_API_KEY_ENV_VAR, MEMDIR_BASE_DIR_ENV_VAR

TEST_API_KEY = "test-super-secret-key"
TEST_DATA_DIR_NAME = "test_memdir_data"

@pytest.fixture(scope='function')
def test_data_dir(tmp_path):
    """Creates a temporary data directory for testing."""
    data_dir = tmp_path / TEST_DATA_DIR_NAME
    data_dir.mkdir()
    # Create necessary subdirectories if the server expects them
    (data_dir / "memories").mkdir()
    (data_dir / "memories" / "_trash").mkdir()
    (data_dir / "folders").mkdir()
    return data_dir

@pytest.fixture(scope='function')
def client(test_data_dir):
    """Configures the Flask app for testing."""
    # Set environment variables for the test session
    os.environ[MEMDIR_BASE_DIR_ENV_VAR] = str(test_data_dir)
    os.environ[MEMDIR_API_KEY_ENV_VAR] = TEST_API_KEY
    # Point config to a non-existent file, relying on env vars, or create a dummy config
    # For simplicity, let's rely on env vars set above
    if MEMDIR_CONFIG_ENV_VAR in os.environ:
        del os.environ[MEMDIR_CONFIG_ENV_VAR] # Ensure config file isn't used

    app = create_app()
    app.config['TESTING'] = True

    # Clean up environment variables after test function finishes
    yield app.test_client()

    del os.environ[MEMDIR_BASE_DIR_ENV_VAR]
    del os.environ[MEMDIR_API_KEY_ENV_VAR]


# --- Test Cases ---

def test_ping_unauthenticated(client):
    """Test accessing a potentially unprotected endpoint (like ping or root)."""
    # Assuming there's a root or ping endpoint that might not require auth
    # If not, we'll adapt this or remove it.
    response = client.get('/')
    # We don't know the exact behavior yet, could be 200, 404, or redirect.
    # Let's start by asserting it's not a server error.
    assert response.status_code != 500
    # If it's meant to be public, assert 200. If protected, this test might change.
    # For now, let's assume root is public or redirects, not a 401/403.

def test_access_protected_endpoint_no_auth(client):
    """Test accessing a protected endpoint without API key."""
    response = client.get('/memories') # Example protected endpoint
    assert response.status_code == 401 # Expect Unauthorized

def test_access_protected_endpoint_invalid_auth(client):
    """Test accessing a protected endpoint with an invalid API key."""
    headers = {'X-API-Key': 'invalid-key'}
    response = client.get('/memories', headers=headers) # Example protected endpoint
    assert response.status_code == 401 # Expect Unauthorized

def test_access_protected_endpoint_valid_auth(client):
    """Test accessing a protected endpoint with a valid API key."""
    headers = {'X-API-Key': TEST_API_KEY}
    response = client.get('/memories', headers=headers) # Example protected endpoint
    # Expect success, likely 200 OK, assuming the endpoint exists and works
    assert response.status_code == 200
    # Check if the response is JSON
    assert response.content_type == 'application/json'
    # Check if the response body is a list (expected for /memories)
    assert isinstance(response.get_json(), list)


def test_create_memory_success(client, test_data_dir):
    """Test successfully creating a new memory."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    memory_data = {
        "content": "This is a test memory.",
        "folder": "tests",
        "tags": ["pytest", "api"]
    }
    response = client.post('/memories', headers=headers, json=memory_data)

    assert response.status_code == 201 # Expect Created
    assert response.content_type == 'application/json'
    response_data = response.get_json()
    assert 'filename' in response_data
    assert 'message' in response_data
    assert response_data['message'] == 'Memory created successfully.'

    # Verify the file was created in the correct location (within the test_data_dir)
    # The filename might be generated, so we get it from the response
    filename = response_data.get('filename')
    assert filename is not None
    # Construct the expected path within the temporary directory structure
    expected_folder_path = test_data_dir / "memories" / memory_data["folder"]
    expected_file_path = expected_folder_path / filename

    assert expected_file_path.exists()
    assert expected_file_path.is_file()

    # Optionally, verify content (requires reading the file)
    # with open(expected_file_path, 'r') as f:
    #     # Need to know the exact format saved (e.g., includes metadata?)
    #     # For now, just check existence. Content check can be added later if needed.
    #     pass

def test_create_memory_missing_content(client):
    """Test creating a memory with missing 'content' field."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    memory_data = {
        # "content": "This is missing.",
        "folder": "tests",
        "tags": ["pytest", "api"]
    }
    response = client.post('/memories', headers=headers, json=memory_data)
    assert response.status_code == 400 # Expect Bad Request
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()


def test_get_memory_success(client, test_data_dir):
    """Test retrieving an existing memory successfully."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    memory_data = {
        "content": "Content to be retrieved.",
        "folder": "retrieval_test",
        "tags": ["get", "specific"]
    }
    # 1. Create a memory first
    create_response = client.post('/memories', headers=headers, json=memory_data)
    assert create_response.status_code == 201
    create_data = create_response.get_json()
    filename = create_data.get('filename')
    assert filename is not None

    # 2. Retrieve the created memory
    get_headers = {'X-API-Key': TEST_API_KEY}
    # Ensure the filename is URL-safe, though secure_filename usually handles this.
    # Flask test client handles URL encoding automatically.
    get_response = client.get(f'/memories/{filename}', headers=get_headers)

    assert get_response.status_code == 200
    assert get_response.content_type == 'application/json'
    retrieved_data = get_response.get_json()

    # 3. Verify the retrieved data
    assert retrieved_data['filename'] == filename
    assert retrieved_data['content'] == memory_data['content']
    assert retrieved_data['folder'] == memory_data['folder']
    # The server might return tags sorted or in a specific format
    assert sorted(retrieved_data['tags']) == sorted(memory_data['tags'])
    assert 'created_at' in retrieved_data
    assert 'modified_at' in retrieved_data
    # We could add more specific checks on timestamps if needed

def test_get_memory_not_found(client):
    """Test retrieving a non-existent memory."""
    headers = {'X-API-Key': TEST_API_KEY}
    response = client.get('/memories/this_file_does_not_exist.md', headers=headers)
    assert response.status_code == 404 # Expect Not Found
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()


@pytest.fixture(scope='function')
def setup_search_data(client):
    """Creates sample memories for search tests."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    memories_to_create = [
        {"content": "First memory about Python programming.", "folder": "programming", "tags": ["python", "code"]},
        {"content": "Second memory discussing Flask web framework.", "folder": "programming", "tags": ["python", "flask", "web"]},
        {"content": "A note about testing APIs.", "folder": "testing", "tags": ["api", "test"]},
        {"content": "Thoughts on project structure.", "folder": "design", "tags": ["architecture", "python"]},
    ]
    created_files = []
    for mem_data in memories_to_create:
        response = client.post('/memories', headers=headers, json=mem_data)
        assert response.status_code == 201
        created_files.append(response.get_json()['filename'])
    return created_files # Return filenames if needed, though not strictly necessary for search tests

# --- Search Tests ---

def test_search_unauthenticated(client):
    """Test searching without authentication."""
    response = client.get('/search?query=test')
    assert response.status_code == 401

def test_search_invalid_auth(client):
    """Test searching with invalid authentication."""
    headers = {'X-API-Key': 'wrong-key'}
    response = client.get('/search?query=test', headers=headers)
    assert response.status_code == 401

def test_search_by_content(client, setup_search_data):
    """Test searching memories by content query."""
    headers = {'X-API-Key': TEST_API_KEY}
    query = "Flask web"
    encoded_query = urllib.parse.quote_plus(query)
    response = client.get(f'/search?query={encoded_query}', headers=headers)

    assert response.status_code == 200
    assert response.content_type == 'application/json'
    results = response.get_json()
    assert isinstance(results, list)
    assert len(results) >= 1 # Expect at least the Flask memory
    # Check if the expected memory is in the results
    found = any("Flask web framework" in mem['content'] for mem in results)
    assert found

def test_search_by_folder(client, setup_search_data):
    """Test searching memories by folder."""
    headers = {'X-API-Key': TEST_API_KEY}
    folder = "programming"
    encoded_folder = urllib.parse.quote_plus(folder)
    response = client.get(f'/search?folder={encoded_folder}', headers=headers)

    assert response.status_code == 200
    results = response.get_json()
    assert isinstance(results, list)
    assert len(results) == 2 # Expecting the two memories in 'programming' folder
    assert all(mem['folder'] == folder for mem in results)

def test_search_by_tag(client, setup_search_data):
    """Test searching memories by tag."""
    headers = {'X-API-Key': TEST_API_KEY}
    tag = "python"
    encoded_tag = urllib.parse.quote_plus(tag)
    response = client.get(f'/search?tag={encoded_tag}', headers=headers)

    assert response.status_code == 200
    results = response.get_json()
    assert isinstance(results, list)
    assert len(results) == 3 # Expecting the three memories tagged 'python'
    assert all(tag in mem['tags'] for mem in results)

def test_search_by_folder_and_tag(client, setup_search_data):
    """Test searching memories by folder and tag."""
    headers = {'X-API-Key': TEST_API_KEY}
    folder = "programming"
    tag = "flask"
    encoded_folder = urllib.parse.quote_plus(folder)
    encoded_tag = urllib.parse.quote_plus(tag)
    response = client.get(f'/search?folder={encoded_folder}&tag={encoded_tag}', headers=headers)

    assert response.status_code == 200
    results = response.get_json()
    assert isinstance(results, list)
    assert len(results) == 1 # Expecting only the Flask memory
    assert results[0]['folder'] == folder
    assert tag in results[0]['tags']
    assert "Flask web framework" in results[0]['content']

def test_search_no_results(client, setup_search_data):
    """Test searching with criteria that yield no results."""
    headers = {'X-API-Key': TEST_API_KEY}
    query = "non_existent_term_xyz"
    encoded_query = urllib.parse.quote_plus(query)
    response = client.get(f'/search?query={encoded_query}', headers=headers)

    assert response.status_code == 200
    results = response.get_json()
    assert isinstance(results, list)
    assert len(results) == 0


# --- Deletion Tests ---

def test_delete_memory_unauthenticated(client):
    """Test deleting without authentication."""
    # We don't need to create a file, just try deleting a plausible name
    response = client.delete('/memories/some_file_to_delete.md')
    assert response.status_code == 401

def test_delete_memory_invalid_auth(client):
    """Test deleting with invalid authentication."""
    headers = {'X-API-Key': 'wrong-key'}
    response = client.delete('/memories/some_file_to_delete.md', headers=headers)
    assert response.status_code == 401

def test_delete_memory_success(client, test_data_dir):
    """Test successfully deleting (moving to trash) an existing memory."""
    # 1. Create a memory to delete
    create_headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    memory_data = {"content": "This memory will be deleted.", "folder": "to_delete"}
    create_response = client.post('/memories', headers=create_headers, json=memory_data)
    assert create_response.status_code == 201
    filename = create_response.get_json()['filename']
    original_folder_path = test_data_dir / "memories" / memory_data["folder"]
    original_file_path = original_folder_path / filename
    assert original_file_path.exists() # Verify it exists before delete

    # 2. Delete the memory
    delete_headers = {'X-API-Key': TEST_API_KEY}
    delete_response = client.delete(f'/memories/{filename}', headers=delete_headers)

    # 3. Verify response
    assert delete_response.status_code == 200 # Or 204 No Content, depending on implementation
    assert delete_response.content_type == 'application/json' # If status is 200
    delete_data = delete_response.get_json()
    assert 'message' in delete_data
    assert 'moved_to_trash' in delete_data['message'] # Check for specific message if possible
    assert filename in delete_data['message']

    # 4. Verify file is gone from original location
    assert not original_file_path.exists()

    # 5. Verify file exists in trash
    # The trash path needs to be constructed based on server logic.
    # Assuming it moves to <base_dir>/memories/_trash/<original_folder>/<filename>
    # Or maybe just <base_dir>/memories/_trash/<filename> ?
    # Let's assume the latter for now, adjust if tests fail based on server.py logic.
    trash_dir = test_data_dir / "memories" / "_trash"
    trashed_file_path = trash_dir / filename
    # It might also include the folder structure in trash, e.g., trash_dir / memory_data["folder"] / filename
    # Let's check the simpler case first.
    assert trashed_file_path.exists()

def test_delete_memory_not_found(client):
    """Test deleting a non-existent memory."""
    headers = {'X-API-Key': TEST_API_KEY}
    response = client.delete('/memories/non_existent_file_for_delete.md', headers=headers)
    assert response.status_code == 404 # Expect Not Found
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()


# --- Folder Listing Tests ---

def test_get_folders_unauthenticated(client):
    """Test listing folders without authentication."""
    response = client.get('/folders')
    assert response.status_code == 401

def test_get_folders_invalid_auth(client):
    """Test listing folders with invalid authentication."""
    headers = {'X-API-Key': 'wrong-key'}
    response = client.get('/folders', headers=headers)
    assert response.status_code == 401

def test_get_folders_success(client, test_data_dir):
    """Test successfully listing folders."""
    # 1. Ensure at least one folder exists by creating a memory in it
    folder_name = "folder_for_listing"
    create_headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    memory_data = {"content": "Memory to create a folder.", "folder": folder_name}
    create_response = client.post('/memories', headers=create_headers, json=memory_data)
    assert create_response.status_code == 201

    # Also create an empty folder directly for testing listing empty ones too
    empty_folder_name = "empty_folder_for_listing"
    (test_data_dir / "memories" / empty_folder_name).mkdir()


    # 2. List folders
    list_headers = {'X-API-Key': TEST_API_KEY}
    list_response = client.get('/folders', headers=list_headers)

    # 3. Verify response
    assert list_response.status_code == 200
    assert list_response.content_type == 'application/json'
    folders_data = list_response.get_json()
    assert isinstance(folders_data, list)

    # Check that the created folder is in the list
    # The API might return just names or objects with more details. Assuming just names for now.
    assert folder_name in folders_data
    assert empty_folder_name in folders_data
    # Ensure system folders like _trash are not listed (assuming this is desired)
    assert "_trash" not in folders_data


# --- Folder Creation Tests ---

def test_create_folder_unauthenticated(client):
    """Test creating a folder without authentication."""
    folder_data = {"foldername": "new_folder_unauth"}
    response = client.post('/folders', json=folder_data)
    assert response.status_code == 401

def test_create_folder_invalid_auth(client):
    """Test creating a folder with invalid authentication."""
    headers = {'X-API-Key': 'wrong-key', 'Content-Type': 'application/json'}
    folder_data = {"foldername": "new_folder_invalid_auth"}
    response = client.post('/folders', headers=headers, json=folder_data)
    assert response.status_code == 401

def test_create_folder_success(client, test_data_dir):
    """Test successfully creating a new folder."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    folder_name = "newly_created_folder"
    folder_data = {"foldername": folder_name}
    response = client.post('/folders', headers=headers, json=folder_data)

    assert response.status_code == 201 # Expect Created
    assert response.content_type == 'application/json'
    response_data = response.get_json()
    assert 'message' in response_data
    assert 'created successfully' in response_data['message']
    assert folder_name in response_data['message']

    # Verify the directory was created on the filesystem
    expected_folder_path = test_data_dir / "memories" / folder_name
    assert expected_folder_path.exists()
    assert expected_folder_path.is_dir()

def test_create_folder_already_exists(client, test_data_dir):
    """Test creating a folder that already exists."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    folder_name = "existing_folder"
    # Create it first directly or via API
    (test_data_dir / "memories" / folder_name).mkdir()

    folder_data = {"foldername": folder_name}
    response = client.post('/folders', headers=headers, json=folder_data)

    assert response.status_code == 409 # Expect Conflict
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()
    assert 'already exists' in response.get_json()['error']

def test_create_folder_missing_foldername(client):
    """Test creating a folder with missing 'foldername' field."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    folder_data = {} # Missing foldername
    response = client.post('/folders', headers=headers, json=folder_data)
    assert response.status_code == 400 # Expect Bad Request
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()

def test_create_folder_invalid_name(client):
    """Test creating a folder with an invalid name (e.g., contains path separators)."""
    headers = {'X-API-Key': TEST_API_KEY, 'Content-Type': 'application/json'}
    # Example invalid names - adjust based on server validation rules
    invalid_names = ["folder/with/slash", "../relative", "_trash", ""]
    for folder_name in invalid_names:
        folder_data = {"foldername": folder_name}
        response = client.post('/folders', headers=headers, json=folder_data)
        assert response.status_code == 400 # Expect Bad Request
        assert response.content_type == 'application/json'
        assert 'error' in response.get_json()
        assert 'Invalid folder name' in response.get_json().get('error', '') # Check specific error if possible


# --- Folder Deletion Tests ---

def test_delete_folder_unauthenticated(client):
    """Test deleting a folder without authentication."""
    response = client.delete('/folders/some_folder_to_delete')
    assert response.status_code == 401

def test_delete_folder_invalid_auth(client):
    """Test deleting a folder with invalid authentication."""
    headers = {'X-API-Key': 'wrong-key'}
    response = client.delete('/folders/some_folder_to_delete', headers=headers)
    assert response.status_code == 401

def test_delete_empty_folder_success(client, test_data_dir):
    """Test successfully deleting an empty folder."""
    headers = {'X-API-Key': TEST_API_KEY}
    folder_name = "empty_folder_to_delete"
    folder_path = test_data_dir / "memories" / folder_name
    folder_path.mkdir() # Create the empty folder
    assert folder_path.exists()

    # Delete the folder via API
    response = client.delete(f'/folders/{folder_name}', headers=headers)

    assert response.status_code == 200 # Or 204 No Content
    if response.status_code == 200:
        assert response.content_type == 'application/json'
        response_data = response.get_json()
        assert 'message' in response_data
        assert 'deleted successfully' in response_data['message']
        assert folder_name in response_data['message']

    # Verify the directory is gone from the filesystem
    assert not folder_path.exists()

def test_delete_non_empty_folder(client, test_data_dir):
    """Test attempting to delete a non-empty folder (should fail)."""
    headers = {'X-API-Key': TEST_API_KEY}
    folder_name = "non_empty_folder_to_delete"
    folder_path = test_data_dir / "memories" / folder_name
    folder_path.mkdir()
    # Create a dummy file inside
    (folder_path / "dummy_memory.md").touch()
    assert folder_path.exists()
    assert any(folder_path.iterdir()) # Verify it's not empty

    # Attempt to delete the folder via API
    response = client.delete(f'/folders/{folder_name}', headers=headers)

    # Expect failure, likely Conflict or Bad Request
    assert response.status_code in [400, 409]
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()
    assert 'not empty' in response.get_json()['error']

    # Verify the directory still exists
    assert folder_path.exists()

def test_delete_non_existent_folder(client):
    """Test deleting a folder that does not exist."""
    headers = {'X-API-Key': TEST_API_KEY}
    folder_name = "folder_that_never_existed"
    response = client.delete(f'/folders/{folder_name}', headers=headers)

    assert response.status_code == 404 # Expect Not Found
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()

def test_delete_protected_folder(client):
    """Test attempting to delete a protected folder like _trash."""
    headers = {'X-API-Key': TEST_API_KEY}
    folder_name = "_trash" # Or other potentially protected names
    response = client.delete(f'/folders/{folder_name}', headers=headers)

    # Expect failure, likely Bad Request or Forbidden
    assert response.status_code in [400, 403]
    assert response.content_type == 'application/json'
    assert 'error' in response.get_json()
    assert 'Cannot delete protected folder' in response.get_json().get('error', '')

# Final placeholder removed as all tests are added.