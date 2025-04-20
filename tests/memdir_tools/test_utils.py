# memdir_tools/tests/test_utils.py
import pytest
import time
import re # Added import
import socket
from pathlib import Path
import os
from datetime import datetime
import shutil # Added import

# Import the functions to be tested
from memdir_tools.utils import (
    generate_memory_filename,
    parse_memory_filename,
    save_memory,
    parse_memory_content,
    create_memory_content,
    list_memories,
    move_memory,
    update_memory_flags, # Now correctly uncommented
    FLAGS,
    STANDARD_FOLDERS
)
# from memdir_tools.folders import MEMDIR_SUBFOLDERS # Let's keep the placeholder for now
# Placeholder for MEMDIR_SUBFOLDERS if not imported yet
MEMDIR_SUBFOLDERS = ["new", "cur", "tmp", ".Trash"] # Keep placeholder for simplicity unless needed

# --- Fixtures ---

@pytest.fixture
def memdir_base(tmp_path: Path) -> str:
    """Provides a temporary directory path for Memdir tests."""
    # tmp_path provides a unique directory for each test function
    base_dir = tmp_path / "memdir_test"
    base_dir.mkdir()
    # No need to pre-create structure here, save_memory should handle it if needed
    return str(base_dir)

# --- Helper Functions ---
def _create_test_memory(base_dir: str, folder: str, status: str, content: str, headers: dict = None, flags: str = "") -> str:
    """Helper to create a memory file directly in a specific status folder."""
    if headers is None: headers = {}
    filename = generate_memory_filename(flags=flags)
    full_content = create_memory_content(headers, content)
    folder_path = Path(base_dir) / folder / status if folder else Path(base_dir) / status
    folder_path.mkdir(parents=True, exist_ok=True)
    file_path = folder_path / filename
    file_path.write_text(full_content, encoding='utf-8')
    # Manually set timestamp for predictable sorting if needed (optional)
    # os.utime(file_path, (parsed['timestamp'], parsed['timestamp']))
    return filename


# --- Tests for generate_memory_filename ---

def test_generate_memory_filename_no_flags():
    """Test generating a filename with no flags."""
    start_time = int(time.time())
    filename = generate_memory_filename()
    end_time = int(time.time())

    # Basic format check
    pattern = r"^(\d+)\.([a-f0-9]+)\.([^:]+):2,$"
    match = re.match(pattern, filename)
    assert match is not None, f"Filename '{filename}' does not match expected pattern."

    # Extract parts
    timestamp_str, unique_id, hostname, _ = match.groups() # Flags part is empty
    timestamp = int(timestamp_str)

    # Check timestamp is recent
    assert start_time <= timestamp <= end_time

    # Check unique_id format (8 hex chars)
    assert len(unique_id) == 8
    assert all(c in '0123456789abcdef' for c in unique_id)

    # Check hostname
    assert hostname == socket.gethostname()

    # Check flags are empty
    assert filename.endswith(":2,"), f"Filename '{filename}' should end with ':2,'"


def test_generate_memory_filename_with_flags():
    """Test generating a filename with valid flags."""
    flags_to_test = "SPF" # Valid flags in random order
    filename = generate_memory_filename(flags=flags_to_test)

    # Basic format check
    pattern = r"^(\d+)\.([a-f0-9]+)\.([^:]+):2,([A-Z]*)$"
    match = re.match(pattern, filename)
    assert match is not None, f"Filename '{filename}' does not match expected pattern."

    # Extract flags
    _, _, _, flags_part = match.groups()

    # Check flags are sorted and valid
    expected_flags = "".join(sorted(list(set(f for f in flags_to_test if f in FLAGS)))) # Should be "FPS"
    assert flags_part == expected_flags, f"Expected flags '{expected_flags}', but got '{flags_part}'"
    assert flags_part == "FPS" # Explicit check for sorting

def test_generate_memory_filename_with_invalid_and_duplicate_flags():
    """Test generating a filename ignores invalid and duplicate flags."""
    flags_to_test = "SFPXSYZ" # S, F, P are valid, X, Y, Z invalid, S duplicate
    filename = generate_memory_filename(flags=flags_to_test)

    # Basic format check
    pattern = r"^(\d+)\.([a-f0-9]+)\.([^:]+):2,([A-Z]*)$"
    match = re.match(pattern, filename)
    assert match is not None, f"Filename '{filename}' does not match expected pattern."

    # Extract flags
    _, _, _, flags_part = match.groups()

    # Check flags are sorted, unique, and only valid ones remain
    expected_flags = "FPS" # Sorted unique valid flags from "SFPXSYZ"
    assert flags_part == expected_flags, f"Expected flags '{expected_flags}', but got '{flags_part}'"

# --- Tests for parse_memory_filename ---

def test_parse_memory_filename_with_flags():
    """Test parsing a valid filename with flags."""
    timestamp = 1678886400 # Example timestamp
    unique_id = "abcdef01"
    hostname = "test-host"
    flags = "FS" # Flags in alphabetical order
    filename = f"{timestamp}.{unique_id}.{hostname}:2,{flags}"

    parsed = parse_memory_filename(filename)

    assert parsed["timestamp"] == timestamp
    assert parsed["unique_id"] == unique_id
    assert parsed["hostname"] == hostname
    assert parsed["flags"] == list(flags) # Should be ['F', 'S']
    assert isinstance(parsed["date"], datetime)
    assert parsed["date"] == datetime.fromtimestamp(timestamp)

def test_parse_memory_filename_no_flags():
    """Test parsing a valid filename with no flags."""
    timestamp = 1678886401
    unique_id = "12345678"
    hostname = "another-host"
    filename = f"{timestamp}.{unique_id}.{hostname}:2," # Note the trailing comma

    parsed = parse_memory_filename(filename)

    assert parsed["timestamp"] == timestamp
    assert parsed["unique_id"] == unique_id
    assert parsed["hostname"] == hostname
    assert parsed["flags"] == [] # Should be an empty list
    assert isinstance(parsed["date"], datetime)
    assert parsed["date"] == datetime.fromtimestamp(timestamp)

def test_parse_memory_filename_no_flags_no_comma():
    """Test parsing a valid filename with no flags and no trailing comma (optional flags)."""
    timestamp = 1678886402
    unique_id = "87654321"
    hostname = "final-host"
    filename = f"{timestamp}.{unique_id}.{hostname}:2" # No comma or flags

    parsed = parse_memory_filename(filename)

    assert parsed["timestamp"] == timestamp
    assert parsed["unique_id"] == unique_id
    assert parsed["hostname"] == hostname
    assert parsed["flags"] == [] # Should be an empty list
    assert isinstance(parsed["date"], datetime)
    assert parsed["date"] == datetime.fromtimestamp(timestamp)


def test_parse_memory_filename_invalid_format():
    """Test parsing an invalid filename raises ValueError."""
    invalid_filenames = [
        "invalid-filename",
        "12345.abcdef.host", # Missing :2,
        "12345.abcdef.host:2,FlagsWithLowercase",
        "12345.abcdef.host:1,", # Incorrect version
        "nodigits.abcdef.host:2,",
    ]
    for filename in invalid_filenames:
        with pytest.raises(ValueError, match=f"Invalid memory filename format: {re.escape(filename)}"):
             parse_memory_filename(filename)


# --- Tests for save_memory ---

def test_save_memory_basic(memdir_base: str):
    """Test saving a basic memory to the root folder."""
    content = "This is the basic memory content."
    folder = "" # Root folder

    # Action
    filename = save_memory(base_dir=memdir_base, folder=folder, content=content)

    # Assertions
    # 1. Check filename validity
    try:
        parsed_name = parse_memory_filename(filename)
        assert parsed_name["flags"] == []
    except ValueError:
        pytest.fail(f"save_memory returned an invalid filename: {filename}")

    # 2. Check file location
    expected_path = Path(memdir_base) / "new" / filename
    assert expected_path.exists(), f"Memory file not found at {expected_path}"
    assert expected_path.is_file()

    # 3. Check file content
    saved_content = expected_path.read_text(encoding='utf-8')
    headers, body = parse_memory_content(saved_content)

    assert body == content
    assert "Date" in headers # Default header
    assert "Subject" in headers # Default header
    assert headers["Subject"] == f"Memory {parsed_name['unique_id']}"

    # 4. Check tmp folder is empty (file moved)
    tmp_folder_path = Path(memdir_base) / "tmp"
    assert not any(tmp_folder_path.iterdir()) if tmp_folder_path.exists() else True


def test_save_memory_with_headers_and_flags(memdir_base: str):
    """Test saving a memory with custom headers and flags."""
    content = "Memory with custom headers and flags."
    folder = ""
    headers = {"Subject": "Custom Subject", "X-Custom-Header": "Value123"}
    flags = "SF" # Seen, Flagged

    # Action
    filename = save_memory(base_dir=memdir_base, folder=folder, content=content, headers=headers, flags=flags)

    # Assertions
    # 1. Check filename validity and flags
    try:
        parsed_name = parse_memory_filename(filename)
        assert sorted(parsed_name["flags"]) == sorted(list(flags)) # Check flags match
    except ValueError:
        pytest.fail(f"save_memory returned an invalid filename: {filename}")

    # 2. Check file location
    expected_path = Path(memdir_base) / "new" / filename
    assert expected_path.exists()

    # 3. Check file content
    saved_content = expected_path.read_text(encoding='utf-8')
    saved_headers, saved_body = parse_memory_content(saved_content)

    assert saved_body == content
    assert saved_headers["Subject"] == "Custom Subject" # Custom subject overrides default
    assert saved_headers["X-Custom-Header"] == "Value123"
    assert "Date" in saved_headers # Default Date should still be added

def test_save_memory_to_subfolder(memdir_base: str):
    """Test saving a memory to a subfolder."""
    content = "This memory goes into a subfolder."
    folder = ".Projects/Testing" # Subfolder path

    # Action
    filename = save_memory(base_dir=memdir_base, folder=folder, content=content)

    # Assertions
    # 1. Check filename validity
    try:
        parse_memory_filename(filename)
    except ValueError:
        pytest.fail(f"save_memory returned an invalid filename: {filename}")

    # 2. Check file location (including subfolder structure)
    expected_path = Path(memdir_base) / ".Projects" / "Testing" / "new" / filename
    assert expected_path.exists(), f"Memory file not found at {expected_path}"
    assert expected_path.is_file()

    # 3. Check parent directories were created
    assert (Path(memdir_base) / ".Projects" / "Testing" / "new").is_dir()
    assert (Path(memdir_base) / ".Projects" / "Testing" / "tmp").is_dir()
    # 'cur' is not explicitly created by save_memory, only 'new' and 'tmp'

    # 4. Check file content
    saved_content = expected_path.read_text(encoding='utf-8')
    _, saved_body = parse_memory_content(saved_content)
    assert saved_body == content

def test_save_memory_empty_content(memdir_base: str):
    """Test saving a memory with empty content."""
    content = ""
    folder = ""

    # Action
    filename = save_memory(base_dir=memdir_base, folder=folder, content=content)

    # Assertions
    expected_path = Path(memdir_base) / "new" / filename
    assert expected_path.exists()
    saved_content = expected_path.read_text(encoding='utf-8')
    _, saved_body = parse_memory_content(saved_content)
    assert saved_body == "" # Body should be empty


# --- Tests for list_memories ---

def test_list_memories_empty(memdir_base: str):
    """Test listing from an empty/non-existent folder/status."""
    assert list_memories(base_dir=memdir_base, folder="", status="new") == []
    assert list_memories(base_dir=memdir_base, folder=".NonExistent", status="cur") == []
    # Create the base 'new' folder but leave it empty
    (Path(memdir_base) / "new").mkdir(exist_ok=True)
    assert list_memories(base_dir=memdir_base, folder="", status="new") == []


def test_list_memories_basic(memdir_base: str):
    """Test listing memories from 'new' and 'cur' status folders."""
    # Create files directly for testing list_memories isolation
    filename_new1 = _create_test_memory(memdir_base, "", "new", "New memory 1", flags="F")
    time.sleep(0.01) # Ensure different timestamps
    filename_new2 = _create_test_memory(memdir_base, "", "new", "New memory 2")
    time.sleep(0.01)
    filename_cur1 = _create_test_memory(memdir_base, "", "cur", "Current memory 1", flags="S")
    time.sleep(0.01)
    filename_cur2 = _create_test_memory(memdir_base, "", "cur", "Current memory 2", flags="SR")
    _create_test_memory(memdir_base, "", "tmp", "Temporary memory") # Should not be listed by default

    # List 'new'
    new_memories = list_memories(base_dir=memdir_base, folder="", status="new")
    assert len(new_memories) == 2
    assert {m["filename"] for m in new_memories} == {filename_new1, filename_new2}
    # Check sorting (most recent first)
    assert new_memories[0]["filename"] == filename_new2
    assert new_memories[1]["filename"] == filename_new1

    # List 'cur'
    cur_memories = list_memories(base_dir=memdir_base, folder="", status="cur")
    assert len(cur_memories) == 2
    assert {m["filename"] for m in cur_memories} == {filename_cur1, filename_cur2}
    # Check sorting (most recent first)
    assert cur_memories[0]["filename"] == filename_cur2
    assert cur_memories[1]["filename"] == filename_cur1

    # Check basic structure of a listed item (from 'cur')
    mem_info = cur_memories[0]
    assert mem_info["filename"] == filename_cur2
    assert mem_info["folder"] == ""
    assert mem_info["status"] == "cur"
    assert "content" not in mem_info # Default is include_content=False
    assert "headers" in mem_info
    assert "metadata" in mem_info
    assert mem_info["metadata"]["flags"] == ['R', 'S'] # Should be parsed correctly


def test_list_memories_include_content(memdir_base: str):
    """Test listing memories with include_content=True."""
    content1 = "Content for memory 1"
    filename1 = _create_test_memory(memdir_base, "", "cur", content1, headers={"Subject": "Subj1"})

    # List with content
    memories = list_memories(base_dir=memdir_base, folder="", status="cur", include_content=True)
    assert len(memories) == 1
    mem_info = memories[0]

    assert mem_info["filename"] == filename1
    assert "content" in mem_info
    assert mem_info["content"] == content1
    assert mem_info["headers"]["Subject"] == "Subj1"


def test_list_memories_subfolder(memdir_base: str):
    """Test listing memories from a subfolder."""
    subfolder = ".Archive/OldStuff"
    filename_arc1 = _create_test_memory(memdir_base, subfolder, "cur", "Archived 1")
    time.sleep(0.01)
    filename_arc2 = _create_test_memory(memdir_base, subfolder, "cur", "Archived 2")
    _create_test_memory(memdir_base, "", "cur", "Root memory") # Should not be listed

    # List from subfolder
    memories = list_memories(base_dir=memdir_base, folder=subfolder, status="cur")
    assert len(memories) == 2
    assert {m["filename"] for m in memories} == {filename_arc1, filename_arc2}
    assert memories[0]["filename"] == filename_arc2 # Sorted
    assert memories[0]["folder"] == subfolder


# --- Tests for move_memory ---

def test_move_memory_new_to_cur(memdir_base: str):
    """Test moving a memory from new to cur in the same folder."""
    folder = ""
    content = "Move me from new to cur"
    filename = _create_test_memory(memdir_base, folder, "new", content)
    source_path = Path(memdir_base) / "new" / filename
    target_path = Path(memdir_base) / "cur" / filename

    assert source_path.exists()
    assert not target_path.exists()

    # Action
    result = move_memory(
        base_dir=memdir_base,
        filename=filename,
        source_folder=folder,
        target_folder=folder,
        source_status="new",
        target_status="cur"
    )

    # Assertions
    assert result is True
    assert not source_path.exists()
    assert target_path.exists()
    assert target_path.read_text(encoding='utf-8') == create_memory_content({}, content) # Check content preserved

def test_move_memory_cur_to_different_folder(memdir_base: str):
    """Test moving a memory from cur to a different folder's cur."""
    source_folder = ""
    target_folder = ".Archive"
    content = "Move me to archive"
    filename = _create_test_memory(memdir_base, source_folder, "cur", content)
    source_path = Path(memdir_base) / "cur" / filename
    target_path = Path(memdir_base) / target_folder / "cur" / filename

    assert source_path.exists()
    assert not (Path(memdir_base) / target_folder / "cur").exists() # Target dir shouldn't exist yet

    # Action
    result = move_memory(
        base_dir=memdir_base,
        filename=filename,
        source_folder=source_folder,
        target_folder=target_folder,
        source_status="cur",
        target_status="cur"
    )

    # Assertions
    assert result is True
    assert not source_path.exists()
    assert target_path.exists() # Target file exists
    assert (Path(memdir_base) / target_folder / "cur").is_dir() # Target dir created
    assert target_path.read_text(encoding='utf-8') == create_memory_content({}, content)

def test_move_memory_with_flag_update(memdir_base: str):
    """Test moving a memory and updating its flags."""
    folder = ""
    content = "Move me and add Seen flag"
    filename_orig = _create_test_memory(memdir_base, folder, "new", content, flags="F") # Start with Flagged
    source_path = Path(memdir_base) / "new" / filename_orig

    assert source_path.exists()

    # Action
    new_flags = "FS" # Add Seen flag
    result = move_memory(
        base_dir=memdir_base,
        filename=filename_orig,
        source_folder=folder,
        target_folder=folder,
        source_status="new",
        target_status="cur",
        new_flags=new_flags
    )

    # Assertions
    assert result is True
    assert not source_path.exists() # Original file gone

    # Find the new file (filename changed due to flags)
    cur_files = list((Path(memdir_base) / "cur").iterdir())
    assert len(cur_files) == 1
    target_path = cur_files[0]
    filename_new = target_path.name

    # Check new filename has correct flags
    try:
        parsed_new = parse_memory_filename(filename_new)
        assert sorted(parsed_new["flags"]) == sorted(list(new_flags))
        # Check other parts are preserved (timestamp might differ slightly, check unique_id/hostname)
        parsed_orig = parse_memory_filename(filename_orig)
        assert parsed_new["unique_id"] == parsed_orig["unique_id"]
        assert parsed_new["hostname"] == parsed_orig["hostname"]
    except ValueError:
        pytest.fail(f"Moved file has invalid filename: {filename_new}")

    assert target_path.read_text(encoding='utf-8') == create_memory_content({}, content)

def test_move_memory_non_existent_file(memdir_base: str):
    """Test moving a non-existent memory file."""
    result = move_memory(
        base_dir=memdir_base,
        filename="does.not.exist:2,",
        source_folder="",
        target_folder=".Trash",
        source_status="cur",
        target_status="cur"
    )
    assert result is False

# --- Tests for update_memory_flags ---

def test_update_memory_flags_add_flag(memdir_base: str):
    """Test adding a flag to an existing memory."""
    folder = ""
    status = "cur"
    content = "Add Seen flag"
    filename_orig = _create_test_memory(memdir_base, folder, status, content, flags="F") # Start with Flagged
    orig_path = Path(memdir_base) / status / filename_orig

    assert orig_path.exists()

    # Action
    new_flags = "FS" # Add Seen
    result = update_memory_flags(
        base_dir=memdir_base,
        filename=filename_orig,
        folder=folder,
        status=status,
        flags=new_flags
    )

    # Assertions
    assert result is True
    assert not orig_path.exists() # Original filename should be gone

    # Find the new file
    cur_files = list((Path(memdir_base) / status).iterdir())
    assert len(cur_files) == 1
    new_path = cur_files[0]
    filename_new = new_path.name

    # Check new filename has correct flags
    try:
        parsed_new = parse_memory_filename(filename_new)
        assert sorted(parsed_new["flags"]) == sorted(list(new_flags))
        # Check other parts are preserved
        parsed_orig = parse_memory_filename(filename_orig)
        assert parsed_new["unique_id"] == parsed_orig["unique_id"]
        assert parsed_new["hostname"] == parsed_orig["hostname"]
    except ValueError:
        pytest.fail(f"Renamed file has invalid filename: {filename_new}")

    # Check content is preserved
    assert new_path.read_text(encoding='utf-8') == create_memory_content({}, content)

def test_update_memory_flags_remove_flag(memdir_base: str):
    """Test removing a flag from an existing memory."""
    folder = ".Sub"
    status = "new"
    content = "Remove Flagged flag"
    filename_orig = _create_test_memory(memdir_base, folder, status, content, flags="FS") # Start with Flagged, Seen
    orig_path = Path(memdir_base) / folder / status / filename_orig

    assert orig_path.exists()

    # Action
    new_flags = "S" # Remove Flagged
    result = update_memory_flags(
        base_dir=memdir_base,
        filename=filename_orig,
        folder=folder,
        status=status,
        flags=new_flags
    )

    # Assertions
    assert result is True
    assert not orig_path.exists()

    # Find the new file
    status_files = list((Path(memdir_base) / folder / status).iterdir())
    assert len(status_files) == 1
    new_path = status_files[0]
    filename_new = new_path.name

    # Check new filename has correct flags
    try:
        parsed_new = parse_memory_filename(filename_new)
        assert sorted(parsed_new["flags"]) == sorted(list(new_flags))
    except ValueError:
        pytest.fail(f"Renamed file has invalid filename: {filename_new}")

def test_update_memory_flags_change_flags(memdir_base: str):
    """Test changing flags completely."""
    folder = ""
    status = "cur"
    content = "Change flags from FS to RP"
    filename_orig = _create_test_memory(memdir_base, folder, status, content, flags="FS")
    orig_path = Path(memdir_base) / status / filename_orig

    # Action
    new_flags = "RP" # Replied, Priority
    result = update_memory_flags(
        base_dir=memdir_base, filename=filename_orig, folder=folder, status=status, flags=new_flags
    )

    # Assertions
    assert result is True
    assert not orig_path.exists()
    status_files = list((Path(memdir_base) / status).iterdir())
    assert len(status_files) == 1
    filename_new = status_files[0].name
    try:
        parsed_new = parse_memory_filename(filename_new)
        assert sorted(parsed_new["flags"]) == sorted(list(new_flags)) # Should be PR
    except ValueError:
        pytest.fail(f"Renamed file has invalid filename: {filename_new}")

def test_update_memory_flags_invalid_and_duplicate(memdir_base: str):
    """Test update ignores invalid and duplicate flags."""
    folder = ""
    status = "cur"
    content = "Update with invalid flags"
    filename_orig = _create_test_memory(memdir_base, folder, status, content, flags="F")
    orig_path = Path(memdir_base) / status / filename_orig

    # Action
    new_flags = "SPXSY" # Should result in FPS
    result = update_memory_flags(
        base_dir=memdir_base, filename=filename_orig, folder=folder, status=status, flags=new_flags
    )

    # Assertions
    assert result is True
    assert not orig_path.exists()
    status_files = list((Path(memdir_base) / status).iterdir())
    assert len(status_files) == 1
    filename_new = status_files[0].name
    try:
        parsed_new = parse_memory_filename(filename_new)
        assert sorted(parsed_new["flags"]) == sorted(list("FPS")) # Check only valid, unique, sorted flags remain
    except ValueError:
        pytest.fail(f"Renamed file has invalid filename: {filename_new}")


def test_update_memory_flags_non_existent_file(memdir_base: str):
    """Test updating flags for a non-existent memory file."""
    result = update_memory_flags(
        base_dir=memdir_base,
        filename="does.not.exist:2,",
        folder="",
        status="cur",
        flags="S"
    )
    assert result is False


# Basic test structure (keeping it for now)
# def test_placeholder():
#     """ Placeholder test to ensure the file is runnable. """
#     assert True