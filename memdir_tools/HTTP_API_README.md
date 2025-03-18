# Memdir HTTP API

This document describes the HTTP API for the Memdir memory management system. The API allows remote access to Memdir functionality, enabling multiple FEI instances to share and communicate memories.

## Getting Started

1. Install dependencies:
   ```bash
   pip install flask werkzeug
   ```

2. Set a secure API key:
   ```bash
   export MEMDIR_API_KEY="your-secure-api-key"
   ```

3. Start the server:
   ```bash
   python -m memdir_tools.server
   ```

## API Authentication

All API requests (except `/health`) require authentication using an API key. The key should be provided in the `X-API-Key` header:

```
X-API-Key: your-secure-api-key
```

## Endpoints

### Health Check

**GET /health**

Check if the server is running.

*No authentication required.*

Response:
```json
{
  "status": "ok",
  "service": "memdir-api"
}
```

### List Memories

**GET /memories**

List memories in a folder.

Query Parameters:
- `folder` (optional): Folder to list (default: root folder)
- `status` (optional): Status folder (default: "cur")
- `with_content` (optional): Include memory content (default: "false")

Response:
```json
{
  "count": 10,
  "folder": "root",
  "status": "cur",
  "memories": [...]
}
```

### Create Memory

**POST /memories**

Create a new memory.

Request Body:
```json
{
  "content": "Memory content goes here...",
  "headers": {
    "Subject": "Memory subject",
    "Tags": "tag1,tag2",
    "Priority": "high"
  },
  "folder": ".Projects/Python",
  "flags": "F"
}
```

Response:
```json
{
  "success": true,
  "message": "Memory created successfully",
  "filename": "1741911915.fcf18e11.Debian12:2,F",
  "folder": ".Projects/Python"
}
```

### Get Memory

**GET /memories/:memory_id**

Retrieve a specific memory by ID or filename.

Query Parameters:
- `folder` (optional): Folder to search in (default: all folders)

Response: The memory object

### Update Memory

**PUT /memories/:memory_id**

Update a memory's flags or move it to another folder.

Request Body:
```json
{
  "source_folder": ".Projects",
  "target_folder": ".Archive",
  "source_status": "cur",
  "target_status": "cur",
  "flags": "FS"
}
```

Response:
```json
{
  "success": true,
  "message": "Memory moved successfully",
  "memory_id": "fcf18e11",
  "source": ".Projects/cur",
  "destination": ".Archive/cur"
}
```

### Delete Memory

**DELETE /memories/:memory_id**

Move a memory to the trash folder.

Query Parameters:
- `folder` (optional): Folder to search in (default: all folders)

Response:
```json
{
  "success": true,
  "message": "Memory moved to trash successfully",
  "memory_id": "fcf18e11"
}
```

### Search Memories

**GET /search**

Search memories using a query.

Query Parameters:
- `q`: Search query string
- `folder` (optional): Folder to search in (default: all folders)
- `status` (optional): Status folder to search in (default: all statuses)
- `format` (optional): Output format (default: "json")
- `limit` (optional): Maximum number of results
- `offset` (optional): Offset for pagination
- `with_content` (optional): Include memory content (default: "false")
- `debug` (optional): Show debug information (default: "false")

Response:
```json
{
  "count": 5,
  "query": "python",
  "results": [...]
}
```

### List Folders

**GET /folders**

List all folders in the Memdir structure.

Response:
```json
{
  "folders": ["", ".Projects", ".Archive", ".Trash"]
}
```

### Create Folder

**POST /folders**

Create a new folder.

Request Body:
```json
{
  "folder": ".Projects/NewProject"
}
```

Response:
```json
{
  "success": true,
  "message": "Folder created successfully: .Projects/NewProject"
}
```

### Delete Folder

**DELETE /folders/:folder_path**

Delete a folder.

Response:
```json
{
  "success": true,
  "message": "Folder deleted successfully: .Projects/NewProject"
}
```

### Rename Folder

**PUT /folders/:folder_path**

Rename a folder.

Request Body:
```json
{
  "new_name": ".Projects/RenamedProject"
}
```

Response:
```json
{
  "success": true,
  "message": "Folder renamed successfully from .Projects/NewProject to .Projects/RenamedProject"
}
```

### Get Folder Stats

**GET /folders/:folder_path/stats**

Get statistics for a specific folder.

Response:
```json
{
  "total_memories": 15,
  "cur": 10,
  "new": 5,
  "tmp": 0,
  "folder": ".Projects"
}
```

### Run Filters

**POST /filters/run**

Run all memory filters to organize memories.

Request Body:
```json
{
  "dry_run": false
}
```

Response:
```json
{
  "success": true,
  "message": "Filters executed successfully",
  "actions": ["Moved memory abc123 to .Projects/Python", "Flagged memory def456"]
}
```

## Using the API Client

A Python client for the API is provided in `examples/memdir_http_client.py`:

```bash
# List memories in the root folder
python memdir_http_client.py list

# Create a new memory
python memdir_http_client.py create -s "Memory subject" -t "tag1,tag2" -p "high"

# Search memories
python memdir_http_client.py search "python"
```

## Integration with FEI

FEI can use the Memdir HTTP API through the provided connector:

```python
from fei.tools.memdir_connector import MemdirConnector

# Initialize connector
memdir = MemdirConnector(
    server_url="http://localhost:5000",
    api_key="your-secure-api-key"
)

# Create a memory
result = memdir.create_memory(
    content="Important information to remember...",
    headers={
        "Subject": "Important Note",
        "Tags": "important,note",
        "Priority": "high"
    }
)

# Search memories
search_results = memdir.search("important", with_content=True)
```

A complete integration example is provided in `examples/fei_memdir_integration.py`.

## Security Considerations

1. Always use a strong, unique API key
2. Consider using HTTPS in production environments
3. Limit server exposure to trusted networks
4. Implement proper access control if multiple users access the system