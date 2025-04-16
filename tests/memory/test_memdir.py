"""
Tests for the Memdir memory system.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil
from datetime import datetime

from fei.memory.memdir import MemdirMemory
from fei.utils.config import Config


class TestMemdirMemory:
    """Test the Memdir memory system."""
    
    def setup_method(self):
        """Set up the test."""
        self.memory = MemdirMemory()
    
    def test_init(self):
        """Test initialization."""
        # Check if memory is initialized correctly
        assert self.memory.server_url is None
        assert self.memory.api_key is None
        assert self.memory.auto_start is False
    
    def test_setup(self):
        """Test setup."""
        # Set up memory
        config = {
            "server_url": "http://localhost:5000",
            "api_key": "test-key",
            "auto_start": False
        }
        self.memory.setup(config)
        
        # Check if memory is set up correctly
        assert self.memory.server_url == "http://localhost:5000"
        assert self.memory.api_key == "test-key"
        assert self.memory.auto_start is False
    
    def test_setup_with_env(self):
        """Test setup with environment variables."""
        # Mock environment variables
        with patch.dict(os.environ, {
            "MEMDIR_SERVER_URL": "http://localhost:5001",
            "MEMDIR_API_KEY": "env-key"
        }):
            # Set up memory without config
            self.memory.setup()
            
            # Check if memory is set up correctly
            assert self.memory.server_url == "http://localhost:5001"
            assert self.memory.api_key == "env-key"
    
    @patch("requests.request")
    def test_search(self, mock_request):
        """Test search."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"id": "test-id", "content": "Test content"}]}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Search
        results = self.memory.search("test query")
        
        # Check if request was made with correct arguments
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert args[1] == "http://localhost:5000/search"
        assert kwargs["params"]["q"] == "test query"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        
        # Check if results are correct
        assert results == [{"id": "test-id", "content": "Test content"}]
    
    @patch("requests.request")
    def test_create(self, mock_request):
        """Test create."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "test-id", "content": "Test content"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Create memory
        result = self.memory.create("Test content", {"Subject": "Test memory"})
        
        # Check if request was made with correct arguments
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "POST"
        assert args[1] == "http://localhost:5000/memories"
        assert kwargs["json"]["content"] == "Test content"
        assert kwargs["json"]["headers"]["Subject"] == "Test memory"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        
        # Check if result is correct
        assert result == {"id": "test-id", "content": "Test content"}
    
    @patch("requests.request")
    def test_get(self, mock_request):
        """Test get."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "test-id", "content": "Test content"}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Get memory
        result = self.memory.get("test-id")
        
        # Check if request was made with correct arguments
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert args[1] == "http://localhost:5000/memories/test-id"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        
        # Check if result is correct
        assert result == {"id": "test-id", "content": "Test content"}
    
    @patch("requests.request")
    def test_update(self, mock_request):
        """Test update."""
        # Set up mock responses for get and update
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {
            "id": "test-id",
            "content": "Original content",
            "metadata": {"folder": "test", "status": "cur", "flags": ""},
            "headers": {"Subject": "Test memory"}
        }
        mock_get_response.raise_for_status = MagicMock()
        
        mock_update_response = MagicMock()
        mock_update_response.json.return_value = {"id": "test-id", "content": "Updated content"}
        mock_update_response.raise_for_status = MagicMock()
        
        mock_request.side_effect = [mock_get_response, mock_update_response]
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Update memory
        result = self.memory.update("test-id", content="Updated content")
        
        # Check if requests were made with correct arguments
        assert mock_request.call_count == 2
        
        # First call should be GET to get current memory
        args1, kwargs1 = mock_request.call_args_list[0]
        assert args1[0] == "GET"
        assert args1[1] == "http://localhost:5000/memories/test-id"
        
        # Second call should be POST to create new memory with updated content
        args2, kwargs2 = mock_request.call_args_list[1]
        assert args2[0] == "POST"
        assert args2[1] == "http://localhost:5000/memories"
        assert kwargs2["json"]["content"] == "Updated content"
        
        # Check if result is correct
        assert result == {"id": "test-id", "content": "Updated content"}
    
    @patch("requests.request")
    def test_delete(self, mock_request):
        """Test delete."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Delete memory
        result = self.memory.delete("test-id")
        
        # Check if request was made with correct arguments
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "DELETE"
        assert args[1] == "http://localhost:5000/memories/test-id"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        
        # Check if result is correct
        assert result is True
    
    @patch("requests.request")
    def test_list(self, mock_request):
        """Test list."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"memories": [{"id": "test-id", "content": "Test content"}]}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # List memories
        results = self.memory.list()
        
        # Check if request was made with correct arguments
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert args[1] == "http://localhost:5000/memories"
        assert kwargs["params"]["folder"] == ""
        assert kwargs["params"]["status"] == "cur"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        
        # Check if results are correct
        assert results == [{"id": "test-id", "content": "Test content"}]
    
    @patch("requests.request")
    def test_search_by_tag(self, mock_request):
        """Test search by tag."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"results": [{"id": "test-id", "content": "Test content"}]}
        mock_response.raise_for_status = MagicMock()
        mock_request.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Search by tag
        results = self.memory.search_by_tag("test-tag")
        
        # Check if request was made with correct arguments
        mock_request.assert_called_once()
        args, kwargs = mock_request.call_args
        assert args[0] == "GET"
        assert args[1] == "http://localhost:5000/search"
        assert kwargs["params"]["q"] == "tag:test-tag"
        assert kwargs["headers"]["X-API-Key"] == "test-key"
        
        # Check if results are correct
        assert results == [{"id": "test-id", "content": "Test content"}]
    
    @patch("requests.get")
    @patch("subprocess.Popen")
    def test_start_server(self, mock_popen, mock_get):
        """Test start server."""
        # Set up mock response for health check
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Set up mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Mock socket to simulate port not in use
        with patch("socket.socket") as mock_socket:
            mock_socket_instance = MagicMock()
            mock_socket_instance.connect_ex.return_value = 1  # Port not in use
            mock_socket.return_value.__enter__.return_value = mock_socket_instance
            
            # Start server
            result = self.memory.start_server("/tmp/memdir")
            
            # Check if subprocess.Popen was called
            mock_popen.assert_called_once()
            
            # Check if result is correct
            assert result["status"] == "started"
    
    @patch("requests.get")
    def test_check_connection(self, mock_get):
        """Test check connection."""
        # Set up mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Check connection
        result = self.memory.check_connection()
        
        # Check if request was made with correct arguments
        mock_get.assert_called_once_with("http://localhost:5000/health", timeout=1.0)
        
        # Check if result is correct
        assert result is True
    
    def test_get_server_status(self):
        """Test get server status."""
        # Set up memory
        self.memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Set class variables
        MemdirMemory._server_process = MagicMock()
        MemdirMemory._server_process.poll.return_value = None
        MemdirMemory._server_process.pid = 12345
        MemdirMemory._managed_data_dir = "/tmp/memdir"
        MemdirMemory._port = 5000
        
        # Get server status
        status = self.memory.get_server_status()
        
        # Check if status is correct
        assert status["running"] is True
        assert status["pid"] == 12345
        assert status["port"] == 5000
        assert status["url"] == "http://localhost:5000"
        assert status["data_dir"] == "/tmp/memdir"
