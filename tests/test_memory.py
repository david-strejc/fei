"""
Tests for the memory system
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json
import shutil
from datetime import datetime

from fei.memory.base import MemoryInterface, MemoryFactory
from fei.memory.manager import MemoryManager
from fei.memory.memdir import MemdirMemory
from fei.utils.config import Config


class TestMemorySystem(unittest.TestCase):
    """Test the memory system"""
    
    def test_memory_factory(self):
        """Test memory factory"""
        # Clear existing memory systems
        MemoryFactory._memory_systems = {}
        
        # Register a mock memory system
        class MockMemory(MemoryInterface):
            def setup(self, config=None):
                self.config = config or {}
            
            def search(self, query, limit=10, **kwargs):
                return []
            
            def create(self, content, metadata=None):
                return {"id": "mock-id"}
            
            def get(self, memory_id):
                return {"id": memory_id}
            
            def update(self, memory_id, content=None, metadata=None):
                return {"id": memory_id}
            
            def delete(self, memory_id):
                return True
            
            def list(self, limit=100, offset=0, **kwargs):
                return []
            
            def search_by_tag(self, tag, limit=10):
                return []
        
        MemoryFactory.register_memory_system("mock", MockMemory)
        
        # Check if memory system is registered
        self.assertIn("mock", MemoryFactory.list_memory_systems())
        
        # Create memory system instance
        memory = MemoryFactory.create_memory_system("mock")
        self.assertIsInstance(memory, MockMemory)
    
    def test_memory_manager(self):
        """Test memory manager"""
        # Clear existing memory systems
        MemoryFactory._memory_systems = {}
        
        # Register a mock memory system
        class MockMemory(MemoryInterface):
            def setup(self, config=None):
                self.config = config or {}
            
            def search(self, query, limit=10, **kwargs):
                return [{"id": "mock-id", "content": query}]
            
            def create(self, content, metadata=None):
                return {"id": "mock-id", "content": content}
            
            def get(self, memory_id):
                return {"id": memory_id, "content": "Mock content"}
            
            def update(self, memory_id, content=None, metadata=None):
                return {"id": memory_id, "content": content or "Updated content"}
            
            def delete(self, memory_id):
                return True
            
            def list(self, limit=100, offset=0, **kwargs):
                return [{"id": "mock-id", "content": "Mock content"}]
            
            def search_by_tag(self, tag, limit=10):
                return [{"id": "mock-id", "content": "Mock content", "tags": [tag]}]
        
        MemoryFactory.register_memory_system("mock", MockMemory)
        
        # Create config with mock memory system
        config = Config()
        config.set("memory_system", "mock")
        
        # Create memory manager
        manager = MemoryManager(config)
        
        # Test search
        results = manager.search("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "test query")
        
        # Test create
        memory = manager.create("Test content")
        self.assertEqual(memory["content"], "Test content")
        
        # Test get
        memory = manager.get("mock-id")
        self.assertEqual(memory["content"], "Mock content")
        
        # Test update
        memory = manager.update("mock-id", content="Updated test content")
        self.assertEqual(memory["content"], "Updated test content")
        
        # Test delete
        result = manager.delete("mock-id")
        self.assertTrue(result)
        
        # Test list
        memories = manager.list()
        self.assertEqual(len(memories), 1)
        
        # Test search by tag
        results = manager.search_by_tag("test-tag")
        self.assertEqual(len(results), 1)
        self.assertIn("test-tag", results[0]["tags"])
    
    @patch("requests.get")
    @patch("requests.post")
    @patch("requests.put")
    @patch("requests.delete")
    def test_memdir_memory(self, mock_delete, mock_put, mock_post, mock_get):
        """Test Memdir memory system"""
        # Set up mock responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "results": [{"id": "test-id", "content": "Test content"}],
            "memories": [{"id": "test-id", "content": "Test content"}]
        }
        
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"id": "test-id", "content": "Test content"}
        
        mock_put.return_value.status_code = 200
        mock_put.return_value.json.return_value = {"id": "test-id", "content": "Updated content"}
        
        mock_delete.return_value.status_code = 200
        mock_delete.return_value.json.return_value = {"status": "success"}
        
        # Create Memdir memory system
        memory = MemdirMemory()
        memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key"})
        
        # Test search
        results = memory.search("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "test-id")
        
        # Test create
        result = memory.create("Test content")
        self.assertEqual(result["id"], "test-id")
        
        # Test get
        result = memory.get("test-id")
        self.assertEqual(result["id"], "test-id")
        
        # Test update
        result = memory.update("test-id", content="Updated content")
        self.assertEqual(result["id"], "test-id")
        
        # Test delete
        result = memory.delete("test-id")
        self.assertTrue(result)
        
        # Test list
        results = memory.list()
        self.assertEqual(len(results), 1)
        
        # Test search by tag
        results = memory.search_by_tag("test-tag")
        self.assertEqual(len(results), 1)


class TestMemdirMemoryWithServer(unittest.TestCase):
    """Test Memdir memory system with a mock server"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a temporary directory for the server
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock memory file
        memory_data = {
            "id": "test-id",
            "content": "Test content",
            "headers": {
                "Subject": "Test memory",
                "Date": datetime.now().isoformat(),
                "Source": "Test"
            },
            "folder": "test",
            "flags": "test-tag"
        }
        
        # Create the folder structure
        os.makedirs(os.path.join(self.temp_dir, "test", "cur"), exist_ok=True)
        
        # Write the memory file
        with open(os.path.join(self.temp_dir, "test", "cur", "test-id"), "w") as f:
            json.dump(memory_data, f)
    
    def tearDown(self):
        """Clean up test environment"""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
    
    @patch("subprocess.Popen")
    @patch("requests.get")
    def test_server_management(self, mock_get, mock_popen):
        """Test server management"""
        # Set up mock responses
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {"status": "ok"}
        
        # Set up mock process
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Create Memdir memory system
        memory = MemdirMemory()
        memory.setup({"server_url": "http://localhost:5000", "api_key": "test-key", "auto_start": False})
        
        # Test start server
        result = memory.start_server(self.temp_dir)
        self.assertEqual(result["status"], "started")
        
        # Test get server status
        status = memory.get_server_status()
        self.assertTrue(status["running"])
        
        # Test stop server
        result = memory.stop_server()
        self.assertEqual(result["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
