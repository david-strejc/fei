"""
Shared test fixtures and configuration.
"""

import os
import sys
import pytest
import tempfile
import json
from unittest.mock import MagicMock
from typing import Dict, Any, List

# Add project root to path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fei.utils.config import Config
from fei.providers.base import LLMProviderInterface
from fei.memory.base import MemoryInterface
from fei.mcp.base import MCPServiceInterface, MCPServerInterface


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def temp_config_file(temp_dir):
    """Create a temporary config file."""
    config_path = os.path.join(temp_dir, "config.json")
    config_data = {
        "provider": "mock",
        "model": "mock-model",
        "memory_system": "mock",
        "evolution_state_file": os.path.join(temp_dir, "evolution_state.json"),
        "evolution_manifest": os.path.join(temp_dir, "manifest.json")
    }
    
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    
    return config_path


@pytest.fixture
def config(temp_config_file):
    """Create a Config instance with a temporary config file."""
    return Config(config_file=temp_config_file)


@pytest.fixture
def mock_provider():
    """Create a mock LLM provider."""
    provider = MagicMock(spec=LLMProviderInterface)
    provider.get_default_model.return_value = "mock-model"
    provider.complete.return_value = {
        "choices": [
            {
                "message": {
                    "content": "Mock response",
                    "role": "assistant"
                }
            }
        ]
    }
    provider.extract_response_content.return_value = "Mock response"
    provider.extract_tool_calls.return_value = []
    
    return provider


@pytest.fixture
def mock_memory():
    """Create a mock memory system."""
    memory = MagicMock(spec=MemoryInterface)
    memory.search.return_value = [{"id": "mock-id", "content": "Mock content"}]
    memory.create.return_value = {"id": "mock-id", "content": "Mock content"}
    memory.get.return_value = {"id": "mock-id", "content": "Mock content"}
    memory.update.return_value = {"id": "mock-id", "content": "Updated content"}
    memory.delete.return_value = True
    memory.list.return_value = [{"id": "mock-id", "content": "Mock content"}]
    
    return memory


@pytest.fixture
def mock_mcp_service():
    """Create a mock MCP service."""
    service = MagicMock(spec=MCPServiceInterface)
    service.get_methods.return_value = [
        {
            "name": "test_method",
            "description": "Test method",
            "parameters": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        }
    ]
    
    async def call_method(method, params):
        return {"result": f"Called {method} with {params}"}
    
    service.call_method = MagicMock(side_effect=call_method)
    
    return service


@pytest.fixture
def mock_mcp_server():
    """Create a mock MCP server."""
    server = MagicMock(spec=MCPServerInterface)
    server.is_running.return_value = True
    server.get_services.return_value = ["mock_service"]
    
    async def start():
        return True
    
    async def stop():
        return True
    
    server.start = MagicMock(side_effect=start)
    server.stop = MagicMock(side_effect=stop)
    
    return server


@pytest.fixture
def sample_evolution_state():
    """Create a sample evolution state."""
    return {
        "stage_id": "stage_0",
        "transitions": [],
        "metadata": {"created_at": "2023-01-01T00:00:00"}
    }


@pytest.fixture
def sample_evolution_manifest():
    """Create a sample evolution manifest."""
    return {
        "stages": {
            "stage_0": {
                "name": "Initial Bootstrap",
                "description": "Initial stage",
                "module": "fei.evolution.stages.stage_0",
                "allowed_transitions": ["stage_1"]
            },
            "stage_1": {
                "name": "Basic Self-Improvement",
                "description": "First evolution stage",
                "module": "fei.evolution.stages.stage_1",
                "allowed_transitions": ["stage_0", "stage_2"]
            }
        }
    }
