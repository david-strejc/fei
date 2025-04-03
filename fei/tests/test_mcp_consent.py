import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from fei.core.mcp import MCPClient, MCPConsentDeniedError, MCPServerConfigError
from fei.utils.config import Config

# Sample Consent Handler for testing
async def mock_consent_handler(server_id: str, service: str, method: str, params: dict) -> bool:
    """Mock consent handler that can be controlled."""
    if hasattr(mock_consent_handler, 'approved'):
        return mock_consent_handler.approved
    return True # Default to approve if not set

@pytest.fixture
def mock_config():
    """Fixture for a mocked Config object."""
    config = MagicMock(spec=Config)
    # Default consent settings
    config.get_dict.return_value = {} # Default empty rules
    config.get_string.return_value = "ask" # Default policy 'ask'
    return config

@pytest.fixture
def mcp_client(mock_config):
    """Fixture for MCPClient with mocked config and consent handler."""
    # Reset mock handler state for each test
    if hasattr(mock_consent_handler, 'approved'):
        delattr(mock_consent_handler, 'approved')

    client = MCPClient(config=mock_config, consent_handler=mock_consent_handler)
    # Mock process manager and http client to avoid actual process/network calls
    client.process_manager = MagicMock()
    client.http_client = AsyncMock()

    # Correctly mock the async context manager _ensure_server_running
    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = {"type": "stdio", "url": "http://mock.server"} # What the context yields
    mock_cm.__aexit__.return_value = None # Return value of __aexit__ (can be None)
    # Patch the method on the instance to return the configured async context manager mock
    client._ensure_server_running = MagicMock(return_value=mock_cm)

    # Mock the actual service call methods
    client._call_stdio_service = AsyncMock(return_value={"result": "stdio_ok"})
    # Ensure the mock HTTP response has raise_for_status
    mock_http_response = MagicMock()
    mock_http_response.status_code = 200
    mock_http_response.json.return_value = {"jsonrpc": "2.0", "result": {"result": "http_ok"}, "id": "123"}
    mock_http_response.raise_for_status = MagicMock() # Mock raise_for_status
    client.http_client.post = AsyncMock(return_value=mock_http_response)

    return client

# --- Tests for _get_consent_policy ---

def test_get_consent_policy_default(mcp_client, mock_config):
    """Test default policy ('ask') when no rules match."""
    mock_config.get_dict.return_value = {} # No rules
    mock_config.get_string.return_value = "ask"
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodX")
    assert policy == "ask"

def test_get_consent_policy_specific_method_allow(mcp_client, mock_config):
    """Test specific server.service.method rule ('allow')."""
    rules = {"server1.serviceA.methodX": "allow"}
    mock_config.get_dict.return_value = rules
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodX")
    assert policy == "allow"

def test_get_consent_policy_specific_method_deny(mcp_client, mock_config):
    """Test specific server.service.method rule ('deny')."""
    rules = {"server1.serviceA.methodX": "deny"}
    mock_config.get_dict.return_value = rules
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodX")
    assert policy == "deny"

def test_get_consent_policy_service_wildcard(mcp_client, mock_config):
    """Test server.service.* rule."""
    rules = {"server1.serviceA.*": "allow"}
    mock_config.get_dict.return_value = rules
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodY")
    assert policy == "allow"

def test_get_consent_policy_server_wildcard(mcp_client, mock_config):
    """Test server.*.* rule."""
    rules = {"server1.*.*": "deny"}
    mock_config.get_dict.return_value = rules
    policy = mcp_client._get_consent_policy("server1", "serviceB", "methodZ")
    assert policy == "deny"

def test_get_consent_policy_global_service_method(mcp_client, mock_config):
    """Test *.service.method rule."""
    rules = {"*.serviceA.methodX": "allow"}
    mock_config.get_dict.return_value = rules
    policy = mcp_client._get_consent_policy("server2", "serviceA", "methodX")
    assert policy == "allow"

def test_get_consent_policy_global_service_wildcard(mcp_client, mock_config):
    """Test *.service.* rule."""
    rules = {"*.serviceB.*": "deny"}
    mock_config.get_dict.return_value = rules
    policy = mcp_client._get_consent_policy("server3", "serviceB", "methodW")
    assert policy == "deny"

def test_get_consent_policy_precedence(mcp_client, mock_config):
    """Test rule precedence (more specific wins)."""
    rules = {
        "server1.serviceA.methodX": "allow", # Most specific
        "server1.serviceA.*": "deny",
        "server1.*.*": "ask",
        "*.serviceA.methodX": "deny",
        "*.serviceA.*": "ask",
    }
    mock_config.get_dict.return_value = rules
    # Test specific method
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodX")
    assert policy == "allow"
    # Test other method in same service/server (should hit server1.serviceA.*)
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodY")
    assert policy == "deny"
    # Test other service in same server (should hit server1.*.*)
    policy = mcp_client._get_consent_policy("server1", "serviceB", "methodZ")
    assert policy == "ask"
    # Test specific method on different server (should hit *.serviceA.methodX)
    policy = mcp_client._get_consent_policy("server2", "serviceA", "methodX")
    assert policy == "deny"
    # Test other method in specific service on different server (should hit *.serviceA.*)
    policy = mcp_client._get_consent_policy("server2", "serviceA", "methodY")
    assert policy == "ask"
    # Test completely different server/service (should hit default)
    mock_config.get_string.return_value = "ask" # Ensure default is ask
    policy = mcp_client._get_consent_policy("server3", "serviceC", "methodW")
    assert policy == "ask"

def test_get_consent_policy_invalid_rule_value(mcp_client, mock_config):
    """Test fallback to default when rule value is invalid."""
    rules = {"server1.serviceA.methodX": "maybe"}
    mock_config.get_dict.return_value = rules
    mock_config.get_string.return_value = "ask" # Default policy
    policy = mcp_client._get_consent_policy("server1", "serviceA", "methodX")
    assert policy == "ask"

# --- Tests for call_service consent handling ---

@pytest.mark.asyncio
async def test_call_service_policy_allow(mcp_client, mock_config):
    """Test call_service proceeds when policy is 'allow'."""
    rules = {"server1.serviceA.methodX": "allow"}
    mock_config.get_dict.return_value = rules
    mock_config.get_string.return_value = "ask" # Default

    result = await mcp_client.call_service("serviceA", "methodX", server_id="server1")
    assert result == {"result": "stdio_ok"}
    mcp_client._call_stdio_service.assert_called_once()
    # Ensure consent handler was NOT called
    assert not hasattr(mock_consent_handler, 'called') or not mock_consent_handler.called

@pytest.mark.asyncio
async def test_call_service_policy_deny(mcp_client, mock_config):
    """Test call_service raises error when policy is 'deny'."""
    rules = {"server1.serviceA.methodX": "deny"}
    mock_config.get_dict.return_value = rules
    mock_config.get_string.return_value = "ask" # Default

    with pytest.raises(MCPConsentDeniedError, match="denied by configuration"):
        await mcp_client.call_service("serviceA", "methodX", server_id="server1")

    mcp_client._call_stdio_service.assert_not_called()
    # Ensure consent handler was NOT called
    assert not hasattr(mock_consent_handler, 'called') or not mock_consent_handler.called

@pytest.mark.asyncio
async def test_call_service_policy_ask_approved(mcp_client, mock_config):
    """Test call_service proceeds when policy is 'ask' and user approves."""
    mock_config.get_string.return_value = "ask" # Default policy

    # Mock consent handler to approve
    mock_consent_handler.approved = True
    mcp_client.consent_handler = AsyncMock(wraps=mock_consent_handler) # Wrap to track calls

    result = await mcp_client.call_service("serviceA", "methodX", server_id="server1", params={"p": 1})
    assert result == {"result": "stdio_ok"}
    mcp_client.consent_handler.assert_called_once_with("server1", "serviceA", "methodX", {"p": 1})
    mcp_client._call_stdio_service.assert_called_once()

@pytest.mark.asyncio
async def test_call_service_policy_ask_denied(mcp_client, mock_config):
    """Test call_service raises error when policy is 'ask' and user denies."""
    mock_config.get_string.return_value = "ask" # Default policy

    # Mock consent handler to deny
    mock_consent_handler.approved = False
    mcp_client.consent_handler = AsyncMock(wraps=mock_consent_handler) # Wrap to track calls

    with pytest.raises(MCPConsentDeniedError, match="User denied consent"):
        await mcp_client.call_service("serviceA", "methodX", server_id="server1", params={"p": 1})

    mcp_client.consent_handler.assert_called_once_with("server1", "serviceA", "methodX", {"p": 1})
    mcp_client._call_stdio_service.assert_not_called()

@pytest.mark.asyncio
async def test_call_service_policy_ask_no_handler(mcp_client, mock_config):
    """Test call_service raises error when policy is 'ask' but no handler is set."""
    mock_config.get_string.return_value = "ask" # Default policy
    mcp_client.consent_handler = None # Explicitly remove handler

    with pytest.raises(MCPServerConfigError, match="no consent handler is configured"):
        await mcp_client.call_service("serviceA", "methodX", server_id="server1")

    mcp_client._call_stdio_service.assert_not_called()

@pytest.mark.asyncio
async def test_call_service_policy_ask_handler_error(mcp_client, mock_config):
    """Test call_service raises error if consent handler itself raises an error."""
    mock_config.get_string.return_value = "ask" # Default policy

    # Mock consent handler to raise an error
    mcp_client.consent_handler = AsyncMock(side_effect=ValueError("Handler failed"))

    with pytest.raises(MCPConsentDeniedError, match="Error during consent handling"):
        await mcp_client.call_service("serviceA", "methodX", server_id="server1")

    mcp_client.consent_handler.assert_called_once()
    mcp_client._call_stdio_service.assert_not_called()
