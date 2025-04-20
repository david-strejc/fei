from typing import Any, Dict, Optional, List

from fei.utils.logging import get_logger
from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
from fei.evolution.loader import StageLoader
from fei.evolution.state import EvolutionStateManager, EvolutionState

logger = get_logger(__name__)


def load_stage(assistant: Assistant, tool_registry: ToolRegistry) -> str:
    """
    Loads the current evolution stage.

    Args:
        assistant: The Assistant instance.
        tool_registry: The ToolRegistry instance.

    Returns:
        The ID of the loaded stage.

    Raises:
        Exception: If the stage cannot be loaded.
    """
    loader = StageLoader()
    return loader.load_current_stage(assistant, tool_registry)


def transition_to_stage(stage_id: str, assistant: Assistant, tool_registry: ToolRegistry,
                       metadata: Optional[Dict[str, Any]] = None) -> bool:
    """
    Transition to a new stage.

    Args:
        stage_id: Target stage ID
        assistant: Assistant instance
        tool_registry: Tool registry instance
        metadata: Optional transition metadata

    Returns:
        True if transition was successful, False otherwise
    """
    loader = StageLoader()
    return loader.transition_to_stage(stage_id, assistant, tool_registry, metadata)


def get_current_stage() -> EvolutionState:
    """
    Get the current evolution state.

    Returns:
        Current evolution state
    """
    state_manager = EvolutionStateManager()
    return state_manager.get_current_state()


def get_available_transitions() -> List[Dict[str, Any]]:
    """
    Get available transitions from the current stage.

    Returns:
        List of available transitions with stage information
    """
    loader = StageLoader()
    return loader.get_available_transitions()
