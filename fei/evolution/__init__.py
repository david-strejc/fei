import importlib
import json
from pathlib import Path
from typing import Any

from fei.utils.logging import get_logger
from fei.core.assistant import Assistant
from fei.tools.registry import ToolRegistry
# Assuming MemdirConnector might be needed later for reading stage from memory
# from fei.tools.memdir_connector import MemdirConnector

logger = get_logger(__name__)

MANIFEST_PATH = Path(__file__).parent / "manifest.json"

def load_stage(assistant: Assistant, tool_registry: ToolRegistry) -> int:
    """
    Loads the current evolution stage based on the manifest.

    Reads the manifest file, determines the current stage ID (currently hardcoded
    from the manifest, but planned to be read from Memdir), dynamically imports
    the corresponding stage module, and calls its apply_stage_modifications function.

    Args:
        assistant: The Assistant instance.
        tool_registry: The ToolRegistry instance.

    Returns:
        The ID of the loaded stage.

    Raises:
        FileNotFoundError: If the manifest file is not found.
        KeyError: If the manifest structure is invalid.
        ImportError: If the stage module cannot be imported.
        AttributeError: If the stage module lacks apply_stage_modifications.
        Exception: For other errors during stage loading/application.
    """
    logger.info("Loading evolution stage...")

    if not MANIFEST_PATH.exists():
        logger.error(f"Evolution manifest not found at {MANIFEST_PATH}")
        raise FileNotFoundError(f"Evolution manifest not found at {MANIFEST_PATH}")

    try:
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)

        # TODO: Replace this with reading from Memdir status
        current_stage_id = manifest.get("current_stage_id", 0)
        logger.info(f"Determined current stage ID: {current_stage_id}")

        stage_info = None
        for stage in manifest.get("stages", []):
            if stage.get("id") == current_stage_id:
                stage_info = stage
                break

        if not stage_info:
            logger.error(f"Stage ID {current_stage_id} not found in manifest.")
            raise KeyError(f"Stage ID {current_stage_id} not found in manifest.")

        module_path = stage_info.get("module")
        if not module_path:
            logger.error(f"Module path not defined for stage ID {current_stage_id}.")
            raise KeyError(f"Module path not defined for stage ID {current_stage_id}.")

        logger.info(f"Importing stage module: {module_path}")
        stage_module = importlib.import_module(module_path)

        apply_func = getattr(stage_module, "apply_stage_modifications", None)
        if not callable(apply_func):
            logger.error(f"Function 'apply_stage_modifications' not found or not callable in {module_path}.")
            raise AttributeError(f"Function 'apply_stage_modifications' not found or not callable in {module_path}.")

        logger.info(f"Applying modifications for stage {current_stage_id}...")
        apply_func(assistant, tool_registry)
        logger.info(f"Successfully loaded and applied stage {current_stage_id}.")

        return current_stage_id

    except (json.JSONDecodeError, FileNotFoundError, KeyError, ImportError, AttributeError) as e:
        logger.exception(f"Failed to load evolution stage: {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred during stage loading: {e}")
        raise
