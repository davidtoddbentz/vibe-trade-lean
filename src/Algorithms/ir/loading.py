"""IR loading utilities.

Phase 12: Extracted from StrategyRuntime.
"""

from __future__ import annotations

import json


def load_ir_from_file(path: str, data_folder: str) -> dict:
    """Load strategy IR from JSON file.

    Args:
        path: Path to IR JSON file
        data_folder: Data folder path (default /Data)

    Returns:
        IR dict

    Raises:
        ValueError: If file not found or invalid JSON
    """
    # If path is relative, resolve against data folder
    if not path.startswith("/"):
        full_path = f"{data_folder}/{path}"
    else:
        full_path = path

    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Strategy IR file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in strategy IR file: {e}")
