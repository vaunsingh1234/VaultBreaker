import os
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load and parse YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def get_default_config_path() -> str:
    """Return standard config path."""
    current = Path(__file__).resolve().parent
    root = current.parent.parent.parent
    candidate = root / "configs" / "default.yaml"
    if candidate.exists():
        return str(candidate)
    return "configs/default.yaml"
