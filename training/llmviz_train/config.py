"""Load training configuration from config.yaml in the training dir."""

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Read config.yaml (relative to the training dir by default)."""
    with open(path) as f:
        return yaml.safe_load(f)
