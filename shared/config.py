from pathlib import Path

import yaml


def load_config():
    """Load config.yaml from the project root."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
