import os
from pathlib import Path

import yaml
from dotenv import load_dotenv, find_dotenv


def load_config():
    load_dotenv(find_dotenv())
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["llm"]["api_key"] = os.getenv("GEMINI_API_KEY")
    return cfg
