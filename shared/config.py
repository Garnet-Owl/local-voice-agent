from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from shared.logging import setup_logging

logger = setup_logging(__name__)


class Settings(BaseSettings):
    GEMINI_API_KEY: str

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
logger.info(
    f"[config] GEMINI_API_KEY loaded: {'YES' if settings.GEMINI_API_KEY else 'NO'}"
)


def load_config(path: Path = None) -> dict:
    if path is None:
        path = Path(__file__).parent.parent / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["llm"]["api_key"] = settings.GEMINI_API_KEY
    return cfg
