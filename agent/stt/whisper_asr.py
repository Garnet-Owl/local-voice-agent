from dataclasses import dataclass

import numpy as np
from transformers import pipeline

from shared.logging import setup_logging

logger = setup_logging(__name__)

MOONSHINE_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SttConfig:
    model_id: str
    device: str


class WhisperAsr:
    """STT engine backed by Moonshine — 9x faster than real-time on CPU."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        logger.info(f"Loading STT model: {self._config.model_id}")
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self._config.model_id,
            device=self._config.device if self._config.device == "cuda" else -1,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_loaded()
        result = self._pipeline(
            {"raw": audio, "sampling_rate": MOONSHINE_SAMPLE_RATE},
        )
        return (result.get("text") or "").strip()
