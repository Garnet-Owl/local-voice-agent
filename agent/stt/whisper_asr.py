from dataclasses import dataclass

import numpy as np
import torch
from transformers import pipeline

from shared.logging import setup_logging

logger = setup_logging(__name__)

WHISPER_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SttConfig:
    model_id: str
    device: str


class WhisperAsr:
    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._pipeline = None
        self._device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        logger.info(f"Loading Whisper model: {self._config.model_id}")
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self._config.model_id,
            device=self._device,
        )
        self._pipeline.model.generation_config.forced_decoder_ids = None

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_loaded()
        result = self._pipeline(
            {"raw": audio, "sampling_rate": WHISPER_SAMPLE_RATE},
            return_timestamps=False,
        )
        return (result.get("text") or "").strip()
