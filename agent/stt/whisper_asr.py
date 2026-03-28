"""stt/whisper_asr.py

Wraps openai/whisper-base for speech-to-text transcription.
~74MB model, CPU-friendly, no reference audio required.

Accepts a float32 numpy array (16 kHz mono) and returns a transcript string.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)

_WHISPER_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SttConfig:
    model_id: str
    device: str


class WhisperAsr:
    """Lazy-loaded Whisper ASR transcriber."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._pipeline = None
        self._device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return

        from transformers import pipeline  # type: ignore

        logger.info("Loading Whisper ASR model: %s", self._config.model_id)
        self._pipeline = pipeline(
            task="automatic-speech-recognition",
            model=self._config.model_id,
            device=self._device,
        )
        logger.info("Whisper ASR loaded on %s", self._device)

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 audio array (16 kHz mono) to text.

        Returns an empty string if no speech is recognized.
        O(n) in audio length.
        """
        self._ensure_loaded()

        result = self._pipeline(
            {"raw": audio, "sampling_rate": _WHISPER_SAMPLE_RATE},
            return_timestamps=False,
        )
        transcript = (result.get("text") or "").strip()
        logger.info("Transcript: %r", transcript)
        return transcript
