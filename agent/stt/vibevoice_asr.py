"""stt/vibevoice_asr.py

Wraps microsoft/VibeVoice-ASR-HF for speech-to-text transcription.
Accepts a float32 numpy array (16 kHz) and returns a plain transcript string.

Requires: transformers >= 5.3.0, accelerate >= 0.34.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SttConfig:
    model_id: str
    device_map: str


class VibeVoiceAsr:
    """Lazy-loaded VibeVoice ASR transcriber."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._processor = None
        self._model = None
        self._device = torch.device(
            "cuda"
            if config.device_map == "auto" and torch.cuda.is_available()
            else "cpu"
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import (  # type: ignore
            AutoProcessor,
            VibeVoiceAsrForConditionalGeneration,
        )

        logger.info("Loading VibeVoice ASR model: %s", self._config.model_id)
        self._processor = AutoProcessor.from_pretrained(self._config.model_id)

        # Use device_map only when accelerate is available and GPU is present.
        # On CPU-only machines, load directly to avoid accelerate dependency.
        if self._device.type == "cuda":
            self._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                self._config.model_id,
                device_map="auto",
            )
        else:
            self._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
                self._config.model_id,
            ).to(self._device)

        logger.info(
            "VibeVoice ASR loaded on %s (dtype=%s)",
            next(self._model.parameters()).device,
            next(self._model.parameters()).dtype,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe float32 audio array (16 kHz mono) to text.

        Returns an empty string if no speech is recognized.
        O(n) in audio length.
        """
        self._ensure_loaded()

        inputs = self._processor.apply_transcription_request(
            audio=audio,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._model.generate(**inputs)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        result = self._processor.decode(
            generated_ids,
            return_format="transcription_only",
        )

        if isinstance(result, list):
            transcript = result[0] if result else ""
        else:
            transcript = result

        transcript = (transcript or "").strip()
        logger.info("Transcript: %r", transcript)
        return transcript
