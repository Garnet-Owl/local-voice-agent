"""tts/speecht5_tts.py

Wraps microsoft/speecht5_tts + speecht5_hifigan for text-to-speech.
~200MB combined, CPU-friendly, no voice cloning or reference audio required.

Uses a fixed speaker embedding from the CMU Arctic dataset (xvector).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
# Speaker embedding dataset — small xvector lookup table (~1MB).
_SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"
# Speaker index 7306 is a clear neutral American-English voice.
_SPEAKER_IDX = 7306


@dataclass(frozen=True)
class TtsConfig:
    model_id: str
    vocoder_id: str
    device: str


class SpeechT5Tts:
    """Lazy-loaded SpeechT5 TTS synthesizer.

    Buffers the full LLM reply before synthesizing — SpeechT5 is a
    sentence-level model, not a streaming token-level model, so we
    collect all chunks first then speak the complete response.
    """

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._tts = None
        self._vocoder = None
        self._speaker_embedding = None
        self._device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return

        from datasets import load_dataset  # type: ignore
        from transformers import (  # type: ignore
            SpeechT5ForTextToSpeech,
            SpeechT5HifiGan,
            SpeechT5Processor,
        )

        logger.info("Loading SpeechT5 TTS: %s", self._config.model_id)
        self._processor = SpeechT5Processor.from_pretrained(self._config.model_id)
        self._tts = SpeechT5ForTextToSpeech.from_pretrained(
            self._config.model_id
        ).to(self._device)

        logger.info("Loading SpeechT5 vocoder: %s", self._config.vocoder_id)
        self._vocoder = SpeechT5HifiGan.from_pretrained(
            self._config.vocoder_id
        ).to(self._device)

        logger.info("Loading speaker embedding from %s", _SPEAKER_DATASET)
        dataset = load_dataset(_SPEAKER_DATASET, split="validation")
        embedding = torch.tensor(
            dataset[_SPEAKER_IDX]["xvector"]
        ).unsqueeze(0).to(self._device)
        self._speaker_embedding = embedding
        logger.info("SpeechT5 TTS ready on %s", self._device)

    def _synthesize(self, text: str) -> np.ndarray:
        """Synthesize a single text string to a float32 audio array."""
        inputs = self._processor(text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            speech = self._tts.generate_speech(
                inputs["input_ids"],
                self._speaker_embedding,
                vocoder=self._vocoder,
            )
        return speech.cpu().numpy()

    def synthesize_and_play(self, text_chunks: Iterable[str]) -> None:
        """Collect all LLM text chunks, synthesize, then play.

        SpeechT5 is a full-sentence model — streaming token-by-token is
        not supported. We buffer the complete reply before synthesis.
        """
        self._ensure_loaded()

        full_text = "".join(text_chunks).strip()
        if not full_text:
            return

        # SpeechT5 has a 600-token limit; split on sentences for long replies.
        sentences = _split_sentences(full_text)
        audio_chunks: list[np.ndarray] = []
        for sentence in sentences:
            if sentence.strip():
                audio_chunks.append(self._synthesize(sentence))

        if not audio_chunks:
            return

        audio = np.concatenate(audio_chunks)
        sd.play(audio, samplerate=_SAMPLE_RATE)
        sd.wait()


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence boundaries to stay within SpeechT5 token limit."""
    import re
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
