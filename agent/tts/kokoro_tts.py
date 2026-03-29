from dataclasses import dataclass
from typing import Generator

import numpy as np

from shared.logging import setup_logging

logger = setup_logging(__name__)

KOKORO_SAMPLE_RATE = 24_000


@dataclass(frozen=True)
class TtsConfig:
    model_id: str
    device: str
    speed: float = 1.0
    thread_count: int | None = None


class KokoroTts:
    """TTS engine backed by Kokoro 82M — fast CPU streaming with high-quality voice."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._pipeline = None
        import torch

        if config.device == "cpu" and config.thread_count is not None:
            torch.set_num_threads(config.thread_count)
            torch.set_num_interop_threads(1)
            logger.info(
                f"Torch thread count configured for TTS: {config.thread_count}."
            )

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        from kokoro import KPipeline

        logger.info(
            f"Loading TTS model: {self._config.model_id} on {self._config.device}"
        )
        self._pipeline = KPipeline(lang_code="a", device=self._config.device)

    def synthesize(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        chunks = list(
            self._pipeline(
                text,
                voice=self._config.model_id,
                speed=self._config.speed,
            )
        )
        return np.concatenate(
            [c.numpy() if hasattr(c, "numpy") else c for _, _, c in chunks]
        )

    def stream(self, text: str) -> Generator[np.ndarray, None, None]:
        self._ensure_loaded()
        for _, _, audio in self._pipeline(
            text,
            voice=self._config.model_id,
            speed=self._config.speed,
        ):
            chunk = audio.numpy() if hasattr(audio, "numpy") else audio
            if chunk is not None and len(chunk) > 0:
                yield chunk
