import base64

import numpy as np
from shared.logging import setup_logging

logger = setup_logging(__name__)


class AudioPreProcessor:
    """Handles audio cleaning and normalization before STT/VAD."""

    def __init__(self) -> None:
        self._pcm_scale = 32768.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio

    def decode_pcm_bytes(self, audio_bytes: bytes) -> np.ndarray:
        return (
            np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
            / self._pcm_scale
        )

    def decode_transport(self, payload: str) -> np.ndarray:
        return self.decode_pcm_bytes(base64.b64decode(payload))
