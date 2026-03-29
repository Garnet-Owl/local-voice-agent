import base64

import numpy as np
from shared.logging import setup_logging

logger = setup_logging(__name__)


class AudioPostProcessor:
    """Handles refinement of synthesized TTS audio."""

    def __init__(self) -> None:
        self._silence_floor = 1e-4
        self._pcm_max = 32767.0

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        peak = np.abs(audio).max()
        if peak < self._silence_floor:
            return np.array([], dtype=audio.dtype)
        if peak > 0:
            return audio / peak
        return audio

    def encode_transport(self, audio: np.ndarray) -> str:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * self._pcm_max).astype(np.int16)
        return base64.b64encode(pcm.tobytes()).decode("utf-8")
