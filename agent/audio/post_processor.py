import numpy as np
from shared.logging import setup_logging

logger = setup_logging("post_processor")


class AudioPostProcessor:
    """Handles refinement of synthesized TTS audio."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        peak = np.abs(audio).max()
        if peak > 0:
            return audio / peak
        return audio
