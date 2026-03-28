import numpy as np
from shared.logging import setup_logging

logger = setup_logging("post_processor")


class AudioPostProcessor:
    """Handles refinement of synthesized TTS audio."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply volume normalization or other post-synthesis refinements."""
        if len(audio) == 0:
            return audio

        # Peak normalization as a default safeguard
        peak = np.abs(audio).max()
        if peak > 0:
            return audio / peak
        return audio
