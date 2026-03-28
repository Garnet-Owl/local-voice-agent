import numpy as np
from shared.logging import setup_logging

logger = setup_logging("pre_processor")


class AudioPreProcessor:
    """Handles audio cleaning and normalization before STT/VAD."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-processing steps to the audio array."""
        # TODO: Implement noise reduction or DC offset removal if needed
        return audio
