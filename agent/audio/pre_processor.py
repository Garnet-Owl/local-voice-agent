import numpy as np
from shared.logging import setup_logging

logger = setup_logging(__name__)


class AudioPreProcessor:
    """Handles audio cleaning and normalization before STT/VAD."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        return audio
