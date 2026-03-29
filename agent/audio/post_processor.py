import numpy as np
import librosa
from shared.logging import setup_logging

logger = setup_logging(__name__)


class AudioPostProcessor:
    """Handles refinement of synthesized TTS audio."""

    def __init__(self) -> None:
        pass

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        # Trim leading/trailing silence (below 30dB)
        trimmed, _ = librosa.effects.trim(audio, top_db=30)

        peak = np.abs(trimmed).max()
        if peak > 0:
            return trimmed / peak
        return trimmed
