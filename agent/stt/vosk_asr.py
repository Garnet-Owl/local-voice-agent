import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared.logging import setup_logging

logger = setup_logging(__name__)

VOSK_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class SttConfig:
    model_path: str
    sample_rate: int = VOSK_SAMPLE_RATE


class VoskAsr:
    """STT engine backed by Vosk for lightweight, offline speech recognition."""

    def __init__(self, config: SttConfig) -> None:
        self._config = config
        self._model = None
        self._sample_rate = config.sample_rate

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from vosk import Model, SetLogLevel

        SetLogLevel(-1)

        model_path = Path(self._config.model_path)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = project_root / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"Vosk model not found at {model_path}. "
                "Download from https://alphacephei.com/vosk/models"
            )

        logger.info(f"Loading STT model: {model_path.name}")
        self._model = Model(str(model_path))

    def transcribe(self, audio: np.ndarray) -> str:
        self._ensure_loaded()
        from vosk import KaldiRecognizer

        recognizer = KaldiRecognizer(self._model, self._sample_rate)

        if audio.dtype == np.float32:
            pcm_data = (audio * 32767).astype(np.int16).tobytes()
        else:
            pcm_data = audio.astype(np.int16).tobytes()

        recognizer.AcceptWaveform(pcm_data)
        result = json.loads(recognizer.FinalResult())
        return (result.get("text") or "").strip()
