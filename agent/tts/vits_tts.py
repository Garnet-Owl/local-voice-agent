import os
from dataclasses import dataclass
import torch
import numpy as np

# EXPLICITLY TELL WINDOWS WHERE ESPEAK IS LOCATED
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

from transformers import VitsModel, AutoTokenizer

@dataclass(frozen=True)
class TtsConfig:
    model_id: str
    device: str

class VitsTts:
    """Ultra-fast, clear TTS using VITS (<150MB)."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._tokenizer = None
        self._model = None
        self._device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_id)
        self._model = VitsModel.from_pretrained(self._config.model_id).to(self._device)

    def synthesize(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output = self._model(**inputs).waveform
        return output.cpu().numpy().squeeze()
