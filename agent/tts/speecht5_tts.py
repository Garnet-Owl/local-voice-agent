from dataclasses import dataclass
from typing import Iterable
import sounddevice as sd
import torch
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset


@dataclass(frozen=True)
class TtsConfig:
    model_id: str = "microsoft/speecht5_tts"
    device: str = "cpu"


class SpeechT5Tts:
    """Fast, CPU-friendly TTS using SpeechT5 (<200MB)."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._processor = None
        self._model = None
        self._vocoder = None
        self._speaker_embeddings = None
        self._device = torch.device(
            "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        self._processor = SpeechT5Processor.from_pretrained(self._config.model_id)
        self._model = SpeechT5ForTextToSpeech.from_pretrained(self._config.model_id).to(
            self._device
        )
        self._vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self._device)

        # ADDED trust_remote_code=True here
        embeddings_dataset = load_dataset(
            "regisss/cmu-arctic-xvectors", split="validation"
        )
        self._speaker_embeddings = (
            torch.tensor(embeddings_dataset[7306]["xvector"])
            .unsqueeze(0)
            .to(self._device)
        )

    def synthesize(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        inputs = self._processor(text=text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            speech = self._model.generate_speech(
                inputs["input_ids"], self._speaker_embeddings, vocoder=self._vocoder
            )
        return speech.cpu().numpy()

    def synthesize_and_play(self, text_chunks: Iterable[str]) -> None:
        full_text = "".join(text_chunks).strip()
        if not full_text:
            return
        audio = self.synthesize(full_text)
        sd.play(audio, samplerate=16000)
        sd.wait()
