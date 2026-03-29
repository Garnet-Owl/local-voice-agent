from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np

from shared.logging import setup_logging

logger = setup_logging(__name__)

LUX_SAMPLE_RATE = 48_000


@dataclass(frozen=True)
class TtsConfig:
    model_id: str
    device: str
    speed: float = 1.0
    thread_count: int = 4
    num_steps: int = 4
    t_shift: float = 0.5
    prompt_audio: str | None = None


class LuxTts:
    """TTS engine backed by LuxTTS — fast CPU/GPU voice synthesis at 48 kHz."""

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._engine = None
        self._encoded_prompt = None

    def _ensure_loaded(self) -> None:
        if self._engine is not None:
            return

        from zipvoice.luxvoice import LuxTTS

        logger.info(
            f"Loading TTS model: {self._config.model_id} on {self._config.device}"
        )

        self._engine = LuxTTS(
            model_path=self._config.model_id,
            device=self._config.device,
            threads=self._config.thread_count,
        )

        if self._config.prompt_audio:
            prompt_path = Path(self._config.prompt_audio)
            if not prompt_path.is_absolute():
                project_root = Path(__file__).resolve().parent.parent.parent
                prompt_path = project_root / prompt_path

            logger.info(f"Encoding voice prompt: {prompt_path}")
            self._encoded_prompt = self._engine.encode_prompt(
                str(prompt_path), duration=5, rms=0.01
            )
        else:
            logger.warning(
                "No prompt_audio configured. Set tts.prompt_audio in config.yaml "
                "for best voice quality."
            )
            self._encoded_prompt = self._build_default_prompt()

    def _build_default_prompt(self) -> dict:
        import torch

        device = self._config.device
        prompt_tokens = torch.zeros(1, 1, dtype=torch.long).to(device)
        prompt_features_lens = torch.tensor([1]).to(device)
        prompt_features = torch.zeros(1, 1, 100).to(device)
        prompt_rms = 0.01
        return {
            "prompt_tokens": prompt_tokens,
            "prompt_features_lens": prompt_features_lens,
            "prompt_features": prompt_features,
            "prompt_rms": prompt_rms,
        }

    def synthesize(self, text: str) -> np.ndarray:
        self._ensure_loaded()
        wav = self._engine.generate_speech(
            text,
            self._encoded_prompt,
            num_steps=self._config.num_steps,
            t_shift=self._config.t_shift,
            speed=self._config.speed,
        )
        return wav.numpy().squeeze()

    def stream(self, text: str) -> Generator[np.ndarray, None, None]:
        self._ensure_loaded()
        audio = self.synthesize(text)
        chunk_size = LUX_SAMPLE_RATE // 4
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) > 0:
                yield chunk
