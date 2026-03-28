from typing import Callable, Optional

import numpy as np
import torch

from shared.logging import setup_logging

logger = setup_logging(__name__)


class NeuralVADScanner:
    """Monitors audio streams for human speech using a neural detection model."""

    def __init__(
        self,
        sample_rate: int,
        threshold: float = 0.5,
        min_speech_chunks: int = 2,
        min_silence_chunks: int = 15,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ):
        self._rate = sample_rate
        self._sensitivity = threshold
        self._speech_trigger_limit = min_speech_chunks
        self._silence_timeout_limit = min_silence_chunks
        self._speech_start_handler = on_speech_start
        self._speech_end_handler = on_speech_end

        self._model = None
        self._is_active = True
        self._stream_buffer = bytearray()
        self._frame_samples = 512 if sample_rate == 16000 else 256
        self._frame_bytes = self._frame_samples * 2

        self._speech_frames_tally = 0
        self._silence_frames_tally = 0
        self._is_speech_active = False

        self._initialize_neural_model()

    def _initialize_neural_model(self):
        if self._model:
            return
        try:
            torch.set_num_threads(1)
            self._model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            logger.info("Neural VAD engine initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize VAD engine: {e}")

    def is_engine_ready(self) -> bool:
        return self._model is not None

    def activate(self):
        self._is_active = True

    def deactivate(self):
        self._is_active = False

    def reset_state(self):
        if self.is_engine_ready():
            self._model.reset_states()
        self._stream_buffer.clear()
        self._speech_frames_tally = 0
        self._silence_frames_tally = 0
        self._is_speech_active = False

    def ingest_audio(self, chunk: bytes):
        if not self._is_active or not self.is_engine_ready():
            return

        self._stream_buffer.extend(chunk)
        while len(self._stream_buffer) >= self._frame_bytes:
            audio_frame = self._stream_buffer[: self._frame_bytes]
            self._stream_buffer = self._stream_buffer[self._frame_bytes :]
            self._evaluate_frame(audio_frame)

    def _evaluate_frame(self, frame_bytes: bytes):
        try:
            audio_data = (
                np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )
            input_tensor = torch.from_numpy(audio_data)
            probability = self._model(input_tensor, self._rate).item()

            if probability > self._sensitivity:
                self._silence_frames_tally = 0
                if self._is_speech_active:
                    return

                self._speech_frames_tally += 1
                if self._speech_frames_tally >= self._speech_trigger_limit:
                    self._is_speech_active = True
                    self._speech_frames_tally = 0
                    if self._speech_start_handler:
                        self._speech_start_handler()
            else:
                self._speech_frames_tally = 0
                if not self._is_speech_active:
                    return

                self._silence_frames_tally += 1
                if self._silence_frames_tally >= self._silence_timeout_limit:
                    self._is_speech_active = False
                    self._silence_frames_tally = 0
                    if self._speech_end_handler:
                        self._speech_end_handler()
        except Exception as e:
            logger.error(f"VAD evaluation error: {e}")
