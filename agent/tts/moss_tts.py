"""tts/moss_tts.py

Wraps MOSS-TTS-Realtime for streaming text-to-speech synthesis.

Requires the MOSS-TTS repository installed as an editable package:
    git clone https://github.com/OpenMOSS/MOSS-TTS.git
    cd MOSS-TTS && pip install -e .

The TTS session is multi-turn aware: the MOSS-TTS-Realtime model
conditions on prior acoustic context across dialogue turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Generator, Iterable

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TtsConfig:
    model_id: str
    codec_id: str
    prompt_wav: str
    sample_rate: int
    device: str


class MossTts:
    """Lazy-loaded MOSS-TTS-Realtime synthesizer.

    Maintains an internal multi-turn session so voice stays coherent
    across conversation turns.
    """

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._model = None
        self._codec = None
        self._session = None
        self._decoder = None
        self._codebook_size: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        try:
            from moss_tts_realtime import (  # type: ignore
                MossTTSRealtime,
                MossAudioTokenizer,
                RealtimeSession,
            )
        except ImportError as exc:
            raise ImportError(
                "MOSS-TTS is not installed. Run:\n"
                "  git clone https://github.com/OpenMOSS/MOSS-TTS.git\n"
                "  cd MOSS-TTS && pip install -e ."
            ) from exc

        logger.info("Loading MOSS-TTS-Realtime: %s", self._config.model_id)
        self._model = MossTTSRealtime.from_pretrained(
            self._config.model_id,
            device=self._config.device,
        )
        logger.info("Loading MOSS Audio Tokenizer: %s", self._config.codec_id)
        self._codec = MossAudioTokenizer.from_pretrained(
            self._config.codec_id,
            device=self._config.device,
        )
        self._codebook_size = self._codec.codebook_size
        self._decoder = self._codec.get_decoder()
        self._session = RealtimeSession(
            model=self._model,
            prompt_wav=self._config.prompt_wav,
        )
        logger.info("MOSS-TTS-Realtime loaded.")

    def _decode_audio_frames(
        self, audio_frames: list
    ) -> Generator[np.ndarray, None, None]:
        """Decode raw codec frames to float32 audio chunks."""
        for frame in audio_frames:
            pcm = self._decoder.decode(frame, self._codebook_size)
            if pcm is not None and len(pcm) > 0:
                yield pcm.astype(np.float32)

    def _flush_decoder(self) -> Generator[np.ndarray, None, None]:
        """Flush any remaining audio from the decoder."""
        remaining = self._decoder.flush()
        if remaining is not None and len(remaining) > 0:
            yield remaining.astype(np.float32)

    def synthesize_and_play(self, text_chunks: Iterable[str]) -> None:
        """Consume streaming text chunks, synthesize speech, play in real time.

        Feeds text deltas into the MOSS-TTS-Realtime session as they arrive
        from the LLM, and plays audio frames as soon as they are decoded.
        """
        self._ensure_loaded()

        with sd.OutputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="float32",
        ) as stream:
            with self._codec.streaming(batch_size=1):
                for delta in text_chunks:
                    if not delta:
                        continue
                    audio_frames = self._session.push_text(delta)
                    for chunk in self._decode_audio_frames(audio_frames):
                        stream.write(chunk)

                # Signal end of text and drain remaining audio.
                audio_frames = self._session.end_text()
                for chunk in self._decode_audio_frames(audio_frames):
                    stream.write(chunk)

                while True:
                    audio_frames = self._session.drain(max_steps=1)
                    if not audio_frames:
                        break
                    for chunk in self._decode_audio_frames(audio_frames):
                        stream.write(chunk)
                    if self._session.inferencer.is_finished:
                        break

                for chunk in self._flush_decoder():
                    stream.write(chunk)
