"""audio_capture/vad_recorder.py

Captures microphone audio using voice activity detection (VAD).
Records until silence_timeout_sec of silence is detected, then
returns the full utterance as a float32 numpy array at 16 kHz.
"""

from __future__ import annotations

import collections
import logging
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import webrtcvad

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2  # int16


@dataclass(frozen=True)
class CaptureConfig:
    sample_rate: int
    vad_aggressiveness: int
    silence_timeout_sec: float
    min_speech_sec: float
    frame_duration_ms: int


def _frame_bytes(config: CaptureConfig) -> int:
    samples = config.sample_rate * config.frame_duration_ms // 1000
    return samples * _BYTES_PER_SAMPLE


def record_utterance(config: CaptureConfig) -> np.ndarray | None:
    """Block until one complete utterance is captured via VAD.

    Returns float32 numpy array at config.sample_rate, or None if the
    captured audio was shorter than min_speech_sec (e.g., background noise).
    """
    vad = webrtcvad.Vad(config.vad_aggressiveness)
    frame_bytes = _frame_bytes(config)
    samples_per_frame = frame_bytes // _BYTES_PER_SAMPLE

    silence_frames_needed = int(
        config.silence_timeout_sec * 1000 / config.frame_duration_ms
    )
    min_speech_frames = int(
        config.min_speech_sec * 1000 / config.frame_duration_ms
    )

    # Ring buffer tracks recent silence for end-of-speech detection.
    ring = collections.deque(maxlen=silence_frames_needed)

    speech_frames: list[bytes] = []
    triggered = False
    audio_buffer = b""

    logger.info("Listening... (speak now)")

    with sd.RawInputStream(
        samplerate=config.sample_rate,
        blocksize=samples_per_frame,
        dtype="int16",
        channels=1,
    ) as stream:
        while True:
            chunk, _ = stream.read(samples_per_frame)
            audio_buffer += bytes(chunk)

            # Accumulate until we have a full VAD frame.
            while len(audio_buffer) >= frame_bytes:
                frame = audio_buffer[:frame_bytes]
                audio_buffer = audio_buffer[frame_bytes:]
                is_speech = vad.is_speech(frame, config.sample_rate)

                if not triggered:
                    ring.append((frame, is_speech))
                    num_voiced = sum(1 for _, speech in ring if speech)
                    if num_voiced > 0.9 * ring.maxlen:
                        triggered = True
                        logger.info("Speech detected.")
                        for f, _ in ring:
                            speech_frames.append(f)
                        ring.clear()
                else:
                    speech_frames.append(frame)
                    ring.append((frame, is_speech))
                    num_unvoiced = sum(1 for _, speech in ring if not speech)
                    if num_unvoiced == ring.maxlen:
                        logger.info("Silence detected — end of utterance.")
                        triggered = False

                        if len(speech_frames) < min_speech_frames:
                            logger.debug("Too short, discarding.")
                            speech_frames.clear()
                            ring.clear()
                            continue

                        raw = b"".join(speech_frames)
                        audio_int16 = np.frombuffer(raw, dtype=np.int16)
                        audio_float32 = audio_int16.astype(np.float32) / 32768.0
                        return audio_float32
