"""agent/orchestrator.py

Wires the voice agent pipeline:
    mic (VAD) -> STT (Whisper) -> LLM (Gemini Flash) -> TTS (SpeechT5)

Maintains full multi-turn LLM conversation history across the session.
"""

from __future__ import annotations

import logging

from google.genai import types

from agent.audio_capture.vad_recorder import CaptureConfig, record_utterance
from agent.llm.gemini_client import GeminiClient, LlmConfig
from agent.stt.whisper_asr import SttConfig, WhisperAsr
from agent.tts.speecht5_tts import SpeechT5Tts, TtsConfig

logger = logging.getLogger(__name__)

_GREETING = (
    "\n  Local Voice Agent ready. Speak after the prompt -- "
    "pause to finish your turn. Press Ctrl+C to quit.\n"
)
_TURN_SEPARATOR = "-" * 48


class VoiceAgentOrchestrator:
    """Runs the voice agent loop until interrupted."""

    def __init__(
        self,
        capture_config: CaptureConfig,
        stt_config: SttConfig,
        llm_config: LlmConfig,
        tts_config: TtsConfig,
    ) -> None:
        self._capture_config = capture_config
        self._asr = WhisperAsr(stt_config)
        self._llm = GeminiClient(llm_config)
        self._tts = SpeechT5Tts(tts_config)
        self._history: list[types.Content] = []

    def run(self) -> None:
        """Block and run the voice agent loop."""
        print(_GREETING)
        try:
            while True:
                self._run_turn()
        except KeyboardInterrupt:
            print("\n\nSession ended.")

    def _run_turn(self) -> None:
        """Execute one full listen -> think -> speak turn."""
        print(_TURN_SEPARATOR)
        audio = record_utterance(self._capture_config)

        if audio is None:
            logger.debug("No utterance captured, retrying.")
            return

        print("Transcribing...")
        transcript = self._asr.transcribe(audio)

        if not transcript:
            print("  (No speech recognized, listening again.)")
            return

        print(f"  You: {transcript}")
        print("  Agent: ", end="", flush=True)

        text_chunks = self._llm.stream_reply(transcript, self._history)

        # Echo text to stdout as it streams, then TTS speaks the full reply.
        collected = list(self._echo_and_yield(text_chunks))
        print()
        self._tts.synthesize_and_play(iter(collected))

    @staticmethod
    def _echo_and_yield(chunks):
        """Print each text chunk to stdout as it passes through."""
        for chunk in chunks:
            print(chunk, end="", flush=True)
            yield chunk
