"""agent/__main__.py

Entry point: python -m agent

Loads config.yaml, constructs all feature configs, and starts the
VoiceAgentOrchestrator loop.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv


from agent.audio_capture.vad_recorder import CaptureConfig
from agent.llm.gemini_client import LlmConfig
from agent.orchestrator import VoiceAgentOrchestrator
from agent.stt.whisper_asr import SttConfig
from agent.tts.speecht5_tts import TtsConfig

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        logger.error("config.yaml not found at %s", _CONFIG_PATH)
        sys.exit(1)
    with _CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        logger.error(
            "Required environment variable '%s' is not set. "
            "Set it before running:  set %s=your_key_here",
            name,
            name,
        )
        sys.exit(1)
    return value


def main() -> None:
    load_dotenv()
    _require_env("GEMINI_API_KEY")

    cfg = _load_config()

    capture_cfg = CaptureConfig(
        sample_rate=cfg["audio_capture"]["sample_rate"],
        vad_aggressiveness=cfg["audio_capture"]["vad_aggressiveness"],
        silence_timeout_sec=cfg["audio_capture"]["silence_timeout_sec"],
        min_speech_sec=cfg["audio_capture"]["min_speech_sec"],
        frame_duration_ms=cfg["audio_capture"]["frame_duration_ms"],
    )
    stt_cfg = SttConfig(
        model_id=cfg["stt"]["model_id"],
        device=cfg["stt"]["device"],
    )
    llm_cfg = LlmConfig(
        model=cfg["llm"]["model"],
        system_prompt=cfg["llm"]["system_prompt"],
    )
    tts_cfg = TtsConfig(
        model_id=cfg["tts"]["model_id"],
        vocoder_id=cfg["tts"]["vocoder_id"],
        device=cfg["tts"]["device"],
    )

    orchestrator = VoiceAgentOrchestrator(capture_cfg, stt_cfg, llm_cfg, tts_cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
