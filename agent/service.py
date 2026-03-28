import asyncio
from agent.audio.silero_vad import NeuralVADScanner
from agent.llm.gemini_client import GeminiClient, LlmConfig
from agent.orchestrator import VoiceAgentOrchestrator
from agent.stt.whisper_asr import SttConfig, WhisperAsr
from agent.tts.vits_tts import TtsConfig, VitsTts
from shared.config import load_config
from shared.logging import setup_logging
from main import check_stt_health, check_llm_health, check_tts_health, check_vad_health

logger = setup_logging(__name__)


class VoiceAgentService:
    """Manages the low-level initialization and lifecycle of agent engines."""

    def __init__(self):
        self.stt_engine = None
        self.llm_engine = None
        self.tts_engine = None
        self.vad_engine = None
        self.orchestrator = None

    def initialize(self):
        cfg = load_config()
        logger.info("Initializing voice agent engines...")

        stt_cfg = SttConfig(
            model_id=cfg["stt"]["model_id"], device=cfg["stt"]["device"]
        )
        self.stt_engine = WhisperAsr(stt_cfg)
        self.stt_engine._ensure_loaded()
        asyncio.run(check_stt_health(self.stt_engine))

        llm_cfg = LlmConfig(
            model=cfg["llm"]["model"], system_prompt=cfg["llm"]["system_prompt"]
        )
        self.llm_engine = GeminiClient(llm_cfg)
        asyncio.run(check_llm_health(self.llm_engine))

        tts_cfg = TtsConfig(
            model_id=cfg["tts"]["model_id"], device=cfg["tts"]["device"]
        )
        self.tts_engine = VitsTts(tts_cfg)
        self.tts_engine._ensure_loaded()
        asyncio.run(check_tts_health(self.tts_engine))

        self.vad_engine = NeuralVADScanner(
            sample_rate=cfg["audio_capture"]["sample_rate"],
            threshold=cfg["audio_capture"]["vad_threshold"],
        )
        asyncio.run(check_vad_health(self.vad_engine))

        self.orchestrator = VoiceAgentOrchestrator(
            self.stt_engine, self.llm_engine, self.tts_engine, self.vad_engine
        )

        logger.info("All engines initialized and warmed up.")
        return self.orchestrator
