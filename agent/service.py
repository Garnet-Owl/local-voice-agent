from agent.llm.gemini_client import GeminiClient, LlmConfig
from agent.orchestrator import VoiceAgentOrchestrator
from agent.stt.whisper_asr import SttConfig, WhisperAsr
from agent.tts.vits_tts import TtsConfig, VitsTts
from shared.config import load_config
from shared.logging import setup_logging

logger = setup_logging("service")


class VoiceAgentService:
    """Manages the low-level initialization and lifecycle of agent engines."""

    def __init__(self):
        self.stt_engine = None
        self.llm_engine = None
        self.tts_engine = None
        self.orchestrator = None

    def initialize(self):
        cfg = load_config()
        logger.info("Initializing voice agent engines...")

        stt_cfg = SttConfig(
            model_id=cfg["stt"]["model_id"], device=cfg["stt"]["device"]
        )
        self.stt_engine = WhisperAsr(stt_cfg)
        self.stt_engine._ensure_loaded()

        llm_cfg = LlmConfig(
            model=cfg["llm"]["model"], system_prompt=cfg["llm"]["system_prompt"]
        )
        self.llm_engine = GeminiClient(llm_cfg)

        tts_cfg = TtsConfig(
            model_id=cfg["tts"]["model_id"], device=cfg["tts"]["device"]
        )
        self.tts_engine = VitsTts(tts_cfg)
        self.tts_engine._ensure_loaded()

        self.orchestrator = VoiceAgentOrchestrator(
            self.stt_engine, self.llm_engine, self.tts_engine
        )

        logger.info("All engines initialized and warmed up.")
        return self.orchestrator
