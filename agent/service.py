import asyncio
from agent.audio.silero_vad import NeuralVADScanner
from agent.llm.gemini_client import GeminiClient, LlmConfig
from agent.orchestrator import VoiceAgentOrchestrator
from agent.stt.vosk_asr import SttConfig, VoskAsr
from agent.tts.lux_tts import TtsConfig, LuxTts
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

    async def initialize(self):
        cfg = load_config()
        logger.info("Initializing voice agent engines...")

        stt_cfg = SttConfig(
            model_path=cfg["stt"]["model_path"],
        )
        self.stt_engine = VoskAsr(stt_cfg)
        self.stt_engine._ensure_loaded()

        api_key = cfg["llm"]["api_key"]
        logger.info(
            f"[service] GEMINI_API_KEY present before GeminiClient: {'YES' if api_key else 'NO'}"
        )
        llm_cfg = LlmConfig(
            model=cfg["llm"]["model"],
            system_prompt=cfg["llm"]["system_prompt"],
            api_key=api_key,
        )
        self.llm_engine = GeminiClient(llm_cfg)

        tts_cfg = TtsConfig(
            model_id=cfg["tts"]["model_id"],
            device=cfg["tts"]["device"],
            speed=cfg["tts"].get("speed", 1.0),
            thread_count=cfg["tts"].get("thread_count", 4),
            num_steps=cfg["tts"].get("num_steps", 4),
            t_shift=cfg["tts"].get("t_shift", 0.5),
            prompt_audio=cfg["tts"].get("prompt_audio"),
        )
        self.tts_engine = LuxTts(tts_cfg)
        self.tts_engine._ensure_loaded()

        self.vad_engine = NeuralVADScanner(
            sample_rate=cfg["audio_capture"]["sample_rate"],
            threshold=cfg["audio_capture"]["vad_threshold"],
        )

        health_results = await asyncio.gather(
            check_stt_health(self.stt_engine),
            check_llm_health(self.llm_engine),
            check_tts_health(self.tts_engine),
            check_vad_health(self.vad_engine),
        )

        if not all(health_results):
            logger.error("One or more engine health checks failed.")
            raise RuntimeError(
                "Voice agent engine health check failed. See logs for details."
            )

        self.orchestrator = VoiceAgentOrchestrator(
            self.stt_engine, self.llm_engine, self.tts_engine, self.vad_engine
        )

        logger.info("All engines initialized and warmed up.")
        return self.orchestrator
