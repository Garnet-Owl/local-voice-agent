import asyncio
import numpy as np
import torch
from shared.logging import setup_logging

logger = setup_logging(__name__)


async def check_stt_health(stt_engine):
    try:
        dummy_audio = np.zeros(16000, dtype=np.float32)
        stt_engine.transcribe(dummy_audio)
        logger.info("WhisperAsr Health Check: OK")
        return True
    except Exception as e:
        logger.error(f"WhisperAsr Health Check FAILED: {e}")
        return False


async def check_llm_health(llm_engine):
    try:
        chat = llm_engine.new_chat()
        async for _ in llm_engine.stream_reply("health check", chat):
            break
        logger.info("GeminiClient Health Check: OK")
        return True
    except Exception as e:
        logger.error(f"GeminiClient Health Check FAILED: {e}")
        return False


async def check_tts_health(tts_engine):
    try:
        tts_engine.synthesize("health check")
        logger.info("KokoroTts Health Check: OK")
        return True
    except Exception as e:
        logger.error(f"KokoroTts Health Check FAILED: {e}")
        return False


async def check_vad_health(vad_engine):
    try:
        if not vad_engine.is_engine_ready():
            logger.error("VAD Health Check: Engine not ready.")
            return False
        dummy_frame = np.zeros(512, dtype=np.float32)
        input_tensor = torch.from_numpy(dummy_frame)
        vad_engine._model(input_tensor, vad_engine._rate)
        logger.info("VAD Health Check: OK")
        return True
    except Exception as e:
        logger.error(f"VAD Health Check FAILED: {e}")
        return False


async def run_pipeline_diagnostic(orchestrator, audio_data: np.ndarray):
    try:
        transcript = await asyncio.to_thread(orchestrator._stt.transcribe, audio_data)
        if not transcript:
            return False

        chat = orchestrator._llm.new_chat()
        reply = ""
        async for chunk in orchestrator._llm.stream_reply(transcript, chat):
            reply += chunk
        if not reply:
            return False

        audio_out = await asyncio.to_thread(orchestrator._tts.synthesize, "Test.")
        if audio_out is None or len(audio_out) == 0:
            return False

        logger.info("Full Pipeline Diagnostic: OK")
        return True
    except Exception as e:
        logger.error(f"Full Pipeline Diagnostic FAILED: {e}")
        return False


async def get_health_status(orchestrator):
    if orchestrator:
        status = {
            "stt": await check_stt_health(orchestrator._stt),
            "llm": await check_llm_health(orchestrator._llm),
            "tts": await check_tts_health(orchestrator._tts),
            "vad": await check_vad_health(orchestrator._vad),
        }
        return {
            "status": "ok" if all(status.values()) else "degraded",
            "details": status,
        }
    return {"status": "initializing"}


async def get_pipeline_health(orchestrator):
    if orchestrator:
        dummy_audio = np.random.uniform(-0.01, 0.01, 16000).astype(np.float32)
        success = await run_pipeline_diagnostic(orchestrator, dummy_audio)
        return {"status": "healthy" if success else "unhealthy"}
    return {"status": "initializing"}
