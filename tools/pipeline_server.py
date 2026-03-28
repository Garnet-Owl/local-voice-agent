import base64
import contextlib
import io
import logging
import os
import warnings
from pathlib import Path

import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agent.llm.gemini_client import GeminiClient, LlmConfig
from agent.orchestrator import VoiceAgentOrchestrator
from agent.stt.whisper_asr import SttConfig, WhisperAsr
from agent.tts.vits_tts import TtsConfig, VitsTts

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("server")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global orchestrator
    cfg = load_config()

    stt_cfg = SttConfig(model_id=cfg["stt"]["model_id"], device=cfg["stt"]["device"])
    stt_engine = WhisperAsr(stt_cfg)
    stt_engine._ensure_loaded()

    llm_cfg = LlmConfig(
        model=cfg["llm"]["model"], system_prompt=cfg["llm"]["system_prompt"]
    )
    llm_engine = GeminiClient(llm_cfg)

    tts_cfg = TtsConfig(model_id=cfg["tts"]["model_id"], device=cfg["tts"]["device"])
    tts_engine = VitsTts(tts_cfg)
    tts_engine._ensure_loaded()

    orchestrator = VoiceAgentOrchestrator(stt_engine, llm_engine, tts_engine)

    logger.info("Server is ready and standing by.")
    yield


app = FastAPI(title="Local Voice Agent Pipeline Server", lifespan=lifespan)
orchestrator = None


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected.")
    history = []

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "audio":
                try:
                    audio_bytes = base64.b64decode(data["data"])
                    audio_array, _ = sf.read(io.BytesIO(audio_bytes))

                    await websocket.send_json({"type": "status", "text": "Thinking..."})

                    async def send_audio_chunk(chunk_bytes):
                        audio_b64 = base64.b64encode(chunk_bytes).decode("utf-8")
                        await websocket.send_json(
                            {"type": "audio_chunk", "data": audio_b64}
                        )

                    transcript, full_reply = await orchestrator.process_audio_turn(
                        audio_array, history, send_audio_chunk
                    )

                    if transcript:
                        history.append({"role": "user", "text": transcript})
                        history.append({"role": "model", "text": full_reply})

                    await websocket.send_json({"type": "turn_complete"})

                except Exception as inner_e:
                    logger.error(f"Error processing turn: {inner_e}")
                    await websocket.send_json({"type": "turn_complete"})

    except WebSocketDisconnect:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"WebSocket Error: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
