import contextlib
import os
import warnings

import uvicorn
from fastapi import FastAPI, WebSocket

from agent.connections.websocket_handler import WebSocketHandler
from agent.llm.gemini_client import GeminiClient, LlmConfig
from agent.orchestrator import VoiceAgentOrchestrator
from agent.stt.whisper_asr import SttConfig, WhisperAsr
from agent.tts.vits_tts import TtsConfig, VitsTts
from shared.config import load_config
from shared.logging import setup_logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

logger = setup_logging("server")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global ws_handler
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
    ws_handler = WebSocketHandler(orchestrator)

    logger.info("Server is ready and standing by.")
    yield


app = FastAPI(title="Local Voice Agent Pipeline Server", lifespan=lifespan)
ws_handler = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if ws_handler:
        await ws_handler.handle_connection(websocket)
    else:
        await websocket.close(code=1001)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
