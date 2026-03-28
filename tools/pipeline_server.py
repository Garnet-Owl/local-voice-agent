import asyncio
import base64
import contextlib
import io
import logging
import os
import re
import time
import warnings
from pathlib import Path

import soundfile as sf
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google.genai import types

from agent.llm.gemini_client import GeminiClient, LlmConfig
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
    global stt_engine, llm_engine, tts_engine
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

    logger.info("Server is ready and standing by.")
    yield


app = FastAPI(title="Local Voice Agent Pipeline Server", lifespan=lifespan)

stt_engine = None
llm_engine = None
tts_engine = None


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

                    t0 = time.perf_counter()
                    transcript = await asyncio.to_thread(
                        stt_engine.transcribe, audio_array
                    )
                    t_stt = time.perf_counter() - t0

                    if not transcript:
                        await websocket.send_json({"type": "turn_complete"})
                        continue

                    logger.info(f"[STT: {t_stt:.2f}s] You: {transcript}")
                    await websocket.send_json({"type": "status", "text": "Thinking..."})

                    history_content = [
                        types.Content(
                            role=h["role"], parts=[types.Part.from_text(text=h["text"])]
                        )
                        for h in history
                    ]

                    tts_queue = asyncio.Queue()

                    # TASK 1: Stream from Gemini and queue sentences
                    async def llm_task():
                        sentence_buffer = ""
                        full_reply = ""
                        t0_llm = time.perf_counter()

                        async for chunk in llm_engine.stream_reply(
                            transcript, history_content
                        ):
                            full_reply += chunk
                            sentence_buffer += chunk

                            # Split exactly AT the punctuation so we don't build huge strings
                            match = re.search(r"([.!?,;])\s*", sentence_buffer)
                            if match:
                                split_idx = match.end()
                                clean_text = (
                                    sentence_buffer[:split_idx]
                                    .strip()
                                    .replace("*", "")
                                    .replace("\n", " ")
                                )
                                if len(clean_text) > 2:
                                    await tts_queue.put(clean_text)
                                sentence_buffer = sentence_buffer[split_idx:]

                        if sentence_buffer.strip():
                            clean_text = (
                                sentence_buffer.strip()
                                .replace("*", "")
                                .replace("\n", " ")
                            )
                            if len(clean_text) > 1:
                                await tts_queue.put(clean_text)

                        await tts_queue.put(None)  # Signal TTS task to finish
                        logger.info(
                            f"[LLM Total: {time.perf_counter() - t0_llm:.2f}s] Agent: {full_reply}"
                        )
                        return full_reply

                    # TASK 2: Generate audio in the background as sentences arrive
                    async def tts_task():
                        while True:
                            text = await tts_queue.get()
                            if text is None:
                                break

                            t0_tts = time.perf_counter()
                            audio_chunk = await asyncio.to_thread(
                                tts_engine.synthesize, text
                            )
                            buffer = io.BytesIO()
                            # VITS uses 22050 Hz sample rate
                            sf.write(buffer, audio_chunk, 22050, format="WAV")
                            audio_b64 = base64.b64encode(buffer.getvalue()).decode(
                                "utf-8"
                            )

                            await websocket.send_json(
                                {"type": "audio_chunk", "data": audio_b64}
                            )
                            logger.info(
                                f"[TTS: {time.perf_counter() - t0_tts:.2f}s] Synthesized: {text}"
                            )

                    # Run both tasks at the exact same time
                    llm_coro = asyncio.create_task(llm_task())
                    tts_coro = asyncio.create_task(tts_task())

                    full_reply = await llm_coro
                    await tts_coro

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
