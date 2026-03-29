import base64
import io

import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

from agent.orchestrator import VoiceAgentOrchestrator
from agent.audio.post_processor import AudioPostProcessor
from shared.logging import setup_logging

logger = setup_logging(__name__)


class WebSocketHandler:
    """Handles WebSocket communication and turn orchestration."""

    def __init__(self, orchestrator: VoiceAgentOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._post_processor = AudioPostProcessor()

    async def handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("Client connected.")
        chat = self._orchestrator._llm.new_chat()

        try:
            while True:
                data = await websocket.receive_json()

                if data.get("type") == "audio":
                    await self._process_audio_message(websocket, data, chat)

        except WebSocketDisconnect:
            logger.info("Client disconnected.")
        except Exception as e:
            logger.error(f"WebSocket Error: {e}")

    async def _process_audio_message(
        self, websocket: WebSocket, data: dict, chat
    ) -> None:
        try:
            audio_bytes = base64.b64decode(data["data"])
            audio_array, _ = sf.read(io.BytesIO(audio_bytes))

            await websocket.send_json({"type": "status", "text": "Thinking..."})

            async def send_audio_chunk(chunk_bytes):
                chunk_array, sr = sf.read(io.BytesIO(chunk_bytes))
                processed_chunk = self._post_processor.process(chunk_array)

                buffer = io.BytesIO()
                sf.write(buffer, processed_chunk, sr, format="WAV")
                processed_bytes = buffer.getvalue()

                audio_b64 = base64.b64encode(processed_bytes).decode("utf-8")
                await websocket.send_json({"type": "audio_chunk", "data": audio_b64})

            await self._orchestrator.process_audio_turn(
                audio_array, chat, send_audio_chunk
            )

            await websocket.send_json({"type": "turn_complete"})

        except Exception as e:
            logger.error(f"Error processing audio turn: {e}")
            await websocket.send_json({"type": "turn_complete"})
