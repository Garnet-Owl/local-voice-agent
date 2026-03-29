import asyncio

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from agent.audio.pre_processor import AudioPreProcessor
from agent.orchestrator import VoiceAgentOrchestrator
from agent.tts.kokoro_tts import KOKORO_SAMPLE_RATE
from shared.logging import setup_logging

logger = setup_logging(__name__)


class StreamingInputState:
    def __init__(self) -> None:
        self.sample_rate = 16_000
        self.active = False
        self.chunks: list[np.ndarray] = []
        self.partial_task = None
        self.last_partial = ""
        self.lock = None
        self.websocket = None


class WebSocketHandler:
    """Handles WebSocket communication and turn orchestration."""

    def __init__(self, orchestrator: VoiceAgentOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._pre_processor = AudioPreProcessor()
        self._partial_interval_sec = 0.35
        self._partial_window_sec = 6.0

    async def handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "audio_format",
                "sample_rate": KOKORO_SAMPLE_RATE,
                "encoding": "pcm_s16le",
            }
        )
        logger.info("Client connected.")
        chat = self._orchestrator._llm.new_chat()
        stream_state = StreamingInputState()
        stream_state.lock = asyncio.Lock()
        stream_state.websocket = websocket

        try:
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type")
                if message_type == "input_audio_start":
                    await self._on_audio_start(stream_state, data)
                elif message_type == "input_audio_chunk":
                    await self._on_audio_chunk(stream_state, data)
                elif message_type == "input_audio_end":
                    await self._on_audio_end(websocket, stream_state, chat)
                elif message_type == "audio":
                    await self._process_audio_message(websocket, data, chat)

        except WebSocketDisconnect:
            await self._cancel_partial_task(stream_state)
            logger.info("Client disconnected.")
        except Exception as e:
            await self._cancel_partial_task(stream_state)
            logger.error(f"WebSocket Error: {e}")

    async def _on_audio_start(self, state: StreamingInputState, data: dict) -> None:
        state.sample_rate = int(data.get("sample_rate", 16_000))
        await self._cancel_partial_task(state)
        async with state.lock:
            state.chunks.clear()
            state.last_partial = ""
            state.active = True
        state.partial_task = asyncio.create_task(self._partial_stt_loop(state))

    async def _on_audio_chunk(self, state: StreamingInputState, data: dict) -> None:
        if not state.active:
            return
        chunk = self._pre_processor.decode_transport(data["data"])
        if len(chunk) == 0:
            return
        async with state.lock:
            state.chunks.append(chunk)

    async def _on_audio_end(
        self,
        websocket: WebSocket,
        state: StreamingInputState,
        chat,
    ) -> None:
        async with state.lock:
            state.active = False
        await self._cancel_partial_task(state)

        async with state.lock:
            if not state.chunks:
                await websocket.send_json({"type": "turn_complete"})
                return
            audio_array = np.concatenate(state.chunks)
            state.chunks.clear()

        transcript = await self._orchestrator.transcribe_audio(audio_array)
        if not transcript:
            await websocket.send_json({"type": "turn_complete"})
            return

        await websocket.send_json({"type": "final_transcript", "text": transcript})
        await websocket.send_json({"type": "status", "text": "Thinking..."})

        async def send_audio_chunk(chunk_pcm_b64):
            await websocket.send_json({"type": "audio_chunk", "data": chunk_pcm_b64})

        await self._orchestrator.process_text_turn(transcript, chat, send_audio_chunk)
        await websocket.send_json({"type": "turn_complete"})

    async def _partial_stt_loop(self, state: StreamingInputState) -> None:
        while True:
            await asyncio.sleep(self._partial_interval_sec)
            async with state.lock:
                if not state.active:
                    break
                if not state.chunks:
                    continue
                audio_array = np.concatenate(state.chunks)

            max_samples = int(state.sample_rate * self._partial_window_sec)
            if len(audio_array) > max_samples:
                audio_array = audio_array[-max_samples:]

            partial = await asyncio.to_thread(
                self._orchestrator._stt.transcribe,
                audio_array,
            )
            if partial and partial != state.last_partial:
                state.last_partial = partial
                logger.info(f"[STT Partial] {partial}")
                await state.websocket.send_json(
                    {"type": "partial_transcript", "text": partial}
                )

    async def _cancel_partial_task(self, state: StreamingInputState) -> None:
        if state.partial_task is None:
            return
        state.partial_task.cancel()
        try:
            await state.partial_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        state.partial_task = None

    async def _process_audio_message(
        self, websocket: WebSocket, data: dict, chat
    ) -> None:
        try:
            audio_array = self._pre_processor.decode_transport(data["data"])

            await websocket.send_json({"type": "status", "text": "Thinking..."})

            async def send_audio_chunk(chunk_pcm_b64):
                await websocket.send_json(
                    {"type": "audio_chunk", "data": chunk_pcm_b64}
                )

            await self._orchestrator.process_audio_turn(
                audio_array, chat, send_audio_chunk
            )

            await websocket.send_json({"type": "turn_complete"})

        except Exception as e:
            logger.error(f"Error processing audio turn: {e}")
            await websocket.send_json({"type": "turn_complete"})
