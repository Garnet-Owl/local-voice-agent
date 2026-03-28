import asyncio
import base64
import io
import json
import queue
import re
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets
from google.genai import types

from agent.audio.interrupt_handler import InterruptHandler
from agent.audio.pre_processor import AudioPreProcessor
from agent.audio.silero_vad import NeuralVADScanner
from agent.llm.gemini_client import GeminiClient
from agent.stt.whisper_asr import WhisperAsr
from agent.tts.vits_tts import VitsTts
from shared.logging import setup_logging

logger = setup_logging(__name__)


class VoiceClientOrchestrator:
    """Orchestrates client-side audio capture, VAD, and server communication."""

    def __init__(self, cfg: dict, ws_url: str):
        self._cfg = cfg
        self._ws_url = ws_url
        self._sample_rate = cfg["audio_capture"]["sample_rate"]
        self._interrupt_handler = InterruptHandler()
        self._pre_processor = AudioPreProcessor()
        self._audio_queue = queue.Queue()
        self._is_running = True

        chunk_dur = 0.032 if self._sample_rate == 16000 else 0.016
        min_speech = max(1, int(cfg["audio_capture"]["min_speech_sec"] / chunk_dur))
        silence_timeout = max(
            1, int(cfg["audio_capture"]["silence_timeout_sec"] / chunk_dur)
        )

        self._vad = NeuralVADScanner(
            sample_rate=self._sample_rate,
            threshold=cfg["audio_capture"]["vad_threshold"],
            min_speech_chunks=min_speech,
            min_silence_chunks=silence_timeout,
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )

        self._recorded_frames = []
        self._is_recording = False

    def _on_speech_start(self):
        if self._interrupt_handler.check_for_interrupt(True):
            sd.stop()
        logger.info("Speech detected.")
        self._is_recording = True
        self._recorded_frames = []

    def _on_speech_end(self):
        logger.info("Silence detected.")
        self._is_recording = False
        if self._recorded_frames:
            raw = b"".join(self._recorded_frames)
            audio_f32 = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            processed = self._pre_processor.process(audio_f32)
            self._audio_queue.put(processed)
            self._recorded_frames = []

    def _audio_callback(self, indata, frames, time, status):
        chunk = bytes(indata)
        self._vad.ingest_audio(chunk)
        if self._is_recording:
            self._recorded_frames.append(chunk)

    async def connect_and_stream(self):
        try:
            async with websockets.connect(self._ws_url) as ws:
                logger.info("Connected to server.")
                with sd.RawInputStream(
                    samplerate=self._sample_rate,
                    blocksize=512 if self._sample_rate == 16000 else 256,
                    dtype="int16",
                    channels=1,
                    callback=self._audio_callback,
                ):
                    send_t = asyncio.create_task(self._send_loop(ws))
                    recv_t = asyncio.create_task(self._receive_loop(ws))
                    await asyncio.gather(send_t, recv_t)
        except Exception as e:
            logger.error(f"Client orchestration error: {e}")
        finally:
            self._is_running = False

    async def _send_loop(self, ws):
        while self._is_running:
            try:
                audio = await asyncio.to_thread(self._audio_queue.get, timeout=0.1)
                buf = io.BytesIO()
                sf.write(buf, audio, self._sample_rate, format="WAV")
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                await ws.send(json.dumps({"type": "audio", "data": encoded}))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")

    async def _receive_loop(self, ws):
        while self._is_running:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                if data["type"] == "audio_chunk":
                    if self._interrupt_handler.is_interrupted():
                        continue
                    self._interrupt_handler.start_agent_speech()
                    raw = base64.b64decode(data["data"])
                    arr, sr = sf.read(io.BytesIO(raw))
                    sd.play(arr, samplerate=sr)
                    await asyncio.to_thread(sd.wait)
                elif data["type"] == "turn_complete":
                    self._interrupt_handler.stop_agent_speech()
            except websockets.exceptions.ConnectionClosed:
                self._is_running = False
            except Exception as e:
                logger.error(f"Receive error: {e}")


class VoiceAgentOrchestrator:
    """Orchestrates server-side STT, LLM, and TTS pipeline."""

    def __init__(
        self,
        stt_engine: WhisperAsr,
        llm_engine: GeminiClient,
        tts_engine: VitsTts,
        vad_engine: NeuralVADScanner,
    ) -> None:
        self._stt = stt_engine
        self._llm = llm_engine
        self._tts = tts_engine
        self._vad = vad_engine

    async def process_audio_turn(
        self,
        audio_array,
        history: list,
        on_chunk_callback,
    ) -> str:
        t0 = time.perf_counter()
        transcript = await asyncio.to_thread(self._stt.transcribe, audio_array)
        t_stt = time.perf_counter() - t0

        if not transcript:
            return ""

        logger.info(f"[STT: {t_stt:.2f}s] You: {transcript}")

        history_content = [
            types.Content(role=h["role"], parts=[types.Part.from_text(text=h["text"])])
            for h in history
        ]

        tts_queue = asyncio.Queue()

        async def llm_task():
            sentence_buffer = ""
            full_reply = ""
            t0_llm = time.perf_counter()

            async for chunk in self._llm.stream_reply(transcript, history_content):
                full_reply += chunk
                sentence_buffer += chunk

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
                clean_text = sentence_buffer.strip().replace("*", "").replace("\n", " ")
                if len(clean_text) > 1:
                    await tts_queue.put(clean_text)

            await tts_queue.put(None)
            logger.info(
                f"[LLM Total: {time.perf_counter() - t0_llm:.2f}s] Agent: {full_reply}"
            )
            return full_reply

        async def tts_task():
            while True:
                text = await tts_queue.get()
                if text is None:
                    break

                t0_tts = time.perf_counter()
                audio_chunk = await asyncio.to_thread(self._tts.synthesize, text)

                buffer = io.BytesIO()
                sf.write(buffer, audio_chunk, 22050, format="WAV")
                await on_chunk_callback(buffer.getvalue())

                logger.info(
                    f"[TTS: {time.perf_counter() - t0_tts:.2f}s] Synthesized: {text}"
                )

        llm_coro = asyncio.create_task(llm_task())
        tts_coro = asyncio.create_task(tts_task())

        full_reply = await llm_coro
        await tts_coro

        return transcript, full_reply
