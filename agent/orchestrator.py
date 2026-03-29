import asyncio
import json
import queue
import re
import time

import numpy as np
import sounddevice as sd
import websockets
from agent.audio.interrupt_handler import InterruptHandler
from agent.audio.pre_processor import AudioPreProcessor
from agent.audio.post_processor import AudioPostProcessor
from agent.audio.silero_vad import NeuralVADScanner
from agent.llm.gemini_client import GeminiClient
from agent.stt.vosk_asr import VoskAsr
from agent.tts.lux_tts import LUX_SAMPLE_RATE, LuxTts
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
        self._post_processor = AudioPostProcessor()
        self._audio_queue = queue.Queue()
        self._playback_queue = queue.Queue()
        self._is_running = True
        self._output_sample_rate = LUX_SAMPLE_RATE
        self._playback_remainder = np.array([], dtype=np.float32)

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

        self._is_recording = False

    def _on_speech_start(self):
        if self._interrupt_handler.check_for_interrupt(True):
            with self._playback_queue.mutex:
                self._playback_queue.queue.clear()
                self._playback_remainder = np.array([], dtype=np.float32)
        logger.info("Speech detected.")
        self._is_recording = True
        self._audio_queue.put(
            {"type": "input_audio_start", "sample_rate": self._sample_rate}
        )

    def _on_speech_end(self):
        logger.info("Silence detected.")
        self._is_recording = False
        self._audio_queue.put({"type": "input_audio_end"})

    def _audio_callback(self, indata, frames, time, status):
        chunk = bytes(indata)
        self._vad.ingest_audio(chunk)
        if self._is_recording:
            audio_f32 = self._pre_processor.decode_pcm_bytes(chunk)
            processed = self._pre_processor.process(audio_f32)
            self._audio_queue.put({"type": "input_audio_chunk", "data": processed})

    def _playback_callback(self, outdata, frames, time_info, status):
        if status:
            logger.warning(f"Playback status: {status}")

        requested = frames
        blocks = []

        if len(self._playback_remainder) > 0:
            if len(self._playback_remainder) <= requested:
                blocks.append(self._playback_remainder)
                requested -= len(self._playback_remainder)
                self._playback_remainder = np.array([], dtype=np.float32)
            else:
                outdata[:, 0] = self._playback_remainder[:requested]
                self._playback_remainder = self._playback_remainder[requested:]
                return

        while requested > 0:
            try:
                chunk = self._playback_queue.get_nowait()
            except queue.Empty:
                blocks.append(np.zeros(requested, dtype=np.float32))
                break

            if len(chunk) <= requested:
                blocks.append(chunk)
                requested -= len(chunk)
                continue

            blocks.append(chunk[:requested])
            self._playback_remainder = chunk[requested:]
            requested = 0

        mixed = np.concatenate(blocks) if blocks else np.zeros(frames, dtype=np.float32)
        outdata[:, 0] = mixed[:frames]

    async def connect_and_stream(self):
        try:
            async with websockets.connect(self._ws_url) as ws:
                logger.info("Connected to server.")
                with (
                    sd.RawInputStream(
                        samplerate=self._sample_rate,
                        blocksize=512 if self._sample_rate == 16000 else 256,
                        dtype="int16",
                        channels=1,
                        callback=self._audio_callback,
                    ),
                    sd.OutputStream(
                        samplerate=self._output_sample_rate,
                        blocksize=1024,
                        dtype="float32",
                        channels=1,
                        callback=self._playback_callback,
                    ),
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
                message = await asyncio.to_thread(self._audio_queue.get, timeout=0.1)
                msg_type = message["type"]
                if msg_type == "input_audio_chunk":
                    encoded = self._post_processor.encode_transport(message["data"])
                    await ws.send(
                        json.dumps({"type": "input_audio_chunk", "data": encoded})
                    )
                elif msg_type == "input_audio_start":
                    await ws.send(json.dumps(message))
                elif msg_type == "input_audio_end":
                    await ws.send(json.dumps(message))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")

    async def _receive_loop(self, ws):
        while self._is_running:
            try:
                msg = await ws.recv()
                data = json.loads(msg)
                if data["type"] == "audio_format":
                    self._output_sample_rate = data["sample_rate"]
                elif data["type"] == "partial_transcript":
                    logger.info(f"[STT Partial] {data['text']}")
                elif data["type"] == "final_transcript":
                    logger.info(f"[STT Final] {data['text']}")
                elif data["type"] == "audio_chunk":
                    if self._interrupt_handler.is_interrupted():
                        continue
                    self._interrupt_handler.start_agent_speech()
                    chunk = self._pre_processor.decode_transport(data["data"])
                    self._playback_queue.put(chunk)
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
        stt_engine: VoskAsr,
        llm_engine: GeminiClient,
        tts_engine: LuxTts,
        vad_engine: NeuralVADScanner,
    ) -> None:
        self._stt = stt_engine
        self._llm = llm_engine
        self._tts = tts_engine
        self._vad = vad_engine
        self._pre_processor = AudioPreProcessor()
        self._post_processor = AudioPostProcessor()

    async def process_audio_turn(
        self,
        audio_array,
        chat,
        on_chunk_callback,
    ) -> str:
        transcript = await self.transcribe_audio(audio_array)
        if not transcript:
            return ""

        await self.process_text_turn(transcript, chat, on_chunk_callback)
        return transcript

    async def transcribe_audio(self, audio_array) -> str:
        t0 = time.perf_counter()
        transcript = await asyncio.to_thread(self._stt.transcribe, audio_array)
        t_stt = time.perf_counter() - t0

        if not transcript:
            return ""

        logger.info(f"[STT: {t_stt:.2f}s] You: {transcript}")
        return transcript

    async def process_text_turn(
        self,
        transcript: str,
        chat,
        on_chunk_callback,
    ) -> None:
        if not transcript:
            return

        await self._stream_llm_and_tts(transcript, chat, on_chunk_callback)

    async def _stream_llm_and_tts(self, transcript, chat, on_chunk_callback) -> None:
        sentence_queue = asyncio.Queue()
        min_phrase_chars = 28
        max_phrase_chars = 110

        def find_split_index(buffer: str) -> int | None:
            terminal_match = re.search(r"[.!?]\s+", buffer)
            if terminal_match and terminal_match.end() >= min_phrase_chars:
                return terminal_match.end()

            clause_match = re.search(r"[,;:]\s+", buffer)
            if clause_match and clause_match.end() >= min_phrase_chars:
                return clause_match.end()

            if len(buffer) < max_phrase_chars:
                return None

            whitespace_idx = buffer.rfind(" ", 0, max_phrase_chars)
            if whitespace_idx >= min_phrase_chars:
                return whitespace_idx + 1

            return max_phrase_chars

        async def llm_task():
            sentence_buffer = ""
            full_reply = ""
            t0_llm = time.perf_counter()

            async for chunk in self._llm.stream_reply(transcript, chat):
                full_reply += chunk
                sentence_buffer += chunk
                while True:
                    split_idx = find_split_index(sentence_buffer)
                    if split_idx is None:
                        break
                    sentence = (
                        sentence_buffer[:split_idx]
                        .strip()
                        .replace("*", "")
                        .replace("\n", " ")
                    )
                    if len(sentence) > 2:
                        await sentence_queue.put(sentence)
                    sentence_buffer = sentence_buffer[split_idx:]

            if sentence_buffer.strip():
                await sentence_queue.put(
                    sentence_buffer.strip().replace("*", "").replace("\n", " ")
                )

            await sentence_queue.put(None)
            logger.info(
                f"[LLM Total: {time.perf_counter() - t0_llm:.2f}s] Agent: {full_reply}"
            )

        async def tts_task():
            while True:
                sentence = await sentence_queue.get()
                if sentence is None:
                    break

                t0_tts = time.perf_counter()
                loop = asyncio.get_running_loop()

                def generate_and_send():
                    for audio_chunk in self._tts.stream(sentence):
                        refined = self._post_processor.process(audio_chunk)
                        if refined is not None and len(refined) > 0:
                            payload = self._post_processor.encode_transport(refined)
                            asyncio.run_coroutine_threadsafe(
                                on_chunk_callback(payload), loop
                            )

                await asyncio.to_thread(generate_and_send)

                logger.info(
                    f"[TTS: {time.perf_counter() - t0_tts:.2f}s] Synthesized: {sentence}"
                )

        await asyncio.gather(
            asyncio.create_task(llm_task()),
            asyncio.create_task(tts_task()),
        )
