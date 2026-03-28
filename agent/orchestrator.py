import asyncio
import io
import logging
import re
import time

import soundfile as sf
from google.genai import types

from agent.llm.gemini_client import GeminiClient
from agent.stt.whisper_asr import WhisperAsr
from agent.tts.vits_tts import VitsTts

logger = logging.getLogger(__name__)


class VoiceAgentOrchestrator:
    def __init__(
        self,
        stt_engine: WhisperAsr,
        llm_engine: GeminiClient,
        tts_engine: VitsTts,
    ) -> None:
        self._stt = stt_engine
        self._llm = llm_engine
        self._tts = tts_engine

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
