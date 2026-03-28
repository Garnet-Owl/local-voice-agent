import asyncio
import base64
import io
import json
import queue
import sys
import warnings

import sounddevice as sd
import soundfile as sf
import websockets

from agent.audio.interrupt_handler import InterruptHandler
from agent.audio.silero_vad import NeuralVADScanner
from agent.audio.pre_processor import AudioPreProcessor
from shared.config import load_config
from shared.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging("client")

WS_URL = "ws://127.0.0.1:8000/ws"


class VoiceClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sample_rate = cfg["audio_capture"]["sample_rate"]
        self.interrupt_handler = InterruptHandler()
        self.pre_processor = AudioPreProcessor()
        self.audio_queue = queue.Queue()
        self.is_running = True
        self.playback_finished = asyncio.Event()
        self.playback_finished.set()

        # Calculate chunks based on seconds
        # Silero window is 512 samples at 16kHz = 32ms
        chunk_duration_sec = 0.032 if self.sample_rate == 16000 else 0.016
        min_speech_chunks = max(
            1, int(cfg["audio_capture"]["min_speech_sec"] / chunk_duration_sec)
        )
        min_silence_chunks = max(
            1, int(cfg["audio_capture"]["silence_timeout_sec"] / chunk_duration_sec)
        )

        self.vad = NeuralVADScanner(
            sample_rate=self.sample_rate,
            threshold=cfg["audio_capture"]["vad_threshold"],
            min_speech_chunks=min_speech_chunks,
            min_silence_chunks=min_silence_chunks,
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )

        self.recorded_frames = []
        self.is_recording = False

    def _on_speech_start(self):
        if self.interrupt_handler.check_for_interrupt(True):
            sd.stop()
            self.playback_finished.set()

        logger.info("Speech detected.")
        self.is_recording = True
        self.recorded_frames = []

    def _on_speech_end(self):
        logger.info("Silence detected.")
        self.is_recording = False
        if self.recorded_frames:
            raw_audio = b"".join(self.recorded_frames)
            import numpy as np

            audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Apply pre-processing
            processed_audio = self.pre_processor.process(audio_float32)

            self.audio_queue.put(processed_audio)
            self.recorded_frames = []

    def _audio_callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")

        chunk = bytes(indata)
        self.vad.ingest_audio(chunk)
        if self.is_recording:
            self.recorded_frames.append(chunk)

    async def run(self):
        try:
            async with websockets.connect(WS_URL) as ws:
                logger.info("Connected to server. Speak to interact.")

                with sd.RawInputStream(
                    samplerate=self.sample_rate,
                    blocksize=512 if self.sample_rate == 16000 else 256,
                    dtype="int16",
                    channels=1,
                    callback=self._audio_callback,
                ):
                    send_task = asyncio.create_task(self._send_audio(ws))
                    recv_task = asyncio.create_task(self._receive_audio(ws))
                    await asyncio.gather(send_task, recv_task)

        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            self.is_running = False

    async def _send_audio(self, ws):
        while self.is_running:
            try:
                processed_audio = await asyncio.to_thread(
                    self.audio_queue.get, timeout=0.1
                )

                buffer = io.BytesIO()
                sf.write(buffer, processed_audio, self.sample_rate, format="WAV")
                audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                await ws.send(json.dumps({"type": "audio", "data": audio_b64}))
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error sending audio: {e}")

    async def _receive_audio(self, ws):
        while self.is_running:
            try:
                msg = await ws.recv()
                data = json.loads(msg)

                if data["type"] == "audio_chunk":
                    if self.interrupt_handler.is_interrupted():
                        continue

                    self.interrupt_handler.start_agent_speech()
                    self.playback_finished.clear()

                    audio_bytes = base64.b64decode(data["data"])
                    chunk_array, sr = sf.read(io.BytesIO(audio_bytes))

                    sd.play(chunk_array, samplerate=sr)
                    await asyncio.to_thread(sd.wait)

                elif data["type"] == "turn_complete":
                    self.interrupt_handler.stop_agent_speech()
                    self.playback_finished.set()

            except Exception as e:
                logger.error(f"Error receiving audio: {e}")
                break


async def main():
    cfg = load_config()
    client = VoiceClient(cfg)
    await client.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
