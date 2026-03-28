import sys
import warnings

# Suppress pkg_resources deprecation warning from webrtcvad
warnings.filterwarnings("ignore", category=UserWarning)

import asyncio
import websockets
import json
import base64
import io
import soundfile as sf
import sounddevice as sd
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent))
from agent.audio_capture.vad_recorder import record_utterance, CaptureConfig

WS_URL = "ws://127.0.0.1:8000/ws"


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


async def run_client():
    cfg = load_config()
    capture_cfg = CaptureConfig(
        sample_rate=cfg["audio_capture"]["sample_rate"],
        vad_aggressiveness=cfg["audio_capture"]["vad_aggressiveness"],
        silence_timeout_sec=cfg["audio_capture"]["silence_timeout_sec"],
        min_speech_sec=cfg["audio_capture"]["min_speech_sec"],
        frame_duration_ms=cfg["audio_capture"]["frame_duration_ms"],
    )

    print("\n  Pipeline Client ready. Transcripts will show on the SERVER terminal.")
    print("  Speak after the prompt. Press Ctrl+C to quit.\n")

    try:
        async with websockets.connect(WS_URL) as ws:
            while True:
                # 1. Record audio (blocks until silence is detected)
                audio_data = await asyncio.to_thread(record_utterance, capture_cfg)
                if audio_data is None:
                    continue

                # 2. Send audio to server
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, capture_cfg.sample_rate, format="WAV")
                audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

                await ws.send(json.dumps({"type": "audio", "data": audio_b64}))

                # 3. Process incoming responses (play audio chunks)
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if data["type"] == "audio_chunk":
                        # Play audio sentence chunks as they stream
                        audio_bytes = base64.b64decode(data["data"])
                        chunk_array, sr = sf.read(io.BytesIO(audio_bytes))
                        sd.play(chunk_array, samplerate=sr)
                        sd.wait()  # Wait for the chunk to finish before playing the next

                    elif data["type"] == "turn_complete":
                        break  # Turn is over, go back to listening to the mic

    except websockets.exceptions.ConnectionClosedError:
        print("\nConnection to server lost.")
    except KeyboardInterrupt:
        print("\nStopping client...")
    except Exception as e:
        print(f"\nClient error: {e}")


if __name__ == "__main__":
    asyncio.run(run_client())
