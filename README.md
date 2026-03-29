# Local Voice Agent

A fully local, real-time voice assistant with a client-server architecture over WebSockets.

| Layer | Engine | Details |
|-------|--------|---------|
| **STT** | [Vosk](https://alphacephei.com/vosk/) | Offline, lightweight speech recognition via `KaldiRecognizer` |
| **LLM** | [Gemini Flash](https://ai.google.dev/) | Async streaming with multi-turn chat via `google-genai` |
| **TTS** | [LuxTTS](https://github.com/ysharma3501/LuxTTS) | 48 kHz, 150× real-time on GPU, voice cloning support |
| **VAD** | [Silero VAD](https://github.com/snakers4/silero-vad) | Neural voice activity detection with barge-in support |

---

## Project Layout

```
local-voice-agent/
├── agent/
│   ├── audio/
│   │   ├── interrupt_handler.py   # Barge-in / interrupt detection
│   │   ├── post_processor.py      # TTS audio normalization + PCM encoding
│   │   ├── pre_processor.py       # Mic audio decoding + preprocessing
│   │   └── silero_vad.py          # Neural VAD engine
│   ├── connections/
│   │   └── websocket_handler.py   # Server-side WebSocket orchestration
│   ├── llm/
│   │   └── gemini_client.py       # Gemini Flash async streaming client
│   ├── stt/
│   │   └── vosk_asr.py            # Vosk offline speech recognition
│   ├── tts/
│   │   └── lux_tts.py             # LuxTTS voice synthesis engine
│   ├── orchestrator.py            # Client + server pipeline controllers
│   └── service.py                 # Engine initialization and lifecycle
├── models/
│   ├── vosk-model-en-us-0.22-lgraph/  # Vosk model (download separately)
│   └── default_voice.wav              # TTS voice reference audio
├── shared/
│   ├── config.py                  # Settings + config loading
│   └── logging.py                 # Logging setup
├── tools/
│   ├── pipeline_server.py         # FastAPI server entry point
│   ├── pipeline_client.py         # WebSocket client entry point
│   └── tts_benchmark.py           # TTS thread-count benchmarking tool
├── main.py                        # Health checks + diagnostics
├── config.yaml                    # All runtime configuration
├── pyproject.toml                 # Dependencies (managed by uv)
└── CHANGELOG.md
```

---

## Setup

### 1. Create environment and install dependencies

```bash
uv venv --python=python3.11
uv sync
```

### 2. Download the Vosk speech recognition model

```bash
curl -L -o models/vosk-model-en-us-0.22-lgraph.zip \
  https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip
cd models && unzip vosk-model-en-us-0.22-lgraph.zip && rm vosk-model-en-us-0.22-lgraph.zip
```

Other models available at [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models).

### 3. Set your Gemini API key

Create a `.env` file at the project root:

```
GEMINI_API_KEY=your_key_here
```

### 4. (Optional) Add a voice reference for TTS

LuxTTS is a voice cloning model. For best quality, provide a 3–5 second WAV
recording of clear speech:

```yaml
# config.yaml
tts:
  prompt_audio: "models/your_voice.wav"
```

A synthetic default is included at `models/default_voice.wav`, but a real
voice recording will produce significantly better output.

---

## Running

Start the server and client in separate terminals:

```bash
# Terminal 1 — Server
python tools/pipeline_server.py

# Terminal 2 — Client
python tools/pipeline_client.py
```

The server starts on `http://127.0.0.1:8000`. The client connects via
WebSocket at `ws://127.0.0.1:8000/ws`, captures your microphone, and plays
back the agent's responses in real time.

### Health checks

```bash
# From client
python tools/pipeline_client.py --health

# Direct HTTP
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/pipeline/health
```

---

## Configuration

All settings are in `config.yaml`. The API key is loaded from `.env` only.

| Setting | Default | Notes |
|---------|---------|-------|
| `stt.model_path` | `models/vosk-model-en-us-0.22-lgraph` | Path to Vosk model directory |
| `tts.model_id` | `YatharthS/LuxTTS` | HuggingFace model ID |
| `tts.device` | `cpu` | Set to `cuda` for GPU acceleration |
| `tts.speed` | `1.0` | Playback speed multiplier |
| `tts.num_steps` | `4` | Synthesis quality (3–4 recommended) |
| `tts.t_shift` | `0.5` | Sampling temperature (higher = richer but more errors) |
| `tts.thread_count` | `4` | Torch threads for CPU inference |
| `tts.prompt_audio` | `models/default_voice.wav` | Voice reference for cloning |
| `audio_capture.vad_threshold` | `0.5` | VAD sensitivity (0.0–1.0, higher = less sensitive) |
| `audio_capture.silence_timeout_sec` | `1.2` | Seconds of silence before turn ends |
| `audio_capture.min_speech_sec` | `0.5` | Minimum speech to trigger transcription |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Client (pipeline_client.py)                            │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ Mic In   │→ │ Silero   │→ │ WebSocket Send         │ │
│  │ 16 kHz   │  │ VAD      │  │ (base64 PCM chunks)    │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
│  ┌────────────────────────┐  ┌──────────┐               │
│  │ WebSocket Recv         │→ │ Speaker  │               │
│  │ (base64 PCM chunks)    │  │ 48 kHz   │               │
│  └────────────────────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────┘
                    ▲ WebSocket ▼
┌─────────────────────────────────────────────────────────┐
│  Server (pipeline_server.py)                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Vosk STT │→ │ Gemini   │→ │ LuxTTS   │              │
│  │ (16 kHz) │  │ Flash    │  │ (48 kHz) │              │
│  └──────────┘  │ Streaming│  │ Streaming│              │
│                └──────────┘  └──────────┘              │
│  LLM and TTS run concurrently via asyncio.gather       │
└─────────────────────────────────────────────────────────┘
```

---

## Benchmarking

Use the TTS benchmark tool to find the optimal thread count for your CPU:

```bash
python -m tools.tts_benchmark
```

This runs isolated and overlapped (STT + LLM + TTS concurrent) benchmarks
across multiple thread counts and recommends the best configuration.

---

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Lint and format
ruff check .
ruff check --fix .
ruff format .
```

---

## License

Apache-2.0
