# local-voice-agent

A fully local voice agent combining:

- **STT**: `microsoft/VibeVoice-ASR-HF` (Transformers >= 5.3.0)
- **Brain**: Gemini Flash streaming (multi-turn conversation history)
- **TTS**: `OpenMOSS-Team/MOSS-TTS-Realtime` (editable install from MOSS-TTS repo)
- **VAD**: `webrtcvad` — auto-detects speech from your microphone

> **CPU-only warning**: Both STT and TTS models are large (~0.5-1.7B params).
> On CPU each turn takes 10-60+ seconds. A CUDA GPU is needed for real-time speed.
> When you have one, set `device_map: "auto"` and `device: "cuda"` in `config.yaml`.

---

## Project Layout

```
local-voice-agent/
├── agent/
│   ├── __init__.py
│   ├── __main__.py               # Entry point: python -m agent
│   ├── orchestrator.py           # Pipeline controller (mic -> STT -> LLM -> TTS)
│   ├── audio_capture/
│   │   ├── __init__.py
│   │   └── vad_recorder.py       # VAD mic capture -> numpy utterance
│   ├── stt/
│   │   ├── __init__.py
│   │   └── vibevoice_asr.py      # VibeVoice-ASR-HF wrapper
│   ├── llm/
│   │   ├── __init__.py
│   │   └── gemini_client.py      # Gemini Flash streaming, multi-turn history
│   └── tts/
│       ├── __init__.py
│       └── moss_tts.py           # MOSS-TTS-Realtime streaming playback
├── audio/
│   └── prompt.wav                # (you provide this) voice reference for TTS
├── config.yaml
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

---

## Setup (one-time)

### 1. Install MOSS-TTS (must be done before pip install -r requirements.txt)

```bash
cd C:\Users\Marcus\Desktop\personal-dev-ops
git clone https://github.com/OpenMOSS/MOSS-TTS.git
cd MOSS-TTS
pip install -e .
```

### 2. Install project dependencies

```bash
cd C:\Users\Marcus\Desktop\personal-dev-ops\local-voice-agent
pip install -r requirements.txt
```

### 3. Set your Gemini API key

```bash
# Windows CMD
set GEMINI_API_KEY=your_key_here

# PowerShell
$env:GEMINI_API_KEY="your_key_here"
```

### 4. Add a voice prompt wav

MOSS-TTS-Realtime requires a short reference audio clip (~5-10 seconds of
clear speech, 24 kHz wav) to establish the voice style for synthesis.

Place it at:

```
local-voice-agent\audio\prompt.wav
```

Any clear recording of someone speaking works. You can record one with Audacity
or download a sample speech wav and resample to 24 kHz.

### 5. Run

```bash
cd C:\Users\Marcus\Desktop\personal-dev-ops\local-voice-agent
python -m agent
```

---

## Configuration

All settings are in `config.yaml` — model IDs, VAD sensitivity, silence timeout,
and device selection. No secrets belong in config; the API key comes from the
environment variable only.

Key settings to tune:

| Setting | Default | Notes |
|---|---|---|
| `audio_capture.vad_aggressiveness` | `2` | 0-3, raise if noisy environment |
| `audio_capture.silence_timeout_sec` | `1.2` | Lower for faster response trigger |
| `stt.device_map` | `"cpu"` | Change to `"auto"` for GPU |
| `tts.device` | `"cpu"` | Change to `"cuda"` for GPU |


## Development Workflow

For local development without Docker, you can use `uv` for environment and dependency management.

1.  **Install uv** (if not already installed)
    ```bash
    pip install uv
    ```

2.  **Create a virtual environment with Python 3.12**
    ```bash
    uv venv --python=python3.11
    ```

3.  **Activate the virtual environment**

    Windows:
    ```bash
    .venv\Scripts\activate
    ```
    Linux/macOS:
    ```bash
    source .venv/bin/activate
    ```
4.  **Install dependencies with uv**
    ```bash
    uv sync
    ```

5.  **Install the pre-commit hooks**
    ```bash
    pre-commit install
    ```

6.  **Running the Application Locally**
    To run the development server locally:
    ```bash
    uvicorn app.main:app --reload
    ```

7.  **Using Ruff for linting and formatting**
    ```bash
    # Run linting
    ruff check .

    # Apply automatic fixes to linting issues
    ruff check --fix .

    # Format code
    ruff format .
    ```

## License

Free
