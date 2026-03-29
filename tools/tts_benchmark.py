import argparse
import asyncio
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from agent.stt.whisper_asr import SttConfig, WhisperAsr
from agent.tts.kokoro_tts import TtsConfig, KokoroTts

DEFAULT_TEXT = (
    "Thanks for waiting. I can help with that right now, and I will keep this short."
)


def default_thread_candidates(logical_cpus: int | None) -> list[int]:
    if logical_cpus is None or logical_cpus < 2:
        return [2, 3, 4]

    candidates = {
        max(2, logical_cpus // 2),
        max(2, round(logical_cpus * 0.67)),
        max(2, logical_cpus - 2),
    }
    return sorted(thread for thread in candidates if thread <= logical_cpus)


def load_local_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def stream_tts_metrics(tts: KokoroTts, text: str) -> dict:
    first_chunk_time = None
    chunk_count = 0
    total_samples = 0
    start = time.perf_counter()

    for chunk in tts.stream(text):
        now = time.perf_counter()
        if first_chunk_time is None:
            first_chunk_time = now - start
        chunk_count += 1
        total_samples += len(chunk)

    total_time = time.perf_counter() - start
    return {
        "first_chunk_ms": round((first_chunk_time or total_time) * 1000, 2),
        "total_synth_ms": round(total_time * 1000, 2),
        "chunk_count": chunk_count,
        "total_samples": total_samples,
    }


def synthesize_stt_input(tts: KokoroTts, text: str) -> np.ndarray:
    audio = tts.synthesize(text)
    sample_points = np.linspace(0, len(audio) - 1, 16000, dtype=np.int32)
    return audio[sample_points].astype(np.float32)


def simulate_llm_load(text: str, passes: int = 2500) -> int:
    sentence_buffer = ""
    fragments = re.findall(r".{1,18}", text * 6)
    produced = 0

    for _ in range(passes):
        for fragment in fragments:
            sentence_buffer += fragment
            match = re.search(r"([.!?])\s+|([,;:])\s+(?=.{{8,}})", sentence_buffer)
            if match:
                produced += len(sentence_buffer[: match.end()].strip())
                sentence_buffer = sentence_buffer[match.end() :]

    produced += len(sentence_buffer.strip())
    return produced


async def overlap_metrics(tts: KokoroTts, stt: WhisperAsr, text: str) -> dict:
    stt_audio = await asyncio.to_thread(synthesize_stt_input, tts, text)

    start = time.perf_counter()
    tts_task = asyncio.to_thread(stream_tts_metrics, tts, text)
    stt_task = asyncio.to_thread(stt.transcribe, stt_audio)
    llm_task = asyncio.to_thread(simulate_llm_load, text)

    tts_result, transcript, llm_units = await asyncio.gather(
        tts_task,
        stt_task,
        llm_task,
    )

    return {
        "tts_first_chunk_ms": tts_result["first_chunk_ms"],
        "tts_total_ms": tts_result["total_synth_ms"],
        "stt_llm_overlap_ms": round((time.perf_counter() - start) * 1000, 2),
        "stt_transcript_chars": len(transcript),
        "llm_work_units": llm_units,
    }


async def benchmark_thread_count(
    cfg: dict,
    threads: int,
    text: str,
    runs: int,
) -> dict:
    tts = KokoroTts(
        TtsConfig(
            model_id=cfg["tts"]["model_id"],
            device=cfg["tts"]["device"],
            speed=cfg["tts"].get("speed", 1.0),
            thread_count=threads,
        )
    )
    tts._ensure_loaded()

    stt = WhisperAsr(
        SttConfig(
            model_id=cfg["stt"]["model_id"],
            device=cfg["stt"]["device"],
        )
    )
    stt._ensure_loaded()

    isolated_runs = []
    overlap_runs = []

    for _ in range(runs):
        isolated_runs.append(await asyncio.to_thread(stream_tts_metrics, tts, text))
        overlap_runs.append(await overlap_metrics(tts, stt, text))

    return {
        "thread_count": threads,
        "isolated": {
            "first_chunk_ms": round(
                statistics.mean(run["first_chunk_ms"] for run in isolated_runs), 2
            ),
            "total_synth_ms": round(
                statistics.mean(run["total_synth_ms"] for run in isolated_runs), 2
            ),
        },
        "overlap": {
            "tts_first_chunk_ms": round(
                statistics.mean(run["tts_first_chunk_ms"] for run in overlap_runs), 2
            ),
            "tts_total_ms": round(
                statistics.mean(run["tts_total_ms"] for run in overlap_runs), 2
            ),
            "stt_llm_overlap_ms": round(
                statistics.mean(run["stt_llm_overlap_ms"] for run in overlap_runs), 2
            ),
        },
    }


def recommend_result(results: list[dict]) -> dict:
    return min(
        results,
        key=lambda result: (
            result["overlap"]["tts_first_chunk_ms"],
            result["overlap"]["tts_total_ms"],
            result["isolated"]["first_chunk_ms"],
        ),
    )


def run_worker_process(
    cfg_path: Path,
    thread_count: int,
    text: str,
    runs: int,
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    try:
        command = [
            sys.executable,
            "-m",
            "tools.tts_benchmark",
            "--worker-thread-count",
            str(thread_count),
            "--runs",
            str(runs),
            "--text",
            text,
            "--config-path",
            str(cfg_path),
            "--output-path",
            str(output_path),
        ]
        project_root = Path(__file__).resolve().parent.parent
        subprocess.run(command, check=True, cwd=project_root)
        return json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        output_path.unlink(missing_ok=True)


async def main() -> None:
    logical_cpus = os.cpu_count()
    default_config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        nargs="+",
        type=int,
        default=default_thread_candidates(logical_cpus),
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--config-path", type=Path, default=default_config_path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--worker-thread-count", type=int)
    args = parser.parse_args()

    cfg = load_local_config(args.config_path)

    if args.worker_thread_count is not None:
        result = await benchmark_thread_count(
            cfg=cfg,
            threads=args.worker_thread_count,
            text=args.text,
            runs=args.runs,
        )
        payload = json.dumps(result, indent=2)
        if args.output_path:
            args.output_path.write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    results = []

    for thread_count in args.threads:
        results.append(
            await asyncio.to_thread(
                run_worker_process,
                args.config_path,
                thread_count,
                args.text,
                args.runs,
            )
        )

    recommendation = recommend_result(results)
    output = {
        "logical_cpu_count": logical_cpus,
        "thread_candidates": args.threads,
        "results": results,
        "recommended_thread_count": recommendation["thread_count"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
