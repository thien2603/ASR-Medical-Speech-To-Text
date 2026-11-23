"""Transcribe a long Vietnamese video with the locally fixed FastConformer model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

# Importing scipy.signal first avoids a Windows DLL/page-file loading issue seen
# when torch is imported before SciPy in this runtime.
import scipy.signal  # noqa: F401
import numpy as np
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict


def hypothesis_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "text"):
        return str(value.text)
    if isinstance(value, (tuple, list)) and value:
        return hypothesis_text(value[0])
    return str(value)


def timestamp(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def quiet_boundaries(
    audio: np.ndarray,
    sample_rate: int,
    max_seconds: float = 24.0,
    search_seconds: float = 6.0,
    min_seconds: float = 14.0,
) -> list[tuple[int, int]]:
    """Split near quiet points, keeping every chunk short enough for a 4 GB GPU."""
    total = len(audio)
    maximum = int(max_seconds * sample_rate)
    search = int(search_seconds * sample_rate)
    minimum = int(min_seconds * sample_rate)
    window = max(1, int(0.30 * sample_rate))
    hop = max(1, int(0.05 * sample_rate))
    segments: list[tuple[int, int]] = []
    start = 0

    while total - start > maximum:
        target = start + maximum
        low = max(start + minimum, target - search)
        high = max(low + 1, target - window)
        best_position = target
        best_energy = float("inf")
        for position in range(low, high, hop):
            frame = audio[position : position + window]
            energy = float(np.mean(np.square(frame, dtype=np.float64)))
            if energy < best_energy:
                best_energy = energy
                best_position = position + window // 2
        segments.append((start, best_position))
        start = best_position

    if start < total:
        segments.append((start, total))
    return segments


def set_safe_decoding(model: ASRModel) -> dict[str, Any]:
    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.greedy.max_symbols_per_step = 1
        cfg.greedy.loop_labels = False
        cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg, verbose=False)
    return OmegaConf.to_container(cfg, resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--title", default="YouTube video")
    parser.add_argument("--url", default="")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz")

    segments = quiet_boundaries(audio, sample_rate)
    print(f"Audio: {len(audio) / sample_rate:.2f}s; segments: {len(segments)}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this local transcription run")

    model = ASRModel.restore_from(str(args.model), map_location="cpu")
    decoding = set_safe_decoding(model)
    model.eval().half().cuda()
    torch.cuda.empty_cache()

    temporary_chunk = args.output_dir / "_current_chunk.wav"
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index, (start, end) in enumerate(segments, start=1):
            sf.write(temporary_chunk, audio[start:end], sample_rate, subtype="PCM_16")
            output = model.transcribe(
                [str(temporary_chunk)],
                batch_size=1,
                verbose=False,
                return_hypotheses=False,
            )
            if isinstance(output, tuple):
                output = output[0]
            text = hypothesis_text(output[0]).strip() if output else ""
            row = {
                "segment": index,
                "start_seconds": round(start / sample_rate, 3),
                "end_seconds": round(end / sample_rate, 3),
                "start_time": timestamp(start / sample_rate),
                "end_time": timestamp(end / sample_rate),
                "text": text,
            }
            rows.append(row)
            print(
                f"[{index:02d}/{len(segments):02d}] "
                f"{row['start_time']}-{row['end_time']}: {text}"
            )

    elapsed = time.perf_counter() - started
    temporary_chunk.unlink(missing_ok=True)

    csv_path = args.output_dir / "transcript_segments.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    txt_path = args.output_dir / "transcript.txt"
    with txt_path.open("w", encoding="utf-8-sig") as handle:
        handle.write(f"Tiêu đề: {args.title}\n")
        if args.url:
            handle.write(f"Nguồn: {args.url}\n")
        handle.write(
            f"Audio: {len(audio) / sample_rate:.2f} giây | "
            f"Xử lý: {elapsed:.2f} giây | "
            f"Thiết bị: {torch.cuda.get_device_name(0)}\n\n"
        )
        for row in rows:
            handle.write(
                f"[{row['start_time']} - {row['end_time']}] {row['text']}\n\n"
            )

    report = {
        "status": "complete",
        "title": args.title,
        "url": args.url,
        "audio_seconds": len(audio) / sample_rate,
        "segments": len(rows),
        "inference_seconds": elapsed,
        "rtf": elapsed / (len(audio) / sample_rate),
        "device": torch.cuda.get_device_name(0),
        "model": str(args.model),
        "decoding": decoding,
        "transcript_txt": str(txt_path),
        "segments_csv": str(csv_path),
    }
    report_path = args.output_dir / "transcription_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
