"""Prepare, score, inspect and verify the dedicated 5m54s audio/video case."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from asr_common.paths import PROJECT_ROOT, VIDEO_DIR, PODCAST_INPUT_DIR


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_saved_results() -> list[dict]:
    """Read-only integration check; never invokes a model or rewrites a report."""
    import jiwer
    from asr_common.text import normalize_vi

    folder = VIDEO_DIR / "two_model_results"
    dataset = json.loads((PODCAST_INPUT_DIR / "evaluation_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    reference_path = VIDEO_DIR / "reference_subtitle.corrected.txt"
    if _digest(VIDEO_DIR / "source_hls.wav") != dataset["source_sha256"]:
        raise ValueError("Source WAV differs from the evaluated file")
    if _digest(PODCAST_INPUT_DIR / "audio_16k.flac") != dataset["audio_sha256"]:
        raise ValueError("Prepared audio differs from the evaluated waveform")
    if _digest(reference_path) != report["reference_sha256"]:
        raise ValueError("Reference changed; run score to produce a new report")
    reference = normalize_vi(reference_path.read_text(encoding="utf-8-sig"))
    if len(reference.split()) != report["reference_words"]:
        raise ValueError("Reference word count differs")
    keys = {"kyle/vi-asr-fastconformer-114m": "kyle", "vinai/PhoWhisper-large": "phowhisper"}
    hashes, verified = None, []
    for result in report["models"]:
        key = keys[result["model"]]
        rows = json.loads((folder / key / "predictions.json").read_text(encoding="utf-8"))
        if len(rows) != len(dataset["chunks"]):
            raise ValueError(f"Missing chunks: {key}")
        for row, expected in zip(rows, dataset["chunks"]):
            if any(row[field] != expected[field] for field in ["id", "start_sample", "end_sample"]):
                raise ValueError(f"Audio span mismatch: {key}")
        actual_hashes = [row["audio_sha256"] for row in rows]
        if hashes is not None and actual_hashes != hashes:
            raise ValueError("Models used different audio chunks")
        hashes = actual_hashes
        prediction = normalize_vi(" ".join(row["prediction"] for row in rows))
        wer, cer = 100 * jiwer.wer(reference, prediction), 100 * jiwer.cer(reference, prediction)
        if abs(wer - result["wer_percent"]) > 1e-9 or abs(cer - result["cer_percent"]) > 1e-9:
            raise ValueError(f"Stored scores differ from transcripts: {key}")
        verified.append({"model": result["model"], "wer_percent": wer, "cer_percent": cer})
    if len(verified) != 2:
        raise ValueError("Expected two completed original models")
    return verified


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Clean the original subtitle and prepare the common audio input")
    commands.add_parser("revise-reference", help="Regenerate the ASR-assisted subtitle draft from independent evidence")
    commands.add_parser("score", help="Rescore existing candidate transcripts against the frozen corrected reference")
    commands.add_parser("results", help="Display the saved comparison without running inference")
    commands.add_parser("verify", help="Verify audio/reference hashes and recalculate saved WER/CER without writing files")
    local = commands.add_parser("run-local", help="Run the original Kyle checkpoint locally; requires a working CUDA runtime")
    local.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from evaluation.audio_video.prepare_input import main as prepare
        prepare()
    elif args.command == "revise-reference":
        from evaluation.audio_video.revise_reference import main as revise
        revise()
    elif args.command == "score":
        from evaluation.audio_video.score import main as score
        score()
    elif args.command == "results":
        path = VIDEO_DIR / "two_model_results" / "comparison.md"
        print(path.read_text(encoding="utf-8"))
    elif args.command == "verify":
        print(json.dumps({"status": "verified", "models": verify_saved_results()}, ensure_ascii=False, indent=2))
    else:
        subprocess.run([sys.executable, "-m", "evaluation.audio_video.kaggle.evaluate", "--local", "--model", "kyle",
                        "--kyle-checkpoint", str(args.checkpoint.resolve())], cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
