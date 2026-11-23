"""Evaluate datasets using a pretrained NeMo model or a fine-tuned .nemo checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from asr_common.text import normalize_vi, text_from_hypothesis
from asr_common.paths import RESULTS_DIR

MODEL_ID = "kyle/vi-asr-fastconformer-114m"


@dataclass(frozen=True)
class DatasetSpec:
    repo: str
    config: str | None
    split: str
    text_column: str


DATASETS = {
    # General-domain, studio read speech.
    "vivos": DatasetSpec("AILAB-VNUHCM/vivos", None, "test", "sentence"),
    # General-domain, international read speech.
    "fleurs": DatasetSpec("google/fleurs", "vi_vn", "test", "transcription"),
    # Medical-domain speech; the most important probe for this project.
    "vietmed": DatasetSpec("leduckhai/VietMed", None, "test", "text"),
}


def iter_samples(name: str, limit: int, seed: int) -> Iterable[dict[str, Any]]:
    from datasets import Audio, load_dataset

    spec = DATASETS[name]
    ds = load_dataset(
        spec.repo,
        spec.config,
        split=spec.split,
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=max(1_000, limit * 20))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    for index, row in enumerate(ds.take(limit)):
        yield {
            "id": row.get("utterance_id") or row.get("id") or f"{name}-{index:05d}",
            "audio": row["audio"],
            "reference": row[spec.text_column],
        }


def evaluate_dataset(model: Any, name: str, limit: int, seed: int, batch_size: int) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf
    from jiwer import cer, wer

    rows = list(iter_samples(name, limit, seed))
    references: list[str] = []
    predictions: list[str] = []
    details: list[dict[str, Any]] = []
    total_audio_seconds = 0.0

    with tempfile.TemporaryDirectory(prefix=f"vi-asr-{name}-") as temp_dir:
        paths: list[str] = []
        for index, row in enumerate(rows):
            audio = row["audio"]
            array = np.asarray(audio["array"], dtype=np.float32)
            if array.ndim > 1:
                array = array.mean(axis=0)
            path = Path(temp_dir) / f"{index:05d}.wav"
            sf.write(path, array, 16_000, subtype="PCM_16")
            paths.append(str(path))
            total_audio_seconds += len(array) / 16_000

        started = time.perf_counter()
        output = model.transcribe(paths, batch_size=batch_size)
        elapsed = time.perf_counter() - started

    # NeMo versions may return either a list or (hypotheses, all_hypotheses).
    if isinstance(output, tuple):
        output = output[0]
    hypotheses = [text_from_hypothesis(item) for item in output]
    if len(hypotheses) != len(rows):
        raise RuntimeError(f"Expected {len(rows)} hypotheses, received {len(hypotheses)}")

    for row, hypothesis in zip(rows, hypotheses):
        ref = normalize_vi(row["reference"])
        hyp = normalize_vi(hypothesis)
        references.append(ref)
        predictions.append(hyp)
        details.append(
            {
                "dataset": name,
                "id": row["id"],
                "reference": row["reference"],
                "prediction": hypothesis,
                "reference_normalized": ref,
                "prediction_normalized": hyp,
                "wer": wer(ref, hyp),
                "cer": cer(ref, hyp),
            }
        )

    return {
        "dataset": name,
        "samples": len(rows),
        "audio_seconds": round(total_audio_seconds, 3),
        "inference_seconds": round(elapsed, 3),
        "real_time_factor": round(elapsed / total_audio_seconds, 4) if total_audio_seconds else None,
        "wer": wer(references, predictions),
        "cer": cer(references, predictions),
        "details": details,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-id", default=MODEL_ID, help="Hugging Face pretrained NeMo model")
    source.add_argument("--checkpoint", type=Path, help="Fine-tuned .nemo checkpoint; evaluation only")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--samples", type=int, default=50, help="Samples per dataset")
    parser.add_argument("--batch-size", type=int, default=1, help="Keep at 1 for a 4 GB GPU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be at least 1")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    import torch
    import nemo.collections.asr as nemo_asr

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    model_source = str(args.checkpoint.resolve()) if args.checkpoint else args.model_id
    print(f"Loading {model_source} on {device} ...")
    if args.checkpoint:
        model = nemo_asr.models.ASRModel.restore_from(model_source, map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_source, map_location=device)
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_details = []
    for name in args.datasets:
        print(f"Evaluating {name}: {args.samples} samples ...")
        result = evaluate_dataset(model, name, args.samples, args.seed, args.batch_size)
        all_details.extend(result.pop("details"))
        results.append(result)
        print(f"  WER={result['wer']:.2%} CER={result['cer']:.2%} RTF={result['real_time_factor']}")

    metadata = {
        "model": model_source,
        "model_source": "checkpoint" if args.checkpoint else "pretrained",
        "device": device,
        "seed": args.seed,
        "normalization": "NFC + lowercase + remove punctuation/symbols + collapse whitespace; keep digits/diacritics",
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_details[0]) if all_details else [])
        if all_details:
            writer.writeheader()
            writer.writerows(all_details)

    print(f"Saved results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
