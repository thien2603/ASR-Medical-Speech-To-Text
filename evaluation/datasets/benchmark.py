"""Reproducible, resumable comparison of three original ASR models on VietMed.

prepare: freeze/verify the same 3,437 test clips and reference normalization.
run: run one model in a separate process, appending completed batches to JSONL.
report: publish measured corpus WER/CER only for complete, matching runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from asr_common.text import normalize_vi, text_from_hypothesis

from asr_common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUTPUT = ROOT / "results" / "medical-local"
RUNTIME = ROOT / ".benchmark_runtime"
from asr_common.paths import EXTERNAL_RUNTIME

OLD_RUNTIME = EXTERNAL_RUNTIME
MODEL_IDS = {
    "kyle": "kyle/vi-asr-fastconformer-114m",
    "phowhisper": "vinai/PhoWhisper-large",
    "whisper": "openai/whisper-large-v3",
}
NORMALIZATION = "NFC; lowercase; punctuation/symbols to space; collapse whitespace; keep digits and diacritics"
_DLL_HANDLES = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    # A killed process may leave an incomplete final line. Refuse corruption
    # rather than silently hiding a missing/misaligned test sample.
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prepare() -> None:
    import math
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parquet = ROOT / "local_decoding_fix" / "test.parquet"
    audio_dir = OLD_RUNTIME / "local_decoding_fix" / "test_audio"
    fallback_dir = RUNTIME / "audio"
    manifest = []
    parquet_file = pq.ParquetFile(parquet)
    for batch in parquet_file.iter_batches(batch_size=64, columns=["audio", "text", "duration"]):
        for item in batch.to_pylist():
            index = len(manifest)
            encoded = item["audio"]["bytes"]
            suffix = Path(item["audio"].get("path") or "audio.ogg").suffix or ".ogg"
            filename = f"{index:08d}{suffix.lower()}"
            path = audio_dir / filename
            digest = hashlib.sha256(encoded).hexdigest()
            if not path.exists():
                fallback_dir.mkdir(parents=True, exist_ok=True)
                path = fallback_dir / filename
                if not path.exists():
                    path.write_bytes(encoded)
            if sha256(path) != digest:
                raise RuntimeError(f"Audio mismatch at test row {index}: {path}")
            # The source VietMed clips are 8 kHz. Materialize one shared 16 kHz
            # PCM input so every backend receives exactly the same waveform.
            audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if original_rate != 16000:
                divisor = math.gcd(original_rate, 16000)
                audio = resample_poly(audio, 16000 // divisor, original_rate // divisor).astype(np.float32)
            pcm_dir = RUNTIME / "audio_16k"
            pcm_dir.mkdir(parents=True, exist_ok=True)
            pcm_path = pcm_dir / f"{index:08d}.wav"
            sf.write(pcm_path, audio, 16000, subtype="PCM_16")
            info = sf.info(pcm_path)
            reference = normalize_vi(item["text"])
            if not reference:
                raise RuntimeError(f"Empty reference at test row {index}")
            manifest.append({
                "id": f"vietmed-test-{index:05d}", "row_index": index,
                "audio_path": str(pcm_path), "audio_sha256": sha256(pcm_path),
                "source_audio_sha256": digest, "source_sample_rate": original_rate,
                "duration_seconds": info.frames / info.samplerate,
                "reference_raw": item["text"], "reference": reference,
            })
    if len(manifest) != 3437:
        raise RuntimeError(f"Expected 3437 test clips, got {len(manifest)}")
    manifest_file = OUTPUT / "manifest.jsonl"
    contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest)
    if manifest_file.exists() and manifest_file.read_text(encoding="utf-8") != contents:
        raise RuntimeError("Refusing to replace a different frozen test manifest")
    manifest_file.write_text(contents, encoding="utf-8")
    write_json(OUTPUT / "dataset.json", {
        "dataset": "leduckhai/VietMed", "split": "test", "samples": len(manifest),
        "source": str(parquet), "parquet_sha256": sha256(parquet),
        "manifest_sha256": sha256(manifest_file),
        "audio_seconds": sum(row["duration_seconds"] for row in manifest),
        "normalization": NORMALIZATION, "selection": "all rows in original parquet order",
        "audio_preprocessing": "source mono mean; scipy.signal.resample_poly to 16 kHz; shared PCM16 WAV",
    })
    print(f"Verified and froze {len(manifest)} test clips", flush=True)


def metrics(rows: list[dict]) -> dict:
    import jiwer
    references = [row["reference"] for row in rows]
    predictions = [row["prediction_normalized"] for row in rows]
    word = jiwer.process_words(references, predictions)
    char = jiwer.process_characters(references, predictions)
    return {
        "wer": word.wer, "cer": char.cer,
        "word_substitutions": word.substitutions, "word_deletions": word.deletions,
        "word_insertions": word.insertions, "word_hits": word.hits,
        "reference_words": word.hits + word.substitutions + word.deletions,
        "exact_match_rate": sum(a == b for a, b in zip(references, predictions)) / len(rows),
    }


def package_versions() -> dict:
    versions = {}
    for name in ["torch", "nemo_toolkit", "transformers", "ctranslate2", "faster-whisper", "jiwer", "soundfile", "pyarrow"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def load_kyle(device: str):
    import scipy.signal  # noqa: F401
    import torch
    from nemo.collections.asr.models import ASRModel
    torch.set_num_threads(2)
    hub = OLD_RUNTIME / "cache" / "huggingface" / "hub" / "models--kyle--vi-asr-fastconformer-114m"
    revision = (hub / "refs" / "main").read_text().strip()
    checkpoint = (hub / "snapshots" / revision / "vi-asr-fastconformer-114m.nemo").resolve()
    model = ASRModel.restore_from(str(checkpoint), map_location="cpu")
    model.eval()
    if device == "cuda":
        model.half().cuda()
    from omegaconf import OmegaConf
    metadata = {
        "source_revision": revision, "checkpoint_sha256": sha256(checkpoint),
        "precision": "float16" if device == "cuda" else "float32", "backend": "NeMo RNNT",
        "decoding": OmegaConf.to_container(model.cfg.decoding, resolve=True),
        "decoding_policy": "original checkpoint configuration; no test-driven changes",
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
    }
    return model, metadata


def load_whisper(key: str, device: str):
    # CT2 loads CUDA DLLs lazily. Use existing CUDA/cuDNN from the user's torch
    # installation without importing torch and consuming another CUDA context.
    dll_dir = OLD_RUNTIME / ".venv" / "Lib" / "site-packages" / "torch" / "lib"
    os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ["PATH"]
    if os.name == "nt":
        _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
    from faster_whisper import WhisperModel
    path = RUNTIME / "models" / key
    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
    if provenance["source_model"] != MODEL_IDS[key]:
        raise RuntimeError("Converted model source does not match the requested model")
    model = WhisperModel(str(path), device=device, compute_type="int8_float16" if device == "cuda" else "int8", cpu_threads=2, num_workers=1)
    return model, {
        **provenance, "precision": "int8_float16" if device == "cuda" else "int8",
        "backend": "faster-whisper / CTranslate2",
        "decoding": {"language": "vi", "task": "transcribe", "beam_size": 1,
                     "temperature": 0.0, "condition_on_previous_text": False,
                     "vad_filter": False, "initial_prompt": None,
                     "no_speech_threshold": None, "log_prob_threshold": None,
                     "compression_ratio_threshold": None},
    }


def run(args) -> None:
    manifest_file = OUTPUT / "manifest.jsonl"
    manifest = read_jsonl(manifest_file)
    if len(manifest) != 3437:
        raise RuntimeError("Run prepare first; a verified full-test manifest is required")
    selected = manifest[:args.limit] if args.limit else manifest
    run_dir = (OUTPUT / "smoke" if args.limit else OUTPUT) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {"model_id": MODEL_IDS[args.model], "manifest_sha256": sha256(manifest_file),
              "expected_samples": len(selected), "device": args.device,
              "batch_size": args.batch_size if args.model == "kyle" else 1,
              "normalization": NORMALIZATION, "protocol_version": 1}
    config_file = run_dir / "config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        raise RuntimeError("Existing run uses another protocol; choose a separate output")
    write_json(config_file, config)
    prediction_file = run_dir / "predictions.jsonl"
    completed = read_jsonl(prediction_file)
    if len(completed) > len(selected):
        raise RuntimeError("More predictions than selected samples")
    for actual, expected in zip(completed, selected):
        if actual["id"] != expected["id"] or actual["reference"] != expected["reference"] or actual["audio_sha256"] != expected["audio_sha256"]:
            raise RuntimeError("Resume mismatch: predictions must be an exact prefix of the manifest")
    model = None
    model_metadata = {}
    try:
        if len(completed) < len(selected):
            started_load = time.perf_counter()
            model, model_metadata = load_kyle(args.device) if args.model == "kyle" else load_whisper(args.model, args.device)
            session_metadata = {**model_metadata, "python": sys.version, "platform": platform.platform(),
                                "packages": package_versions(), "loaded_seconds": time.perf_counter() - started_load,
                                "started_utc": datetime.now(timezone.utc).isoformat()}
            with (run_dir / "sessions.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(session_metadata, ensure_ascii=False) + "\n")
            import soundfile as sf
            if args.model == "kyle":
                import torch
            batch_size = config["batch_size"]
            with prediction_file.open("a", encoding="utf-8") as handle:
                for start in range(len(completed), len(selected), batch_size):
                    batch = selected[start:start + batch_size]
                    if args.model != "kyle":
                        audio, sample_rate = sf.read(batch[0]["audio_path"], dtype="float32")
                        if sample_rate != 16000 or audio.ndim != 1:
                            raise RuntimeError("Audio format changed since prepare")
                    if args.model == "kyle" and args.device == "cuda":
                        torch.cuda.synchronize()
                    started = time.perf_counter()
                    if args.model == "kyle":
                        with torch.inference_mode():
                            output = model.transcribe([row["audio_path"] for row in batch], batch_size=len(batch), verbose=False, return_hypotheses=False)
                        if isinstance(output, tuple):
                            output = output[0]
                        hypotheses = [text_from_hypothesis(value) for value in output]
                        if args.device == "cuda":
                            torch.cuda.synchronize()
                    else:
                        segments, _ = model.transcribe(audio, **model_metadata["decoding"])
                        hypotheses = [" ".join(segment.text.strip() for segment in segments).strip()]
                    elapsed = time.perf_counter() - started
                    if len(hypotheses) != len(batch):
                        raise RuntimeError("Hypothesis count mismatch")
                    for row, hypothesis in zip(batch, hypotheses):
                        prediction = {**row, "prediction": hypothesis,
                                      "prediction_normalized": normalize_vi(hypothesis),
                                      "batch_inference_seconds_share": elapsed / len(batch)}
                        handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                        completed.append(prediction)
                    handle.flush()
                    if start == 0 or len(completed) % 20 < batch_size or len(completed) == len(selected):
                        seconds = sum(row["batch_inference_seconds_share"] for row in completed)
                        print(f"{args.model}: {len(completed)}/{len(selected)}; inference {seconds:.1f}s", flush=True)
        sessions = read_jsonl(run_dir / "sessions.jsonl")
        inference_seconds = sum(row["batch_inference_seconds_share"] for row in completed)
        audio_seconds = sum(row["duration_seconds"] for row in completed)
        summary = {**config, "status": "complete", "samples": len(completed),
                   **metrics(completed), "audio_seconds": audio_seconds,
                   "inference_seconds": inference_seconds, "rtf": inference_seconds / audio_seconds,
                   "timing_note": "sum of actual batch wall times; model loading excluded; Kyle includes native audio IO; Whisper excludes external sf.read",
                   "model_metadata": sessions[-1] if sessions else model_metadata}
        write_json(run_dir / "summary.json", summary)
        with (run_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            fields = ["id", "row_index", "audio_sha256", "duration_seconds", "reference_raw", "reference", "prediction", "prediction_normalized"]
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(completed)
        print(json.dumps({k: v for k, v in summary.items() if k != "model_metadata"}, indent=2), flush=True)
    except Exception as error:
        write_json(run_dir / "failure.json", {"status": "incomplete", "completed_samples": len(completed),
                                             "error_type": type(error).__name__, "error": str(error)})
        raise


def report() -> None:
    dataset = json.loads((OUTPUT / "dataset.json").read_text(encoding="utf-8"))
    manifest = read_jsonl(OUTPUT / "manifest.jsonl")
    rows = []
    for key, model_id in MODEL_IDS.items():
        path = OUTPUT / key / "summary.json"
        row = {"model": model_id, "status": "pending", "samples": 0, "wer_percent": None, "cer_percent": None, "rtf": None, "precision": None}
        if path.exists():
            summary = json.loads(path.read_text(encoding="utf-8"))
            predictions = read_jsonl(OUTPUT / key / "predictions.jsonl")
            if summary["manifest_sha256"] != dataset["manifest_sha256"] or summary["samples"] != dataset["samples"] or len(predictions) != len(manifest):
                raise RuntimeError(f"Incomplete or mismatched run: {key}")
            for prediction, reference in zip(predictions, manifest):
                if any(prediction[field] != reference[field] for field in ["id", "audio_sha256", "reference"]):
                    raise RuntimeError(f"Prediction alignment mismatch: {key}")
            recalculated = metrics(predictions)
            if any(abs(recalculated[field] - summary[field]) > 1e-12 for field in ["wer", "cer"]):
                raise RuntimeError(f"Saved metric does not match predictions: {key}")
            row.update(status="complete", samples=summary["samples"], wer_percent=100 * summary["wer"], cer_percent=100 * summary["cer"], rtf=summary["rtf"], precision=summary["model_metadata"]["precision"])
        else:
            completed = read_jsonl(OUTPUT / key / "predictions.jsonl")
            failure = OUTPUT / key / "failure.json"
            if failure.exists():
                row["status"] = "incomplete"
                row["reason"] = json.loads(failure.read_text(encoding="utf-8"))["error"]
            elif completed:
                row["status"] = "running"
            row["samples"] = len(completed)
        rows.append(row)
    write_json(OUTPUT / "comparison.json", {"dataset": dataset, "models": rows})
    with (OUTPUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "status", "samples", "wer_percent", "cer_percent", "rtf", "precision"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# So sánh ASR trên VietMed test", "", f"Cùng {dataset['samples']:,} mẫu, {dataset['audio_seconds']/3600:.3f} giờ. Manifest SHA-256: `{dataset['manifest_sha256']}`.", "", "| Model | Trạng thái | Số mẫu | WER (%) | CER (%) | RTF | Precision |", "|---|---|---:|---:|---:|---:|---|"]
    for row in rows:
        values = [f"{row['wer_percent']:.2f}" if row["wer_percent"] is not None else "—", f"{row['cer_percent']:.2f}" if row["cer_percent"] is not None else "—", f"{row['rtf']:.4f}" if row["rtf"] is not None else "—"]
        lines.append(f"| {row['model']} | {row['status']} | {row['samples']} | {' | '.join(values)} | {row['precision'] or '—'} |")
    lines += ["", "WER/CER là tỷ lệ lỗi toàn corpus, không phải trung bình WER từng câu. Giữ nguyên dấu tiếng Việt và chữ số.", "", "Hai model Whisper dùng INT8/FP16 trên CTranslate2 để phù hợp GPU 4 GB; Kyle dùng NeMo. RTF phản ánh cả backend, precision và batch size, không chỉ kiến trúc. Chưa đo recall thuật ngữ y khoa do chưa có danh sách thuật ngữ được kiểm chứng.", "", "Bản Kyle fine-tune/sửa decoding được phân tích riêng; không trộn với checkpoint gốc trong bảng chính.", ""]
    (OUTPUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", choices=MODEL_IDS, required=True)
    run_parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--limit", type=int, help="Separate smoke run; never published in the full comparison")
    sub.add_parser("report")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run":
        if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
            parser.error("batch-size and limit must be positive")
        run(args)
    else:
        report()


if __name__ == "__main__":
    main()
