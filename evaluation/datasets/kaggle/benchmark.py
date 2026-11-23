"""Three original checkpoints, one full VietMed test, same T4/FP16/batch=4.

Private Kaggle job. Public benchmark audio only; no training and no test tuning.
The pretrained Kyle checkpoint is reused from the project's previous output.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

# This benchmark only uses PyTorch. Prevent optional TensorFlow imports from
# coupling Transformers to the unrelated system TensorFlow/protobuf versions.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Inputs are verified local WAVs/manifests. Avoid eagerly importing optional
# cloud transports (and their unrelated Google protobuf generated code).
os.environ["LHOTSE_IO_BACKEND"] = "BuiltinIOBackend"
os.environ["LHOTSE_AUDIO_BACKEND"] = "LibsndfileBackend"

OUT = Path("/kaggle/working/medical_benchmark")
INPUT = Path("/kaggle/input")
MODEL_IDS = {"kyle": "kyle/vi-asr-fastconformer-114m", "phowhisper": "vinai/PhoWhisper-large", "whisper": "openai/whisper-large-v3"}
REVISIONS = {"kyle": "3f498775970e051972b791b4428f5ad3a13a53fc", "phowhisper": "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb", "whisper": "06f233fe06e710322aca913c1bc4249a0d71fce1"}
KYLE_SHA = "b4813d92fccc47a686b8185c6706d73f6f0a4458fab73820a8341b4efe905034"
KYLE_STATE_SHA = "e30c130f25e3ab092d0d7b1a09ba45629ad5e66add405f194e496216c9946383"
REFERENCE_SHA = "0882a921c4afb6bac3cdd4063664c42019ac0bddcf328498b53205f359d3498a"
PARQUET_SHA = "e36c65d77b365fb725610ac77c9634ba20847156beecdc665447060336d24a25"
BATCH = 4
# A fresh reproduction runs all three models. Archived Kaggle versions 3/4
# repaired Kyle separately; version-2 Whisper runs were retained unchanged.
RUN_MODELS = tuple(MODEL_IDS)


def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare():
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    from scipy.signal import resample_poly
    OUT.mkdir(parents=True, exist_ok=True)
    audio_dir = OUT / "audio"
    audio_dir.mkdir(exist_ok=True)
    candidates = sorted(p for p in INPUT.rglob("*.parquet") if "test" in p.name.lower())
    matched = [p for p in candidates if digest(p) == PARQUET_SHA]
    rows = []
    if len(matched) == 1:
        source = str(matched[0])
        mode = "original parquet; scipy resample_poly; shared PCM16 mono 16kHz"
        for batch in pq.ParquetFile(matched[0]).iter_batches(batch_size=64, columns=["audio", "text"]):
            for item in batch.to_pylist():
                data, sr = sf.read(io.BytesIO(item["audio"]["bytes"]), dtype="float32", always_2d=True)
                data = data.mean(axis=1)
                if sr != 16000:
                    gcd = math.gcd(sr, 16000)
                    data = resample_poly(data, 16000 // gcd, sr // gcd).astype(np.float32)
                path = audio_dir / f"{len(rows):08d}.wav"
                sf.write(path, data, 16000, subtype="PCM_16")
                rows.append({"id": f"vietmed-test-{len(rows):05d}", "audio_path": str(path),
                             "audio_sha256": digest(path), "duration": len(data) / 16000,
                             "reference_raw": item["text"], "reference": normalize(item["text"])})
    else:
        manifests = sorted(INPUT.rglob("test_manifest.jsonl"))
        if len(manifests) != 1:
            raise RuntimeError(f"Cannot uniquely locate exact test data: parquets={candidates}; manifests={manifests}")
        source = str(manifests[0])
        mode = "reuse prior shared PCM16 mono 16kHz test preparation (librosa soxr_hq)"
        for index, item in enumerate(read_rows(manifests[0])):
            path = manifests[0].parent / "test" / Path(item["audio_filepath"]).name
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1:
                raise RuntimeError("Expected mono 16kHz prepared input")
            rows.append({"id": f"vietmed-test-{index:05d}", "audio_path": str(path),
                         "audio_sha256": digest(path), "duration": info.frames / 16000,
                         "reference_raw": item["text"], "reference": normalize(item["text"])})
    reference_sha = hashlib.sha256(("\n".join(row["reference"] for row in rows) + "\n").encode()).hexdigest()
    if len(rows) != 3437 or reference_sha != REFERENCE_SHA:
        raise RuntimeError(f"Test differs from locally audited test: n={len(rows)}, refs={reference_sha}")
    manifest = OUT / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    save(OUT / "dataset.json", {"dataset": "leduckhai/VietMed", "split": "test", "samples": len(rows),
                               "audio_seconds": sum(row["duration"] for row in rows), "manifest_sha256": digest(manifest),
                               "reference_sha256": reference_sha, "source": source, "audio_preprocessing": mode,
                               "normalization": "NFC + lowercase + punctuation/symbols to space + whitespace collapse; keep digits/diacritics"})
    print("DATASET", json.dumps(json.loads((OUT / "dataset.json").read_text())), flush=True)


def evaluate(key):
    import csv
    import importlib.metadata as metadata
    import platform
    import jiwer
    import numpy as np
    import soundfile as sf
    import torch
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    rows = read_rows(OUT / "manifest.jsonl")
    run_dir = OUT / key
    run_dir.mkdir(exist_ok=True)
    config = {"model_id": MODEL_IDS[key], "revision": REVISIONS[key], "precision": "float16",
              "batch_size": BATCH, "gpu": torch.cuda.get_device_name(0), "seed": 42,
              "manifest_sha256": digest(OUT / "manifest.jsonl"), "platform": platform.platform(),
              "packages": {name: metadata.version(name) for name in ["torch", "transformers", "nemo_toolkit", "jiwer", "scipy", "soundfile"]}}
    started_load = time.perf_counter()
    if key == "kyle":
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf
        candidates = list(INPUT.rglob("vi-asr-fastconformer-114m-pretrained.nemo"))
        if len(candidates) != 1:
            raise RuntimeError(f"Missing uniquely verified original Kyle checkpoint: {candidates}")
        # NeMo save_to reserializes an archive, changing its byte hash. Check
        # actual tensor names/dtypes/shapes/bytes against the original HF file.
        model = ASRModel.restore_from(str(candidates[0]), map_location="cpu")
        state_hash = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            state_hash.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode() + b"\n")
            state_hash.update(tensor.detach().contiguous().numpy().tobytes())
        fingerprint = state_hash.hexdigest()
        if fingerprint != KYLE_STATE_SHA:
            raise RuntimeError(f"Kyle tensor fingerprint differs from original checkpoint: {fingerprint}")
        if model.cfg.decoding.strategy != "greedy_batch" or model.cfg.decoding.greedy.max_symbols_per_step != 10:
            raise RuntimeError("Kyle decoding differs from original checkpoint")
        model.eval().half().cuda()
        config.update(backend="NeMo", checkpoint_sha256=digest(candidates[0]),
                      original_hf_archive_sha256=KYLE_SHA, checkpoint_tensor_sha256=fingerprint,
                      decoding=OmegaConf.to_container(model.cfg.decoding, resolve=True))

        def transcribe(batch):
            output = model.transcribe([row["audio_path"] for row in batch], batch_size=len(batch), num_workers=0, verbose=False, return_hypotheses=False)
            if isinstance(output, tuple):
                output = output[0]
            return [value if isinstance(value, str) else str(value.text) for value in output]
    else:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        processor = AutoProcessor.from_pretrained(MODEL_IDS[key], revision=REVISIONS[key])
        model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_IDS[key], revision=REVISIONS[key],
                    torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
        decode_kwargs = {"language": "vi", "task": "transcribe", "do_sample": False, "num_beams": 1,
                         "max_new_tokens": 440, "return_timestamps": False, "condition_on_prev_tokens": False}
        config.update(backend="Transformers SDPA", decoding=decode_kwargs,
                      generation_config=model.generation_config.to_dict(),
                      long_audio="non-overlapping <=30s chunks; concatenate in source order; no reference-dependent boundaries")

        def transcribe(batch):
            chunks, owners = [], []
            for index, row in enumerate(batch):
                audio, sr = sf.read(row["audio_path"], dtype="float32")
                if sr != 16000 or audio.ndim != 1:
                    raise RuntimeError("Unexpected input format")
                for start in range(0, len(audio), 30 * 16000):
                    chunks.append(audio[start:start + 30 * 16000])
                    owners.append(index)
            decoded = []
            for start in range(0, len(chunks), BATCH):
                features = processor(chunks[start:start + BATCH], sampling_rate=16000,
                                     return_tensors="pt", return_attention_mask=True)
                generated = model.generate(input_features=features.input_features.to("cuda", torch.float16),
                                           attention_mask=features.attention_mask.to("cuda"), **decode_kwargs)
                decoded.extend(processor.batch_decode(generated, skip_special_tokens=True))
            answers = [[] for _ in batch]
            for owner, text in zip(owners, decoded):
                answers[owner].append(text.strip())
            return [" ".join(parts).strip() for parts in answers]

    config["load_seconds"] = time.perf_counter() - started_load
    save(run_dir / "config.json", config)
    predictions, elapsed = [], 0.0
    with torch.inference_mode():
        transcribe(rows[:1])  # warm-up only; no scoring/configuration selection
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for start in range(0, len(rows), BATCH):
                batch = rows[start:start + BATCH]
                torch.cuda.synchronize()
                tick = time.perf_counter()
                hypotheses = transcribe(batch)
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - tick
                if len(hypotheses) != len(batch):
                    raise RuntimeError("Prediction count mismatch")
                for row, hypothesis in zip(batch, hypotheses):
                    prediction = {**row, "prediction": hypothesis, "prediction_normalized": normalize(hypothesis)}
                    predictions.append(prediction)
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                handle.flush()
                if start == 0 or len(predictions) % 100 < BATCH or len(predictions) == len(rows):
                    print(f"PROGRESS {key} {len(predictions)}/{len(rows)} inference_seconds={elapsed:.2f}", flush=True)
    refs = [row["reference"] for row in predictions]
    hyps = [row["prediction_normalized"] for row in predictions]
    word, char = jiwer.process_words(refs, hyps), jiwer.process_characters(refs, hyps)
    summary = {**config, "status": "complete", "samples": len(rows), "wer": word.wer, "cer": char.cer,
               "word_substitutions": word.substitutions, "word_deletions": word.deletions,
               "word_insertions": word.insertions, "word_hits": word.hits,
               "reference_words": word.hits + word.substitutions + word.deletions,
               "exact_match_rate": float(np.mean([r == h for r, h in zip(refs, hyps)])),
               "audio_seconds": sum(row["duration"] for row in rows), "inference_seconds": elapsed,
               "rtf": elapsed / sum(row["duration"] for row in rows),
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
               "timing_note": "synchronized batch wall time including audio IO, preprocessing and decoding; excludes model load and one warm-up clip"}
    save(run_dir / "summary.json", summary)
    with (run_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print("RESULT", key, json.dumps(summary, ensure_ascii=False), flush=True)


def report():
    import csv
    dataset = json.loads((OUT / "dataset.json").read_text())
    manifest = read_rows(OUT / "manifest.jsonl")
    rows = []
    for key, model_id in MODEL_IDS.items():
        path = OUT / key / "summary.json"
        row = {"model": model_id, "status": "pending", "samples": 0, "wer_percent": None, "cer_percent": None, "rtf": None, "gpu_peak_allocated_gib": None}
        if path.exists():
            s = json.loads(path.read_text())
            pred = read_rows(OUT / key / "predictions.jsonl")
            if s["samples"] != 3437 or s["manifest_sha256"] != dataset["manifest_sha256"] or len(pred) != len(manifest):
                raise RuntimeError("Incomplete/mismatched run")
            if any(any(a[f] != b[f] for f in ["id", "audio_sha256", "reference"]) for a, b in zip(pred, manifest)):
                raise RuntimeError("Prediction alignment differs between models")
            row.update(status="complete", samples=s["samples"], wer_percent=s["wer"] * 100,
                       cer_percent=s["cer"] * 100, rtf=s["rtf"], gpu_peak_allocated_gib=s["gpu_peak_allocated_gib"])
        rows.append(row)
    save(OUT / "comparison.json", {"dataset": dataset, "models": rows})
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# So sánh ASR trên VietMed test", "", "Cùng 3.437 mẫu / 6,022 giờ; cùng GPU, FP16, batch size 4; không fine-tune trên VietMed.", "", "| Model | Trạng thái | WER (%) | CER (%) | RTF | Peak GPU GiB |", "|---|---|---:|---:|---:|---:|"]
    for row in rows:
        fmt = lambda value, digits: f"{value:.{digits}f}" if value is not None else "—"
        lines.append(f"| {row['model']} | {row['status']} | {fmt(row['wer_percent'],2)} | {fmt(row['cer_percent'],2)} | {fmt(row['rtf'],4)} | {fmt(row['gpu_peak_allocated_gib'],2)} |")
    lines += ["", "WER/CER tính trên toàn corpus; giữ chữ số và dấu tiếng Việt. RTF gồm IO, tiền xử lý và giải mã, bỏ load model và warm-up. Kyle dùng NeMo/decoding gốc; hai Whisper dùng Transformers SDPA/greedy, không prompt hoặc VAD. Với Whisper, audio >30s được chia đoạn liên tiếp để không mất phần cuối.", "", "Đây là benchmark phiên âm miền y khoa; chưa chấm riêng recall tên thuốc/liều lượng/phủ định. Kết quả Kyle fine-tune + adaptive decoding 26,91% được trình bày riêng vì đã dùng test khi chọn cấu hình.", ""]
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def diagnose_whisper():
    """Check three fixed clips with an official pipeline/eager attention.

    This checks implementation correctness, not decoding hyperparameter search.
    Full-test version-2 scores are retained until this check is understood.
    """
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
    processor = AutoProcessor.from_pretrained(MODEL_IDS["whisper"], revision=REVISIONS["whisper"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_IDS["whisper"], revision=REVISIONS["whisper"],
                torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").eval().cuda()
    kwargs = {"language": "vi", "task": "transcribe", "do_sample": False, "num_beams": 1,
              "max_new_tokens": 440, "return_timestamps": False, "condition_on_prev_tokens": False}
    transcriber = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer,
                           feature_extractor=processor.feature_extractor, device=0)
    results = []
    with torch.inference_mode():
        for row in read_rows(OUT / "manifest.jsonl")[:3]:
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            features = processor(audio, sampling_rate=sr, return_tensors="pt", return_attention_mask=True)
            generated = model.generate(input_features=features.input_features.to("cuda", torch.float16),
                                       attention_mask=features.attention_mask.to("cuda"), **kwargs)
            direct = processor.batch_decode(generated, skip_special_tokens=True)[0]
            official = transcriber({"array": audio.copy(), "sampling_rate": sr}, generate_kwargs=kwargs)["text"]
            item = {"id": row["id"], "reference": row["reference"], "audio_rms": float(np.sqrt(np.mean(audio**2))),
                    "direct_eager": direct, "official_pipeline_eager": official}
            results.append(item)
            print("DIAGNOSTIC", json.dumps(item, ensure_ascii=False), flush=True)
    save(OUT / "whisper_implementation_diagnostic.json", {"model": MODEL_IDS["whisper"], "revision": REVISIONS["whisper"],
         "attention": "eager", "decoding": kwargs, "feature_size": processor.feature_extractor.feature_size,
         "model_mel_bins": model.config.num_mel_bins, "samples": results})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--model", choices=MODEL_IDS)
    parser.add_argument("--diagnose-whisper", action="store_true")
    args = parser.parse_args()
    if args.model:
        evaluate(args.model)
        return
    if args.diagnose_whisper:
        diagnose_whisper()
        return
    if not args.evaluate:
        packages = ["numpy==2.1.3", "scipy==1.16.3", "torchmetrics==1.9.0", "pandas==2.3.3",
                    "protobuf==5.29.5", "librosa==0.11.0", "soundfile>=0.12.1", "nemo_toolkit[asr]==2.7.3",
                    "transformers==4.57.6", "jiwer==3.1.0"]
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "--upgrade-strategy", "only-if-needed", *packages], check=True)
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--evaluate"], check=True)
        return
    prepare()
    failures = []
    for key in RUN_MODELS:
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--model", key])
        if result.returncode:
            failures.append(key)
            save(OUT / key / "failure.json", {"status": "failed", "returncode": result.returncode})
        report()
    diagnostic = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--diagnose-whisper"])
    if diagnostic.returncode:
        save(OUT / "whisper_diagnostic_failure.json", {"returncode": diagnostic.returncode})
    if failures:
        raise RuntimeError(f"Incomplete models: {failures}; see logs and saved predictions")


if __name__ == "__main__":
    main()
