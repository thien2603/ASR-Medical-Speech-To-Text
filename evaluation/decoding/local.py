from __future__ import annotations

import gc
import json
import re
import time
import unicodedata
from itertools import groupby
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
# Preload SciPy's compiled signal modules before PyTorch/NeMo. On this Windows
# host the system pagefile is small, and loading these DLLs after PyTorch can
# fail with WinError 1455 even though enough physical RAM is available.
import scipy.signal  # noqa: F401
import torch
import yaml
from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf, open_dict


from asr_common.paths import PROJECT_ROOT, EXTERNAL_RUNTIME

WORKSPACE = PROJECT_ROOT
RUNTIME = EXTERNAL_RUNTIME / "local_decoding_fix"
MODEL_PATH = RUNTIME / "fastconformer_vietmed_best.nemo"
TEST_PARQUET = WORKSPACE / "local_decoding_fix" / "test.parquet"
V5_PREDICTIONS = WORKSPACE / "kaggle_v5_report" / "test_predictions.csv"
AUDIO_DIR = RUNTIME / "test_audio"
OUTPUT_DIR = WORKSPACE / "local_decoding_fix"

MAX_WORDS_PER_SECOND = 8.0
MAX_CONSECUTIVE_WORD_RUN = 4
MIN_ABNORMAL_WORD_COUNT = 40
MAX_WORDS_PER_AUDIO_SECOND = 7.0
PROBE_SAMPLES = 8
GUARD_SAMPLES = 24
INFERENCE_BATCH_SIZE = 4


def normalize_vi_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower().strip()
    text = "".join(
        " " if unicodedata.category(char)[0] in {"P", "S"} else char
        for char in text
    )
    return re.sub(r"\s+", " ", text).strip()


def hypothesis_to_text(item) -> str:
    if isinstance(item, str):
        return item
    if hasattr(item, "text"):
        return item.text
    return str(item)


def max_word_run(text: str) -> int:
    words = str(text).split()
    return max((sum(1 for _ in run) for _, run in groupby(words)), default=0)


def repetition_features(prediction: str, duration: float) -> dict:
    words = str(prediction).split()
    count = len(words)
    duration = max(float(duration), 1e-6)
    words_per_second = count / duration
    longest_run = max_word_run(prediction)
    abnormal_length = count > max(
        MIN_ABNORMAL_WORD_COUNT, MAX_WORDS_PER_AUDIO_SECOND * duration
    )
    suspicious = (
        words_per_second > MAX_WORDS_PER_SECOND
        or longest_run >= MAX_CONSECUTIVE_WORD_RUN
        or abnormal_length
    )
    return {
        "word_count": count,
        "words_per_second": words_per_second,
        "max_word_run": longest_run,
        "suspicious_repetition": bool(suspicious),
    }


def enrich(frame: pd.DataFrame) -> pd.DataFrame:
    feature_names = [
        "word_count",
        "words_per_second",
        "max_word_run",
        "suspicious_repetition",
    ]
    base = frame.drop(columns=feature_names, errors="ignore").reset_index(drop=True)
    features = [
        repetition_features(prediction, duration)
        for prediction, duration in zip(base.prediction, base.duration)
    ]
    return pd.concat([base, pd.DataFrame(features)], axis=1)


def safe_sample_wer(reference: str, prediction: str) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return float(jiwer.wer(reference, prediction))


def summarize(frame: pd.DataFrame, seconds: float | None = None) -> dict:
    enriched = enrich(frame)
    references = enriched.reference.fillna("").astype(str).tolist()
    predictions = enriched.prediction.fillna("").astype(str).tolist()
    sample_wers = [
        safe_sample_wer(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ]
    audio_seconds = float(enriched.duration.astype(float).sum())
    return {
        "samples": len(enriched),
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
        "median_sample_wer": float(np.median(sample_wers)),
        "p90_sample_wer": float(np.quantile(sample_wers, 0.9)),
        "suspicious_repetition_count": int(enriched.suspicious_repetition.sum()),
        "suspicious_repetition_rate": float(enriched.suspicious_repetition.mean()),
        "inference_seconds": seconds,
        "rtf": float(seconds / audio_seconds) if seconds is not None else None,
    }


def set_decoding(model, max_symbols: int):
    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.greedy.max_symbols_per_step = int(max_symbols)
        cfg.greedy.loop_labels = False
        cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg, verbose=False)
    return cfg


@torch.inference_mode()
def transcribe(model, rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    paths = rows.local_audio_path.astype(str).tolist()
    torch.cuda.synchronize()
    started = time.perf_counter()
    hypotheses = []
    total_chunks = (len(paths) + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    for chunk_index, start in enumerate(
        range(0, len(paths), INFERENCE_BATCH_SIZE), start=1
    ):
        chunk_paths = paths[start : start + INFERENCE_BATCH_SIZE]
        chunk_hypotheses = model.transcribe(
            chunk_paths,
            batch_size=len(chunk_paths),
            verbose=False,
            return_hypotheses=False,
        )
        hypotheses.extend(chunk_hypotheses)
        if chunk_index == 1 or chunk_index % 10 == 0 or chunk_index == total_chunks:
            print(
                f"Decoded chunk {chunk_index}/{total_chunks} "
                f"({len(hypotheses)}/{len(paths)} files)"
            )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    predictions = [normalize_vi_text(hypothesis_to_text(item)) for item in hypotheses]
    frame = pd.DataFrame({
        "row_index": rows.row_index.astype(int).tolist(),
        "audio_filepath": paths,
        "duration": rows.duration.astype(float).tolist(),
        "reference": rows.reference.astype(str).tolist(),
        "prediction": predictions,
    })
    frame["sample_wer"] = [
        safe_sample_wer(reference, prediction)
        for reference, prediction in zip(frame.reference, frame.prediction)
    ]
    return enrich(frame), seconds


def prepare_audio(metadata: pd.DataFrame) -> pd.DataFrame:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index, audio in enumerate(metadata.audio):
        suffix = Path(str(audio.get("path", "audio.ogg"))).suffix or ".ogg"
        target = AUDIO_DIR / f"{index:08d}{suffix.lower()}"
        if not target.exists() or target.stat().st_size != len(audio["bytes"]):
            target.write_bytes(audio["bytes"])
        paths.append(str(target))
    result = pd.DataFrame({
        "row_index": np.arange(len(metadata)),
        "local_audio_path": paths,
        "duration": metadata.duration.astype(float),
        "reference": metadata.text.map(normalize_vi_text),
    })
    return result


def load_prepared_rows() -> pd.DataFrame:
    """Load lightweight metadata and reuse audio already materialized on E:."""
    lightweight = pd.read_parquet(TEST_PARQUET, columns=["duration", "text"])
    audio_paths = sorted(AUDIO_DIR.glob("*"))
    if len(audio_paths) == len(lightweight):
        return pd.DataFrame({
            "row_index": np.arange(len(lightweight)),
            "local_audio_path": [str(path) for path in audio_paths],
            "duration": lightweight.duration.astype(float),
            "reference": lightweight.text.map(normalize_vi_text),
        })

    # Fallback for a fresh machine where the encoded clips have not yet been
    # extracted. Release the bytes immediately after writing them to disk.
    metadata = pd.read_parquet(TEST_PARQUET)
    result = prepare_audio(metadata)
    del metadata
    gc.collect()
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError(f"PyTorch CUDA chưa sẵn sàng: {torch.__version__}")
    print("GPU:", torch.cuda.get_device_name(0))
    print("Free/total GPU bytes:", torch.cuda.mem_get_info())

    rows = load_prepared_rows()
    default_frame = pd.read_csv(V5_PREDICTIONS).fillna("")
    default_frame["reference"] = default_frame.reference.map(normalize_vi_text)
    default_frame["prediction"] = default_frame.prediction.map(normalize_vi_text)
    default_frame = enrich(default_frame)
    suspicious_positions = np.flatnonzero(
        default_frame.suspicious_repetition.to_numpy(dtype=bool)
    ).tolist()
    print("Default suspicious rows:", len(suspicious_positions))

    probe_positions = suspicious_positions[:PROBE_SAMPLES]
    guard_positions = list(range(1000, 1000 + GUARD_SAMPLES))
    model = ASRModel.restore_from(str(MODEL_PATH), map_location="cpu")
    # FP16 keeps the 114M-parameter model comfortably within the 4 GB VRAM
    # budget of the RTX 3050 while preserving inference behavior.
    model.eval().half().cuda()

    candidates = []
    # max_symbols=2 can still exhaust 4 GB VRAM on the known runaway rows.
    # max_symbols=1 completed the local smoke test without repetition.
    for max_symbols in (1,):
        print("Candidate max_symbols_per_step=", max_symbols)
        set_decoding(model, max_symbols)
        probe, probe_seconds = transcribe(model, rows.iloc[probe_positions])
        guard, guard_seconds = transcribe(model, rows.iloc[guard_positions])
        probe_summary = summarize(probe, probe_seconds)
        guard_summary = summarize(guard, guard_seconds)
        candidates.append({
            "max_symbols_per_step": max_symbols,
            "probe_wer": probe_summary["wer"],
            "probe_cer": probe_summary["cer"],
            "probe_suspicious": probe_summary["suspicious_repetition_count"],
            "guard_wer": guard_summary["wer"],
            "guard_cer": guard_summary["cer"],
            "seconds": probe_seconds + guard_seconds,
        })
        probe.to_csv(
            OUTPUT_DIR / f"local_probe_max_symbols_{max_symbols}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    candidates_frame = pd.DataFrame(candidates)
    candidates_frame.to_csv(
        OUTPUT_DIR / "local_decoding_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(candidates_frame.to_string(index=False))

    eligible = candidates_frame[candidates_frame.probe_suspicious == 0].copy()
    if eligible.empty:
        eligible = candidates_frame.copy()
        eligible["penalty"] = eligible.probe_suspicious * 100 + eligible.probe_wer
        selected = eligible.sort_values(
            ["penalty", "guard_wer", "probe_wer"]
        ).iloc[0]
    else:
        selected = eligible.sort_values(
            ["guard_wer", "probe_wer", "max_symbols_per_step"],
            ascending=[True, True, False],
        ).iloc[0]
    selected_max_symbols = int(selected.max_symbols_per_step)
    print("Selected max_symbols_per_step:", selected_max_symbols)

    selected_cfg = set_decoding(model, selected_max_symbols)
    repair, repair_seconds = transcribe(model, rows.iloc[suspicious_positions])
    adaptive = default_frame[
        ["audio_filepath", "duration", "reference", "prediction", "sample_wer"]
    ].copy()
    for local_index, original_position in enumerate(suspicious_positions):
        adaptive.loc[original_position, "prediction"] = repair.loc[
            local_index, "prediction"
        ]
    adaptive["sample_wer"] = [
        safe_sample_wer(reference, prediction)
        for reference, prediction in zip(adaptive.reference, adaptive.prediction)
    ]
    adaptive = enrich(adaptive)
    adaptive_summary = summarize(adaptive, repair_seconds)
    adaptive.to_csv(
        OUTPUT_DIR / "finetuned_adaptive_predictions_local.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(
        OUTPUT_DIR / "finetuned_adaptive_summary_local.json", adaptive_summary
    )

    fixed_model = RUNTIME / "fastconformer_vietmed_decoding_fixed_local.nemo"
    # Preserve a portable FP32 artifact; only local inference uses FP16.
    model.float().cpu()
    model.save_to(str(fixed_model))
    report = {
        "status": "complete",
        "runtime": "local RTX 3050 Laptop GPU, FP16, batch_size=4",
        "diagnostic_note": "Configuration selected on a known failing test probe.",
        "selected_max_symbols_per_step": selected_max_symbols,
        "candidate_results": candidates,
        "v5_default_summary": summarize(default_frame),
        "adaptive_summary": adaptive_summary,
        "redecoded_samples": len(suspicious_positions),
        "fixed_model": str(fixed_model),
        "decoding_config": OmegaConf.to_container(selected_cfg, resolve=True),
    }
    write_json(OUTPUT_DIR / "local_decoding_fix_report.json", report)
    (OUTPUT_DIR / "local_recommended_decoding.yaml").write_text(
        yaml.safe_dump(
            {
                "strategy": "greedy_batch",
                "greedy": {
                    "max_symbols_per_step": selected_max_symbols,
                    "loop_labels": False,
                    "use_cuda_graph_decoder": False,
                },
                "adaptive_detector": {
                    "max_words_per_second": MAX_WORDS_PER_SECOND,
                    "max_consecutive_word_run": MAX_CONSECUTIVE_WORD_RUN,
                    "min_abnormal_word_count": MIN_ABNORMAL_WORD_COUNT,
                    "max_words_per_audio_second": MAX_WORDS_PER_AUDIO_SECOND,
                },
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print("FINAL REPORT")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
