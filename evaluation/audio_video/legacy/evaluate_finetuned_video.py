from __future__ import annotations

import gc
import json
import math
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
import torch
from nemo.collections.asr.models import ASRModel
from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
from omegaconf import OmegaConf, open_dict


from asr_common.paths import VIDEO_DIR, EXTERNAL_RUNTIME

ROOT = VIDEO_DIR
SOURCE_WAV = ROOT / "source_hls.wav"
REFERENCE_PATH = ROOT / "reference_subtitle.txt"
MODEL_PATH = EXTERNAL_RUNTIME / "local_decoding_fix" / "fastconformer_vietmed_best.nemo"
MODEL_EXTRACTED = ROOT / "model_extracted"
CHUNK_DIR = ROOT / "chunks"
TRANSCRIPT_PATH = ROOT / "model_transcript.txt"
SEGMENTS_PATH = ROOT / "segment_predictions.csv"
REPORT_PATH = ROOT / "video_wer_report.json"

SAMPLE_RATE = 16_000
TARGET_CHUNK_SECONDS = 7.5
MIN_CHUNK_SECONDS = 4.0
MAX_CHUNK_SECONDS = 10.0
BATCH_SIZE = 2


def normalize_text(text: str, *, reference: bool = False) -> str:
    text = unicodedata.normalize("NFC", str(text)).lower()
    if reference:
        text = re.sub(r"\[[^\]]*\]", " ", text)
        text = text.replace(">>", " ")
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


def load_audio() -> np.ndarray:
    audio, source_rate = sf.read(SOURCE_WAV, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        audio = scipy.signal.resample_poly(
            audio,
            SAMPLE_RATE // divisor,
            source_rate // divisor,
        ).astype(np.float32, copy=False)
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio /= peak
    return audio


def find_silence_centers(audio: np.ndarray) -> list[float]:
    frame = int(0.03 * SAMPLE_RATE)
    hop = int(0.01 * SAMPLE_RATE)
    if len(audio) < frame:
        return []
    count = 1 + (len(audio) - frame) // hop
    view = np.lib.stride_tricks.as_strided(
        audio,
        shape=(count, frame),
        strides=(audio.strides[0] * hop, audio.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.mean(view * view, axis=1) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-8)
    threshold = float(np.clip(np.quantile(db, 0.20) + 8.0, -45.0, -28.0))
    silent = db < threshold

    centers: list[float] = []
    start = None
    for index, is_silent in enumerate(np.append(silent, False)):
        if is_silent and start is None:
            start = index
        elif not is_silent and start is not None:
            duration = (index - start) * hop / SAMPLE_RATE
            if duration >= 0.25:
                centers.append(((start + index) / 2 * hop + frame / 2) / SAMPLE_RATE)
            start = None
    return centers


def choose_boundaries(duration: float, silence_centers: list[float]) -> list[float]:
    boundaries = [0.0]
    while duration - boundaries[-1] > MAX_CHUNK_SECONDS:
        start = boundaries[-1]
        low = start + MIN_CHUNK_SECONDS
        high = min(start + MAX_CHUNK_SECONDS, duration)
        target = start + TARGET_CHUNK_SECONDS
        candidates = [value for value in silence_centers if low <= value <= high]
        cut = min(candidates, key=lambda value: abs(value - target)) if candidates else high
        boundaries.append(cut)
    if duration - boundaries[-1] < 1.5 and len(boundaries) > 1:
        boundaries.pop()
    boundaries.append(duration)
    return boundaries


def materialize_chunks(audio: np.ndarray) -> pd.DataFrame:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    duration = len(audio) / SAMPLE_RATE
    boundaries = choose_boundaries(duration, find_silence_centers(audio))
    rows = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        path = CHUNK_DIR / f"chunk_{index:04d}.wav"
        left = int(round(start * SAMPLE_RATE))
        right = int(round(end * SAMPLE_RATE))
        sf.write(path, audio[left:right], SAMPLE_RATE, subtype="PCM_16")
        rows.append(
            {
                "segment": index,
                "start_seconds": start,
                "end_seconds": end,
                "duration_seconds": end - start,
                "audio_path": str(path),
            }
        )
    return pd.DataFrame(rows)


def load_existing_chunks() -> pd.DataFrame | None:
    paths = sorted(CHUNK_DIR.glob("chunk_*.wav"))
    if not paths:
        return None
    rows = []
    start = 0.0
    for index, path in enumerate(paths):
        info = sf.info(path)
        duration = float(info.frames / info.samplerate)
        rows.append(
            {
                "segment": index,
                "start_seconds": start,
                "end_seconds": start + duration,
                "duration_seconds": duration,
                "audio_path": str(path),
            }
        )
        start += duration
    return pd.DataFrame(rows)


def configure_decoder(model) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(model.cfg.decoding, resolve=True))
    with open_dict(cfg):
        cfg.strategy = "greedy_batch"
        cfg.greedy.max_symbols_per_step = 1
        cfg.greedy.loop_labels = False
        cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg, verbose=False)


def load_model_low_memory():
    """Build directly on CUDA and stream mmap'ed weights into FP16 parameters."""
    config = OmegaConf.load(MODEL_EXTRACTED / "model_config.yaml")
    config.tokenizer.dir = str(MODEL_EXTRACTED)
    config.tokenizer.model_path = str(
        MODEL_EXTRACTED / "de4724d69923495e9046ace865df70e1_tokenizer.model"
    )
    config.tokenizer.vocab_path = str(
        MODEL_EXTRACTED / "a9ea54b1e4c34f218e9f5de344a45448_vocab.txt"
    )
    config.tokenizer.spe_tokenizer_vocab = str(
        MODEL_EXTRACTED / "b55532e89da043f5a34545216a9f98e3_tokenizer.vocab"
    )
    # The archived config contains training manifests from the original server.
    # They are irrelevant for inference and unavailable on this machine.
    with open_dict(config):
        config.train_ds = None
        config.validation_ds = None
        config.test_ds = None
    # Construct FP16 parameters directly on CUDA. Setting the default dtype
    # before construction avoids a transient full-size FP32 parameter set.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("cuda"):
            model = ASRModel.from_config_dict(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval().half()
    state_dict = torch.load(
        MODEL_EXTRACTED / "model_weights.ckpt",
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    load_result = model.load_state_dict(state_dict, strict=True)
    print(
        f"Loaded mmap weights; missing={len(load_result.missing_keys)}, "
        f"unexpected={len(load_result.unexpected_keys)}",
        flush=True,
    )
    del state_dict
    gc.collect()
    return model


@torch.inference_mode()
def transcribe(model, segments: pd.DataFrame) -> tuple[list[str], float]:
    paths = segments.audio_path.astype(str).tolist()
    predictions: list[str] = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    total_batches = math.ceil(len(paths) / BATCH_SIZE)
    for batch_index, offset in enumerate(range(0, len(paths), BATCH_SIZE), start=1):
        batch_paths = paths[offset : offset + BATCH_SIZE]
        output = model.transcribe(
            batch_paths,
            batch_size=len(batch_paths),
            verbose=False,
            return_hypotheses=False,
        )
        predictions.extend(normalize_text(hypothesis_to_text(item)) for item in output)
        print(f"Decoded batch {batch_index}/{total_batches}", flush=True)
    torch.cuda.synchronize()
    return predictions, time.perf_counter() - started


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")

    reference_raw = REFERENCE_PATH.read_text(encoding="utf-8-sig")
    reference = normalize_text(reference_raw, reference=True)
    segments = load_existing_chunks()
    if segments is None:
        audio = load_audio()
        segments = materialize_chunks(audio)
        audio_seconds = len(audio) / SAMPLE_RATE
        del audio
        gc.collect()
    else:
        audio_seconds = float(segments.duration_seconds.sum())
    print(
        f"Audio: {audio_seconds:.3f}s; segments: {len(segments)}; "
        f"reference words: {len(reference.split())}",
        flush=True,
    )
    # Restore directly to CUDA. Restoring to CPU first keeps both the model and
    # checkpoint weights in system RAM at once, which is too much on this host.
    # Reuse the extracted archive so NeMo also skips a second unpacking step.
    connector = SaveRestoreConnector()
    connector.model_extracted_dir = str(MODEL_EXTRACTED)
    model = ASRModel.restore_from(
        str(MODEL_PATH),
        map_location=torch.device("cuda"),
        save_restore_connector=connector,
    )
    model.eval().half()
    configure_decoder(model)
    predictions, inference_seconds = transcribe(model, segments)
    hypothesis = normalize_text(" ".join(predictions))

    segments["prediction"] = predictions
    segments.to_csv(SEGMENTS_PATH, index=False, encoding="utf-8-sig")
    TRANSCRIPT_PATH.write_text(hypothesis + "\n", encoding="utf-8")

    word_result = jiwer.process_words(reference, hypothesis)
    reference_words = len(reference.split())
    hypothesis_words = len(hypothesis.split())
    word_accuracy = max(0.0, 1.0 - float(word_result.wer))
    reference_word_hit_rate = (
        float(word_result.hits / reference_words) if reference_words else 0.0
    )
    report = {
        "status": "complete",
        "source_video": "https://www.youtube.com/watch?v=Kwnvs2Zx9yI",
        "reference": "YouTube Vietnamese auto-generated subtitle supplied by user",
        "model": str(MODEL_PATH),
        "decoder": {
            "strategy": "greedy_batch",
            "max_symbols_per_step": 1,
            "loop_labels": False,
            "use_cuda_graph_decoder": False,
        },
        "audio_seconds": audio_seconds,
        "segments": len(segments),
        "reference_words": reference_words,
        "hypothesis_words": hypothesis_words,
        "wer": float(word_result.wer),
        "cer": float(jiwer.cer(reference, hypothesis)),
        "word_accuracy_one_minus_wer": word_accuracy,
        "reference_word_hit_rate": reference_word_hit_rate,
        "hits": int(word_result.hits),
        "substitutions": int(word_result.substitutions),
        "deletions": int(word_result.deletions),
        "insertions": int(word_result.insertions),
        "inference_seconds": inference_seconds,
        "rtf": inference_seconds / audio_seconds,
        "normalization": (
            "NFC + lowercase + remove [tags]/speaker markers + punctuation/symbols "
            "to spaces + collapse whitespace"
        ),
        "diagnostic_note": (
            "The reference is an automatic YouTube subtitle rather than a manually "
            "verified transcript, so this measures agreement with that subtitle."
        ),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("FINAL REPORT", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
