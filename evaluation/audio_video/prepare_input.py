"""Prepare a word-preserving subtitle reference and common podcast audio.

No model predictions are used to edit the reference. Lexical corrections require
independent verification against audio and are outside this cleanup operation.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from asr_common.text import normalize_vi

from asr_common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
VIDEO = ROOT / "video_eval"
BUNDLE = VIDEO / "kaggle_input"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_subtitle(raw: str) -> str:
    """Remove non-speech markup and join wrapped lines, preserving all words."""
    paragraphs, current = [], []
    for line in unicodedata.normalize("NFC", raw).splitlines():
        speaker = bool(re.match(r"\s*>>", line))
        text = re.sub(r"\[[^\]]*\]", " ", line)
        text = re.sub(r"^\s*>>\s*", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        if speaker and current:
            paragraphs.append(" ".join(current))
            current = []
        if text:
            current.append(text)
    if current:
        paragraphs.append(" ".join(current))
    return "\n\n".join(paragraphs).strip() + "\n"


def boundaries(audio: np.ndarray, sample_rate=16000) -> list[int]:
    """Choose waveform-only cuts near quiet points; retain every sample once."""
    maximum, minimum = 24 * sample_rate, 14 * sample_rate
    search, window, hop = 6 * sample_rate, int(.3 * sample_rate), int(.05 * sample_rate)
    cuts, start = [0], 0
    while len(audio) - start > maximum:
        low, high = start + minimum, start + maximum - window
        best, best_energy = start + maximum, float("inf")
        for point in range(max(low, start + maximum - search), high + 1, hop):
            energy = float(np.mean(audio[point:point + window] ** 2))
            if energy < best_energy:
                best, best_energy = point + window // 2, energy
        cuts.append(best)
        start = best
    cuts.append(len(audio))
    return cuts


def main():
    source = VIDEO / "source_hls.wav"
    original = VIDEO / "reference_subtitle.original.txt"
    editable = VIDEO / "reference_subtitle.txt"
    if not original.exists():
        original.write_bytes(editable.read_bytes())
    raw = original.read_text(encoding="utf-8-sig")
    cleaned = clean_subtitle(raw)
    stripped = re.sub(r"\[[^\]]*\]", " ", raw).replace(">>", " ")
    if normalize_vi(cleaned) != normalize_vi(stripped):
        raise RuntimeError("Subtitle cleanup changed the spoken-word sequence")
    (VIDEO / "reference_subtitle.cleaned.txt").write_text(cleaned, encoding="utf-8")
    BUNDLE.mkdir(exist_ok=True)
    (BUNDLE / "reference.txt").write_text(cleaned, encoding="utf-8")

    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
    if not np.isfinite(audio).all():
        raise RuntimeError("Non-finite audio")
    output_audio = BUNDLE / "audio_16k.flac"
    sf.write(output_audio, audio, 16000, subtype="PCM_16")
    cuts = boundaries(audio)
    manifest = {
        "source_file": source.name, "source_sha256": digest(source),
        "source_subtitle_sha256": digest(original),
        "audio_file": output_audio.name, "audio_sha256": digest(output_audio),
        "reference_file": "reference.txt", "reference_sha256": digest(BUNDLE / "reference.txt"),
        "sample_rate": 16000, "channels": 1, "audio_samples": len(audio),
        "audio_seconds": len(audio) / 16000,
        "reference_words": len(normalize_vi(cleaned).split()),
        "reference_status": "User-supplied automatic subtitle; formatting cleaned; wording/numbers not independently verified",
        "cleanup": "NFC; remove [non-speech tags] and >> markers; join wrapped lines; preserve all words, numbers, fillers and repetitions",
        "normalization": "NFC + lowercase + punctuation/symbols to spaces + collapse whitespace; preserve digits/diacritics",
        "chunk_rule": "waveform-only quiet cuts; max 24s, min 14s except tail; no overlap or dropped samples",
        "chunks": [{"id": f"podcast-{i:03d}", "start_sample": start, "end_sample": end,
                    "start_seconds": start/16000, "end_seconds": end/16000}
                   for i, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]))],
    }
    (BUNDLE / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k != "chunks"}, ensure_ascii=False, indent=2))
    print("chunks", len(manifest["chunks"]))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
