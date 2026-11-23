"""Compare original Kyle and PhoWhisper on the user's frozen podcast reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

os.environ.update(USE_TF="0", USE_FLAX="0", TOKENIZERS_PARALLELISM="false",
                  LHOTSE_IO_BACKEND="BuiltinIOBackend", LHOTSE_AUDIO_BACKEND="LibsndfileBackend")
OUT = Path("/kaggle/working/podcast_evaluation")
INPUT = Path("/kaggle/input")
MODELS = {
    "kyle": ("kyle/vi-asr-fastconformer-114m", "3f498775970e051972b791b4428f5ad3a13a53fc"),
    "phowhisper": ("vinai/PhoWhisper-large", "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"),
}
AUDIO_SHA = "788ca25aad67aaedd90ce11a8f31dce1195afc3011f32c6b7801cfb85f168e82"
REFERENCE_SHA = "a87e614573616c98b4f0df7d5343131656c26f3e5790aab9240638a6136eb124"
KYLE_STATE_SHA = "e30c130f25e3ab092d0d7b1a09ba45629ad5e66add405f194e496216c9946383"
BATCH = 4
KYLE_CHECKPOINT = None
CTC_REFERENCE_ONLY = False  # Fresh reproduction: both independent helpers and two candidates.


def normalize(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = "".join(" " if unicodedata.category(c)[0] in {"P", "S"} else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare():
    import soundfile as sf
    candidates = list(INPUT.rglob("evaluation_manifest.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one frozen input: {candidates}")
    source = candidates[0].parent
    metadata = json.loads(candidates[0].read_text(encoding="utf-8"))
    if digest(source / "audio_16k.flac") != AUDIO_SHA or digest(source / "reference.txt") != REFERENCE_SHA:
        raise RuntimeError("Uploaded input differs from locally prepared audio/reference")
    audio, rate = sf.read(source / "audio_16k.flac", dtype="float32")
    if rate != 16000 or audio.ndim != 1 or len(audio) != metadata["audio_samples"]:
        raise RuntimeError("Unexpected input audio format")
    reference = normalize((source / "reference.txt").read_text(encoding="utf-8-sig"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reference.normalized.txt").write_text(reference + "\n", encoding="utf-8")
    segments, previous = [], 0
    for row in metadata["chunks"]:
        start, end = row["start_sample"], row["end_sample"]
        if start != previous or not start < end <= len(audio):
            raise RuntimeError("Overlapping, dropped or invalid audio samples")
        path = OUT / "chunks" / f"{row['id']}.wav"
        path.parent.mkdir(exist_ok=True)
        sf.write(path, audio[start:end], rate, subtype="PCM_16")
        segments.append({**row, "audio_path": str(path), "audio_sha256": digest(path)})
        previous = end
    if previous != len(audio):
        raise RuntimeError("Missing end of audio")
    save(OUT / "input.json", {**metadata, "normalized_reference_words": len(reference.split())})
    save(OUT / "segments.json", segments)
    print("INPUT", metadata["audio_seconds"], "seconds", len(segments), "segments", len(reference.split()), "reference words", flush=True)


def evaluate(key):
    import importlib.metadata
    import jiwer
    import soundfile as sf
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    torch.manual_seed(42)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = json.loads((OUT / "segments.json").read_text())
    metadata = json.loads((OUT / "input.json").read_text())
    reference = (OUT / "reference.normalized.txt").read_text(encoding="utf-8").strip()
    model_id, revision = MODELS[key]
    config = {"model_id": model_id, "revision": revision, "precision": "float16", "batch_size": BATCH,
              "gpu": torch.cuda.get_device_name(0), "input_audio_sha256": AUDIO_SHA,
              "input_reference_sha256": REFERENCE_SHA,
              "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "nemo_toolkit", "jiwer"]}}
    if key == "kyle":
        from nemo.collections.asr.models import ASRModel
        from omegaconf import OmegaConf
        candidates = [KYLE_CHECKPOINT] if KYLE_CHECKPOINT else list(INPUT.rglob("vi-asr-fastconformer-114m-pretrained.nemo"))
        if len(candidates) != 1:
            raise RuntimeError("Original Kyle checkpoint not uniquely found")
        restore_kwargs = {}
        if KYLE_CHECKPOINT:
            from nemo.core.connectors.save_restore_connector import SaveRestoreConnector

            class MMapConnector(SaveRestoreConnector):
                @staticmethod
                def _load_state_dict_from_disk(model_weights, map_location=None):
                    return torch.load(model_weights, map_location=map_location or "cpu", weights_only=True, mmap=True)

            restore_kwargs["save_restore_connector"] = MMapConnector()
        model = ASRModel.restore_from(str(candidates[0]), map_location="cpu", **restore_kwargs)
        fingerprint = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            fingerprint.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode() + b"\n")
            fingerprint.update(tensor.detach().contiguous().numpy().tobytes())
        if fingerprint.hexdigest() != KYLE_STATE_SHA:
            raise RuntimeError("Kyle is not the original pretrained weights")
        model.eval().half().cuda()
        config.update(backend="NeMo", checkpoint_tensor_sha256=KYLE_STATE_SHA,
                      decoding=OmegaConf.to_container(model.cfg.decoding, resolve=True))

        def transcribe(batch):
            result = model.transcribe([r["audio_path"] for r in batch], batch_size=len(batch), num_workers=0,
                                      verbose=False, return_hypotheses=False)
            if isinstance(result, tuple):
                result = result[0]
            return [item if isinstance(item, str) else str(item.text) for item in result]
    else:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16,
                    low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
        kwargs = {"language": "vi", "task": "transcribe", "num_beams": 1, "do_sample": False,
                  "max_new_tokens": 440, "return_timestamps": False, "condition_on_prev_tokens": False}
        config.update(backend="Transformers SDPA", decoding=kwargs)

        def transcribe(batch):
            arrays = [sf.read(row["audio_path"], dtype="float32")[0] for row in batch]
            features = processor(arrays, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            generated = model.generate(input_features=features.input_features.to("cuda", torch.float16),
                                       attention_mask=features.attention_mask.to("cuda"), **kwargs)
            return processor.batch_decode(generated, skip_special_tokens=True)

    folder = OUT / key
    folder.mkdir(exist_ok=True)
    save(folder / "config.json", config)
    predictions, seconds = [], 0.0
    with torch.inference_mode():
        transcribe(rows[:1])
        torch.cuda.synchronize()
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start+BATCH]
            torch.cuda.synchronize()
            tick = time.perf_counter()
            hypotheses = transcribe(batch)
            torch.cuda.synchronize()
            seconds += time.perf_counter() - tick
            if len(hypotheses) != len(batch):
                raise RuntimeError("Prediction count mismatch")
            predictions.extend({**row, "prediction": hyp, "prediction_normalized": normalize(hyp)}
                               for row, hyp in zip(batch, hypotheses))
            print("PROGRESS", key, len(predictions), "/", len(rows), flush=True)
    raw = " ".join(row["prediction"].strip() for row in predictions)
    hypothesis = normalize(raw)
    (folder / "transcript.txt").write_text(raw + "\n", encoding="utf-8")
    (folder / "transcript.normalized.txt").write_text(hypothesis + "\n", encoding="utf-8")
    save(folder / "predictions.json", predictions)
    with (folder / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    words, characters = jiwer.process_words(reference, hypothesis), jiwer.process_characters(reference, hypothesis)
    summary = {**config, "status": "complete", "audio_seconds": metadata["audio_seconds"], "segments": len(rows),
               "reference_status": metadata["reference_status"], "reference_words": words.hits + words.substitutions + words.deletions,
               "wer": words.wer, "cer": characters.cer, "substitutions": words.substitutions,
               "deletions": words.deletions, "insertions": words.insertions, "hits": words.hits,
               "inference_seconds": seconds, "rtf": seconds / metadata["audio_seconds"],
               "timing": "batch wall time including input/preprocessing/decoding; model load and warm-up excluded",
               "normalization": metadata["normalization"]}
    save(folder / "summary.json", summary)
    (folder / "word_alignment.txt").write_text(jiwer.visualize_alignment(words), encoding="utf-8")
    print("RESULT", json.dumps(summary, ensure_ascii=False), flush=True)


def reference_assist():
    """Independent ASR evidence for subtitle repair; never a scored candidate."""
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
    model_id = "openai/whisper-large-v2"
    revision = "ae4642769ce2ad8fc292556ccea8e901f1530655"
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16,
                low_cpu_mem_usage=True, attn_implementation="sdpa").eval().cuda()
    rows = json.loads((OUT / "segments.json").read_text())
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            arrays = [sf.read(row["audio_path"], dtype="float32")[0] for row in batch]
            features = processor(arrays, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
            output = model.generate(input_features=features.input_features.to("cuda", torch.float16),
                                    attention_mask=features.attention_mask.to("cuda"), language="vi", task="transcribe",
                                    do_sample=False, num_beams=1, max_new_tokens=440,
                                    condition_on_prev_tokens=False, return_timestamps=False)
            texts = processor.batch_decode(output, skip_special_tokens=True)
            records.extend({**row, "text": text.strip()} for row, text in zip(batch, texts))
            print("REFERENCE_ASSIST", len(records), "/", len(rows), flush=True)
    folder = OUT / "reference_assist"
    folder.mkdir(exist_ok=True)
    save(folder / "segments.json", records)
    save(folder / "provenance.json", {"model": model_id, "revision": revision,
         "purpose": "independent automatic transcription for subtitle editorial review; not manually verified ground truth",
         "uses_candidate_predictions": False})
    (folder / "transcript.txt").write_text("\n\n".join(row["text"] for row in records) + "\n", encoding="utf-8")


def reference_assist_ctc():
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, AutoModelForCTC
    model_id = "nguyenvulebinh/wav2vec2-base-vietnamese-250h"
    revision = "69e9000591623e5a4fc2f502407860bcdc0de0b2"
    torch.set_num_threads(2)
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForCTC.from_pretrained(model_id, revision=revision).eval().cuda()
    rows = json.loads((OUT / "segments.json").read_text())
    records = []
    with torch.inference_mode():
        for row in rows:
            audio, rate = sf.read(row["audio_path"], dtype="float32")
            features = processor(audio, sampling_rate=rate, return_tensors="pt")
            output = model(**features.to("cuda")).logits.argmax(dim=-1)
            text = processor.batch_decode(output)[0]
            records.append({**row, "text": text})
            print("CTC_REFERENCE_ASSIST", len(records), "/", len(rows), flush=True)
    folder = OUT / "reference_assist_ctc"
    folder.mkdir(exist_ok=True)
    save(folder / "segments.json", records)
    save(folder / "provenance.json", {"model": model_id, "revision": revision,
         "purpose": "independent Vietnamese CTC evidence for subtitle editorial review; not manually verified ground truth",
         "uses_candidate_predictions": False})
    (folder / "transcript.txt").write_text("\n\n".join(row["text"] for row in records) + "\n", encoding="utf-8")


def main():
    global OUT, INPUT, KYLE_CHECKPOINT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--reference-assist", action="store_true")
    parser.add_argument("--reference-assist-ctc", action="store_true")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--kyle-checkpoint", type=Path)
    args = parser.parse_args()
    if args.local:
        root = Path(__file__).resolve().parents[3]
        OUT = root / "video_eval" / "two_model_local"
        INPUT = root / "video_eval" / "kaggle_input"
        KYLE_CHECKPOINT = args.kyle_checkpoint
        if args.model != "kyle" or not KYLE_CHECKPOINT or not KYLE_CHECKPOINT.is_file():
            raise RuntimeError("Local mode requires --model kyle and a valid --kyle-checkpoint")
        prepare()
    if args.model:
        evaluate(args.model)
        return
    if args.reference_assist:
        reference_assist()
        return
    if args.reference_assist_ctc:
        reference_assist_ctc()
        return
    if not args.run:
        dependencies = ["numpy==2.1.3", "scipy==1.16.3", "torchmetrics==1.9.0", "pandas==2.3.3",
                        "protobuf==5.29.5", "librosa==0.11.0", "soundfile>=0.12.1", "nemo_toolkit[asr]==2.7.3",
                        "transformers==4.57.6", "jiwer==3.1.0"]
        if CTC_REFERENCE_ONLY:
            dependencies = ["numpy==2.1.3", "scipy==1.16.3", "librosa==0.11.0", "soundfile>=0.12.1", "transformers==4.57.6", "jiwer==3.1.0"]
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "-q", "--upgrade-strategy", "only-if-needed", *dependencies], check=True)
        subprocess.run([sys.executable, "-u", __file__, "--run"], check=True)
        return
    prepare()
    if CTC_REFERENCE_ONLY:
        subprocess.run([sys.executable, "-u", __file__, "--reference-assist-ctc"], check=True)
        return
    assist = subprocess.run([sys.executable, "-u", __file__, "--reference-assist"])
    if assist.returncode:
        save(OUT / "reference_assist" / "failure.json", {"returncode": assist.returncode})
    subprocess.run([sys.executable, "-u", __file__, "--reference-assist-ctc"], check=True)
    failures = []
    for key in MODELS:
        result = subprocess.run([sys.executable, "-u", __file__, "--model", key])
        if result.returncode:
            failures.append(key)
            save(OUT / key / "failure.json", {"returncode": result.returncode})
    if failures:
        raise RuntimeError(f"Incomplete models: {failures}")


if __name__ == "__main__":
    main()
