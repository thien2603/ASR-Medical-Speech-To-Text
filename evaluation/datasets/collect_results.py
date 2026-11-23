"""Verify downloaded Kaggle predictions before publishing the main comparison.

Download metrics first using the Kaggle CLI command documented in the project.
No network access or inference is performed by this verifier.
"""
from __future__ import annotations

import hashlib
import csv
from collections import Counter
import json
import shutil
from pathlib import Path

import jiwer
from asr_common.text import normalize_vi

from asr_common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
SOURCE = ROOT / "results" / "medical-kaggle" / "medical_benchmark"
TARGET = ROOT / "results" / "medical-main"
MODELS = {
    "kyle": ("kyle/vi-asr-fastconformer-114m", "3f498775970e051972b791b4428f5ad3a13a53fc"),
    "phowhisper": ("vinai/PhoWhisper-large", "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"),
    "whisper": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
}
REFERENCE_SHA = "0882a921c4afb6bac3cdd4063664c42019ac0bddcf328498b53205f359d3498a"


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(source: Path, check_table: bool = True) -> dict:
    dataset = json.loads((source / "dataset.json").read_text(encoding="utf-8"))
    manifest_bytes = (source / "manifest.jsonl").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != dataset["manifest_sha256"]:
        raise ValueError("Manifest hash mismatch")
    manifest = read_rows(source / "manifest.jsonl")
    if len(manifest) != 3437 or dataset["samples"] != 3437:
        raise ValueError("Comparison requires the complete 3437-sample test")
    ref_sha = hashlib.sha256(("\n".join(row["reference"] for row in manifest) + "\n").encode()).hexdigest()
    if ref_sha != REFERENCE_SHA:
        raise ValueError("References differ from the locally audited VietMed test")
    gpu = None
    summary_rows = []
    for key, (model_id, revision) in MODELS.items():
        summary = json.loads((source / key / "summary.json").read_text(encoding="utf-8"))
        predictions = read_rows(source / key / "predictions.jsonl")
        if summary["status"] != "complete" or summary["samples"] != 3437 or len(predictions) != 3437:
            raise ValueError(f"Incomplete run: {key}")
        if summary["model_id"] != model_id or summary["revision"] != revision:
            raise ValueError(f"Checkpoint differs from the agreed model: {key}")
        if key == "kyle" and summary.get("checkpoint_tensor_sha256") != "e30c130f25e3ab092d0d7b1a09ba45629ad5e66add405f194e496216c9946383":
            raise ValueError("Kyle tensors do not match the original Hugging Face checkpoint")
        if summary["manifest_sha256"] != dataset["manifest_sha256"] or summary["precision"] != "float16" or summary["batch_size"] != 4:
            raise ValueError(f"Protocol differs: {key}")
        if gpu is not None and summary["gpu"] != gpu:
            raise ValueError("Models ran on different GPU types")
        gpu = summary["gpu"]
        for actual, expected in zip(predictions, manifest):
            if any(actual[field] != expected[field] for field in ["id", "audio_sha256", "reference"]):
                raise ValueError(f"Prediction alignment mismatch: {key}")
        references = [row["reference"] for row in predictions]
        hypotheses = [row["prediction_normalized"] for row in predictions]
        word = jiwer.process_words(references, hypotheses)
        char = jiwer.process_characters(references, hypotheses)
        if abs(word.wer - summary["wer"]) > 1e-12 or abs(char.cer - summary["cer"]) > 1e-12:
            raise ValueError(f"Stored metrics differ from transcripts: {key}")
        for field, actual in [("word_substitutions", word.substitutions), ("word_deletions", word.deletions), ("word_insertions", word.insertions)]:
            if summary[field] != actual:
                raise ValueError(f"Incorrect error count: {key}/{field}")
        if abs(summary["rtf"] - summary["inference_seconds"] / dataset["audio_seconds"]) > 1e-12:
            raise ValueError(f"Incorrect RTF denominator: {key}")
        summary_rows.append({"model": model_id, "status": "complete", "samples": 3437,
                             "wer_percent": 100 * word.wer, "cer_percent": 100 * char.cer, "rtf": summary["rtf"],
                             "inference_seconds": summary["inference_seconds"],
                             "gpu_peak_allocated_gib": summary.get("gpu_peak_allocated_gib"),
                             "backend": summary.get("backend"), "precision": summary["precision"],
                             "batch_size": summary["batch_size"]})
    if check_table:
        table = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
        if len(table["models"]) != 3:
            raise ValueError("Comparison must contain exactly three models")
        for row in summary_rows:
            saved = next(item for item in table["models"] if item["model"] == row["model"])
            if saved["status"] != "complete" or any(abs(saved[field] - row[field]) > 1e-12 for field in ["wer_percent", "cer_percent", "rtf"]):
                raise ValueError("Comparison table differs from verified predictions")
    return {"status": "verified", "samples_per_model": 3437, "gpu": gpu, "models": summary_rows}


def main():
    # Merge independently completed runs after validating every model against
    # the same frozen manifest. Rebuild the table from verified raw predictions,
    # rather than copying a partial table from the latest repair-only job.
    result = verify(SOURCE, check_table=False)
    diagnostic_path = SOURCE / "whisper_implementation_diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    whisper_rows = read_rows(SOURCE / "whisper" / "predictions.jsonl")
    by_id = {row["id"]: row for row in whisper_rows}
    if len(diagnostic["samples"]) != 3:
        raise ValueError("Expected the three fixed Whisper implementation checks")
    for sample in diagnostic["samples"]:
        expected = by_id[sample["id"]]
        if sample["reference"] != expected["reference"] or any(normalize_vi(sample[field]) != expected["prediction_normalized"] for field in ["direct_eager", "official_pipeline_eager"]):
            raise ValueError("Whisper eager/official pipeline differs from the main run; investigate before publishing")
    result["whisper_implementation_check"] = {"matched_samples": 3, "methods": ["full-run SDPA batch=4", "direct eager batch=1", "official pipeline eager batch=1"]}
    TARGET.mkdir(parents=True, exist_ok=True)
    files = ["dataset.json", "manifest.jsonl", "whisper_implementation_diagnostic.json"]
    files += [f"{key}/{name}" for key in MODELS for name in ["config.json", "summary.json", "predictions.jsonl", "predictions.csv"]]
    for name in files:
        target = TARGET / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / name, target)
    (TARGET / "verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    repeated = Counter(row["prediction_normalized"] for row in whisper_rows).most_common(5)
    (TARGET / "whisper_repeated_outputs.json").write_text(json.dumps({"samples": len(whisper_rows), "most_common_predictions": repeated}, ensure_ascii=False, indent=2), encoding="utf-8")
    dataset = json.loads((TARGET / "dataset.json").read_text(encoding="utf-8"))
    combined = {"dataset": dataset, "gpu": result["gpu"], "models": result["models"],
                "source": "Private Kaggle benchmark; Whisper runs version 2, Kyle environment repair run; all runs verified against identical audio hashes/references",
                "note": "Same GPU type/FP16/batch=4, separate Kaggle sessions after environment repair; no test-driven decoding changes"}
    (TARGET / "comparison.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    with (TARGET / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["models"][0]))
        writer.writeheader()
        writer.writerows(result["models"])
    best = min(row["wer_percent"] for row in result["models"])
    lines = ["# So sánh chính: ASR trên VietMed test", "", "Mỗi model: toàn bộ 3.437 mẫu / 6,022 giờ; GPU Tesla T4, FP16, batch size 4.", "", "| Model gốc | WER (%) ↓ | CER (%) ↓ | RTF ↓ | Thời gian suy luận (s) | Peak GPU (GiB) |", "|---|---:|---:|---:|---:|---:|"]
    for row in result["models"]:
        wer = f"{row['wer_percent']:.2f}"
        if row["wer_percent"] == best:
            wer = f"**{wer}**"
        lines.append(f"| {row['model']} | {wer} | {row['cer_percent']:.2f} | {row['rtf']:.4f} | {row['inference_seconds']:.2f} | {row['gpu_peak_allocated_gib']:.2f} |")
    lines += ["", "WER/CER tính trên toàn corpus theo cùng chuẩn hóa; giữ chữ số và dấu tiếng Việt. RTF gồm đọc WAV, tiền xử lý và decoding, loại trừ load model và warm-up. Kyle dùng NeMo, hai Whisper dùng Transformers. Các lần chạy được thực hiện trong những phiên Kaggle khác nhau sau khi sửa lỗi môi trường, cùng loại GPU và cùng precision/batch size.", "", "Chỉ so sánh chất lượng phiên âm trên benchmark này; chưa đo riêng recall tên thuốc, liều lượng hoặc phủ định. Kết quả phụ thuộc cấu hình giải mã đã ghi trong config.json. Bản Kyle fine-tune + adaptive decoding WER 26,91% là thí nghiệm bổ sung có chọn cấu hình dựa trên test, không trộn vào bảng chính.", "", "Các metric đã được tính lại từ predictions.jsonl và đối chiếu ID/hash audio/reference. Xem verification.json, summary.json và transcript từng model để kiểm tra lại.", ""]
    lines += [f"Whisper-large-v3 sinh nhiều câu lạc nội dung: transcript chuẩn hóa phổ biến nhất xuất hiện {repeated[0][1]} lần. Ba mẫu đầu đã được đối chiếu bằng direct eager và pipeline Transformers chính thức (batch=1), cho cùng đầu ra với lần SDPA batch=4. Đây là kiểm tra ba mẫu, không chứng minh nguyên nhân của mọi lỗi trên toàn corpus; xem whisper_implementation_diagnostic.json và whisper_repeated_outputs.json.", ""]
    (TARGET / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
