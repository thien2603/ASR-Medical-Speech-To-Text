"""Verify the two podcast runs and score against a separately revised reference."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import jiwer
from asr_common.text import normalize_vi

from asr_common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
SOURCE = ROOT / "results" / "podcast-kaggle" / "podcast_evaluation"
OUT = ROOT / "video_eval" / "two_model_results"
REFERENCE = ROOT / "video_eval" / "reference_subtitle.corrected.txt"
MODELS = {
    "kyle": ("kyle/vi-asr-fastconformer-114m", "3f498775970e051972b791b4428f5ad3a13a53fc"),
    "phowhisper": ("vinai/PhoWhisper-large", "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata = read(ROOT / "video_eval" / "kaggle_input" / "evaluation_manifest.json")
    actual_input = read(SOURCE / "input.json")
    edits = read(ROOT / "video_eval" / "subtitle_correction_audit.json")
    if edits["candidate_predictions_used_to_edit_reference"]:
        raise ValueError("Reference must be revised independently of candidate predictions")
    if digest(ROOT / "video_eval" / "source_hls.wav") != metadata["source_sha256"]:
        raise ValueError("Source WAV changed since preparation")
    if edits["corrected_reference_sha256"] != digest(REFERENCE):
        raise ValueError("Corrected reference changed after it was frozen")
    for field in ["audio_sha256", "reference_sha256", "audio_samples", "chunks"]:
        if actual_input[field] != metadata[field]:
            raise ValueError(f"Input differs from locally prepared source: {field}")
    reference = normalize_vi(REFERENCE.read_text(encoding="utf-8-sig"))
    initial_reference = normalize_vi((ROOT / "video_eval" / "kaggle_input" / "reference.txt").read_text(encoding="utf-8-sig"))
    models, common_chunk_hashes = [], None
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (model_id, revision) in MODELS.items():
        summary = read(SOURCE / key / "summary.json")
        predictions = read(SOURCE / key / "predictions.json")
        if summary["status"] != "complete" or summary["model_id"] != model_id or summary["revision"] != revision:
            raise ValueError(f"Incomplete or wrong model: {key}")
        if summary["input_audio_sha256"] != metadata["audio_sha256"] or summary["input_reference_sha256"] != metadata["reference_sha256"]:
            raise ValueError(f"Input provenance differs: {key}")
        if key == "kyle" and summary.get("checkpoint_tensor_sha256") != "e30c130f25e3ab092d0d7b1a09ba45629ad5e66add405f194e496216c9946383":
            raise ValueError("Kyle weights differ from original pretrained checkpoint")
        if len(predictions) != len(metadata["chunks"]):
            raise ValueError(f"Missing audio chunks: {key}")
        for actual, expected in zip(predictions, metadata["chunks"]):
            if any(actual[field] != expected[field] for field in ["id", "start_sample", "end_sample"]):
                raise ValueError(f"Audio span mismatch: {key}")
        hashes = [row["audio_sha256"] for row in predictions]
        if common_chunk_hashes is not None and hashes != common_chunk_hashes:
            raise ValueError("Models did not receive identical PCM chunks")
        common_chunk_hashes = hashes
        hypothesis = normalize_vi(" ".join(row["prediction"] for row in predictions))
        old_words = jiwer.process_words(initial_reference, hypothesis)
        old_chars = jiwer.process_characters(initial_reference, hypothesis)
        if abs(old_words.wer - summary["wer"]) > 1e-12 or abs(old_chars.cer - summary["cer"]) > 1e-12:
            raise ValueError(f"Remote metrics do not match the saved transcripts: {key}")
        words, chars = jiwer.process_words(reference, hypothesis), jiwer.process_characters(reference, hypothesis)
        folder = OUT / key
        folder.mkdir(exist_ok=True)
        for name in ["transcript.txt", "predictions.json", "predictions.csv", "config.json"]:
            shutil.copy2(SOURCE / key / name, folder / name)
        result = {**summary, "wer": words.wer, "cer": chars.cer,
                  "reference_file": str(REFERENCE), "corrected_reference_sha256": digest(REFERENCE),
                  "reference_status": edits["reference_status"],
                  "reference_words": words.hits + words.substitutions + words.deletions,
                  "substitutions": words.substitutions, "deletions": words.deletions,
                  "insertions": words.insertions, "hits": words.hits,
                  "initial_subtitle_wer": old_words.wer, "initial_subtitle_cer": old_chars.cer}
        (folder / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (folder / "word_alignment.txt").write_text(jiwer.visualize_alignment(words), encoding="utf-8")
        models.append({"model": model_id, "wer_percent": words.wer * 100, "cer_percent": chars.cer * 100,
                       "inference_seconds": summary["inference_seconds"], "rtf": summary["rtf"],
                       "initial_subtitle_wer_percent": old_words.wer * 100,
                       "initial_subtitle_cer_percent": old_chars.cer * 100})
    report = {"status": "verified_against_stated_reference", "audio_seconds": metadata["audio_seconds"],
              "segments": len(metadata["chunks"]), "reference_words": len(reference.split()),
              "reference_status": edits["reference_status"], "reference_sha256": digest(REFERENCE),
              "candidate_predictions_used_to_edit_reference": False, "models": models}
    (OUT / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(models[0]))
        writer.writeheader()
        writer.writerows(models)
    lines = ["# WER/CER trên podcast của dự án", "", f"Audio: {metadata['audio_seconds']:.3f} giây, {len(metadata['chunks'])} đoạn giống nhau cho cả hai model.", "", "| Model gốc | WER | CER | Thời gian nhận dạng |", "|---|---:|---:|---:|"]
    for model in models:
        lines.append(f"| {model['model']} | {model['wer_percent']:.2f}% | {model['cer_percent']:.2f}% | {model['inference_seconds']:.2f} giây |")
    lines += ["", "Reference được sửa từ phụ đề ban đầu với bằng chứng từ Whisper-large-v2 độc lập, chưa được con người nghe duyệt toàn bộ. Đây là số đo theo bản tham chiếu đã nêu, chưa phải benchmark với transcript chuẩn thủ công.", "", "Không sử dụng đầu ra Kyle/PhoWhisper để sửa reference. Giữ chữ số và dấu tiếng Việt khi chấm. Mô hình nhận dạng cùng waveform/chunk; WER/CER được tính lại cục bộ từ transcript. Các từ, số tuổi hoặc chỗ còn nghi vấn cần xem subtitle_correction_audit.json trước khi dùng số liệu như kết quả chính thức.", ""]
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.copy2(REFERENCE, OUT / "reference_subtitle.corrected.txt")
    shutil.copy2(ROOT / "video_eval" / "subtitle_correction_audit.json", OUT / "subtitle_correction_audit.json")
    status_path = ROOT / "video_eval" / "two_model_evaluation_status.json"
    status = read(status_path)
    status.update(status="scored_provisionally_awaiting_listener_reference_confirmation",
                  wer_cer_results=models, reference_words=len(reference.split()),
                  pending_action="Listener confirmation of ambiguous phrases around 02:56-03:15; candidate runs complete and do not need repeating",
                  reference_status=edits["reference_status"])
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    review = ["# Hiệu chỉnh phụ đề podcast", "",
              "Đã nhận dạng xong Kyle gốc và PhoWhisper-large. Reference được sửa với hỗ trợ của Whisper-large-v2 và wav2vec2 tiếng Việt độc lập; chưa được con người nghe duyệt toàn bộ.", "",
              "Bản gốc: reference_subtitle.original.txt. Bản sạch định dạng: reference_subtitle.cleaned.txt. Bản hiệu chỉnh: reference_subtitle.corrected.txt; đồng thời cập nhật reference_subtitle.txt.", "",
              "## Các đề xuất sửa đã áp dụng", "", "| Trước | Sau | Khoảng audio |", "|---|---|---|"]
    for edit in edits["edits"]:
        ranges = ", ".join(f"{item['start_seconds']:.2f}-{item['end_seconds']:.2f} giây" for item in edit["evidence"])
        review.append(f"| {edit['old']} | {edit['new']} | {ranges} |")
    review += ["", "Các sửa đổi trên là đề xuất biên tập có hỗ trợ ASR, chưa được nghe xác minh trực tiếp. Không dùng transcript của Kyle/PhoWhisper để sửa reference.", "",
               "## Cần nghe xác nhận", "",
               "- 2:56-3:15: cụm 'trong y gia' và '20 năm thì chỉ là trước 40 tuổi'. Giữ lời cũ tới khi người nghe xác nhận, đặc biệt không tự đổi con số hoặc chiều trước/trên tuổi.",
               "- 1:50-2:12: cụm 'phát hiện chuyển trên cái niêm mạc'. Nguồn độc lập chưa thống nhất; giữ lời cũ.", "",
               "[Nghe đoạn số tuổi](review_clips/age_02m56_03m15.mp3).", "",
               "## Số liệu tạm tính", "", "| Model | WER | CER |", "|---|---:|---:|"]
    review += [f"| {item['model']} | {item['wer_percent']:.2f}% | {item['cer_percent']:.2f}% |" for item in models]
    review += ["", "Các metric đã tính lại từ transcript nhưng chỉ phản ánh mức độ khớp bản reference đã nêu. Cần nghe duyệt reference trước khi coi đây là benchmark với đáp án chuẩn thủ công.", ""]
    (ROOT / "video_eval" / "SUBTITLE_REVIEW.md").write_text("\n".join(review), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
