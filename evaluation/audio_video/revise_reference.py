"""Apply traceable ASR-assisted edits without using either scored model."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from asr_common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
VIDEO = ROOT / "video_eval"
EVIDENCE = ROOT / "results" / "podcast-kaggle" / "podcast_evaluation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = VIDEO / "reference_subtitle.cleaned.txt"
    text = source.read_text(encoding="utf-8")
    whisper = json.loads((EVIDENCE / "reference_assist" / "segments.json").read_text(encoding="utf-8"))
    ctc = json.loads((EVIDENCE / "reference_assist_ctc" / "segments.json").read_text(encoding="utf-8"))
    rules = [
        ("À, trên thực tế là cái khi", "Thì thật ra là cái khi", [1],
         "Whisper-v2: 'thực ra là khi'; CTC: 'thì thựt ra là cái khi'. Automatic editorial correction, not ear-verified."),
        ("chỉ số ung thư trên thực tế là", "chỉ số ung thư thật ra là", [3],
         "Both independent recognizers produce a 'thật/thực ra' construction rather than 'trên thực tế'."),
        ("nhưng mà thực ra nó cũng", "nhưng mà thật ra nó cũng", [4],
         "Both independent recognizers render 'thật ra' at this position."),
        ("mà thực ra nó có bệnh", "mà thật ra nó có bệnh", [4],
         "Both independent recognizers render 'thật ra' at this position."),
        ("tỷ lệ mà nó trẻ quá", "tỷ lệ mà nó trẻ hóa", [8, 12],
         "Automatic spelling/term proposal, consistent with 'khuynh hướng là trẻ hóa' elsewhere in the original. The independent ASR phonetics remain ambiguous; requires listener review."),
        ("Nhưng mà riêng đó lẽ là người bình thường", "Nhưng mà riêng đó là người bình thường", [10, 11],
         "CTC spans 'nhưng ... riêng' / 'đó là người bình thường'; Whisper-v2 produces 'đó là người bình thường'; no independent support for inserted 'lẽ'."),
    ]
    edits = []
    for old, new, ids, rationale in rules:
        if text.count(old) != 1:
            raise RuntimeError(f"Edit target is missing or ambiguous: {old}")
        text = text.replace(old, new, 1)
        edits.append({"old": old, "new": new, "kind": "ASR-assisted editorial proposal", "ear_verified": False,
                      "rationale": rationale,
                      "evidence": [{"id": ctc[i]["id"], "start_seconds": ctc[i]["start_seconds"],
                                    "end_seconds": ctc[i]["end_seconds"],
                                    "whisper_v2": whisper[i]["text"], "vietnamese_ctc": ctc[i]["text"]} for i in ids]})
    target = VIDEO / "reference_subtitle.corrected.txt"
    target.write_text(text, encoding="utf-8")
    (VIDEO / "reference_subtitle.txt").write_text(text, encoding="utf-8")
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_status": "ASR-assisted editorial draft; no direct listening by agent; age/wording ambiguities pending listener confirmation; not manually verified gold",
        "finalized_before_reading_candidate_predictions": True,
        "candidate_predictions_used_to_edit_reference": False,
        "original_subtitle_sha256": sha(VIDEO / "reference_subtitle.original.txt"),
        "cleaned_reference_sha256": sha(source), "corrected_reference_sha256": sha(target),
        "independent_sources": [
            {"model": "openai/whisper-large-v2", "revision": "ae4642769ce2ad8fc292556ccea8e901f1530655", "segments_sha256": sha(EVIDENCE / "reference_assist" / "segments.json")},
            {"model": "nguyenvulebinh/wav2vec2-base-vietnamese-250h", "revision": "69e9000591623e5a4fc2f502407860bcdc0de0b2", "segments_sha256": sha(EVIDENCE / "reference_assist_ctc" / "segments.json")},
        ],
        "edits": edits,
        "unresolved": [
            {"start_seconds": 175.95, "end_seconds": 194.95,
             "original": "trong y gia ... 20 năm thì chỉ là trước 40 tuổi",
             "note": "Independent CTC suggests '... y văn ...' and both ASR outputs are ambiguous about the age qualifier. Original wording/numbers retained until user confirms.",
             "review_clip": "review_clips/age_02m56_03m15.mp3"},
            {"start_seconds": 109.7, "end_seconds": 132.3,
             "original": "phát hiện chuyển trên cái niêm mạc",
             "note": "Independent sources disagree (triển/trên); original wording retained rather than guessing."},
        ],
    }
    (VIDEO / "subtitle_correction_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Reference draft saved;", len(edits), "traceable proposals; candidate predictions not used.")


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
