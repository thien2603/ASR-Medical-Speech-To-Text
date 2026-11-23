import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.datasets import collect_results as collector


class ResultIntegrityTests(unittest.TestCase):
    def make_fixture(self, root):
        rows = [{"id": f"sample-{i}", "reference": "thuốc", "audio_sha256": f"audio-{i}"} for i in range(3437)]
        manifest_text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        manifest_hash = hashlib.sha256(manifest_text.encode()).hexdigest()
        ref_hash = hashlib.sha256(("thuốc\n" * 3437).encode()).hexdigest()
        (root / "manifest.jsonl").write_bytes(manifest_text.encode())
        (root / "dataset.json").write_text(json.dumps({"samples": 3437, "manifest_sha256": manifest_hash, "audio_seconds": 3437}), encoding="utf-8")
        predictions = "".join(json.dumps({**row, "prediction_normalized": "thuốc"}, ensure_ascii=False) + "\n" for row in rows)
        comparison = []
        for key, (model, revision) in collector.MODELS.items():
            (root / key).mkdir()
            (root / key / "predictions.jsonl").write_text(predictions, encoding="utf-8")
            summary = {"status": "complete", "samples": 3437, "model_id": model, "revision": revision,
                       "manifest_sha256": manifest_hash, "precision": "float16", "batch_size": 4,
                       "gpu": "Tesla T4", "wer": 0, "cer": 0, "word_substitutions": 0,
                       "word_insertions": 0, "word_deletions": 0, "inference_seconds": 3437, "rtf": 1,
                       "checkpoint_tensor_sha256": "e30c130f25e3ab092d0d7b1a09ba45629ad5e66add405f194e496216c9946383"}
            (root / key / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            comparison.append({"model": model, "status": "complete", "wer_percent": 0, "cer_percent": 0, "rtf": 1})
        (root / "comparison.json").write_text(json.dumps({"models": comparison}), encoding="utf-8")
        return ref_hash

    def test_validates_full_three_model_result_and_rejects_wrong_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference_hash = self.make_fixture(root)
            with patch.object(collector, "REFERENCE_SHA", reference_hash):
                self.assertEqual(collector.verify(root)["status"], "verified")
                path = root / "whisper" / "predictions.jsonl"
                rows = collector.read_rows(path)
                rows[3]["audio_sha256"] = "different-clip"
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                    collector.verify(root)

    def test_rejects_score_that_does_not_match_transcripts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference_hash = self.make_fixture(root)
            path = root / "phowhisper" / "summary.json"
            summary = json.loads(path.read_text())
            summary["wer"] = 0.1
            path.write_text(json.dumps(summary), encoding="utf-8")
            with patch.object(collector, "REFERENCE_SHA", reference_hash):
                with self.assertRaisesRegex(ValueError, "metrics differ"):
                    collector.verify(root)


if __name__ == "__main__":
    unittest.main()
