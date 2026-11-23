"""Checks that comparison tables cannot publish misleading benchmark numbers."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.datasets import benchmark


class MedicalBenchmarkTests(unittest.TestCase):
    def test_wer_is_corpus_weighted_not_mean_sentence_wer(self):
        result = benchmark.metrics([
            {"reference": "thuốc", "prediction_normalized": ""},
            {"reference": "một hai ba bốn năm sáu bảy tám chín mười",
             "prediction_normalized": "một hai ba bốn năm sáu bảy tám chín mười"},
        ])
        self.assertAlmostEqual(result["wer"], 1 / 11)
        self.assertEqual(result["word_deletions"], 1)
        self.assertEqual(result["reference_words"], 11)

    def test_pending_rows_do_not_reuse_old_subset_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "dataset.json").write_text(json.dumps({
                "samples": 3437, "audio_seconds": 21680, "manifest_sha256": "frozen",
            }), encoding="utf-8")
            (output / "manifest.jsonl").write_text("", encoding="utf-8")
            with patch.object(benchmark, "OUTPUT", output), patch("builtins.print"):
                benchmark.report()
            rows = json.loads((output / "comparison.json").read_text(encoding="utf-8"))["models"]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["wer_percent"] is None for row in rows))

    def test_report_rejects_incomplete_test_run(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            (output / "dataset.json").write_text(json.dumps({
                "samples": 3437, "audio_seconds": 21680, "manifest_sha256": "frozen",
            }), encoding="utf-8")
            (output / "manifest.jsonl").write_text("", encoding="utf-8")
            (output / "kyle").mkdir()
            (output / "kyle" / "summary.json").write_text(json.dumps({
                "samples": 100, "manifest_sha256": "frozen",
            }), encoding="utf-8")
            with patch.object(benchmark, "OUTPUT", output):
                with self.assertRaisesRegex(RuntimeError, "Incomplete or mismatched"):
                    benchmark.report()


if __name__ == "__main__":
    unittest.main()
