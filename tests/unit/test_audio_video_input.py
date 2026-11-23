import unittest
import numpy as np

from asr_common.text import normalize_vi
from evaluation.audio_video.prepare_input import clean_subtitle, boundaries


class PodcastInputTests(unittest.TestCase):
    def test_cleanup_keeps_medical_negation_numbers_and_disfluencies(self):
        raw = '[âm nhạc]\n>> Không không có máu; 50%\nthì chưa đủ.\n>> À, thuốc 20 mg.\n'
        result = clean_subtitle(raw)
        self.assertEqual(normalize_vi(result), normalize_vi('Không không có máu; 50% thì chưa đủ. À, thuốc 20 mg.'))
        self.assertNotIn('[âm nhạc]', result)
        self.assertNotIn('>>', result)

    def test_chunks_cover_audio_once_with_bounded_lengths(self):
        audio = np.zeros(53 * 16000 + 7, dtype=np.float32)
        cuts = boundaries(audio)
        self.assertEqual(cuts[0], 0)
        self.assertEqual(cuts[-1], len(audio))
        lengths = np.diff(cuts)
        self.assertTrue(np.all(lengths > 0))
        self.assertTrue(np.all(lengths <= 24 * 16000))
        self.assertEqual(int(lengths.sum()), len(audio))

    def test_short_clip_is_not_truncated_or_split(self):
        self.assertEqual(boundaries(np.zeros(1000, dtype=np.float32)), [0, 1000])


if __name__ == '__main__':
    unittest.main()
