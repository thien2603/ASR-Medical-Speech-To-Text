import unittest

from asr_common.text import normalize_vi, text_from_hypothesis


class NormalizeTests(unittest.TestCase):
    def test_vietnamese_diacritics_and_digits_are_preserved(self):
        self.assertEqual(normalize_vi("  Bệnh ÁN: số 123! "), "bệnh án số 123")

    def test_punctuation_and_whitespace_are_normalized(self):
        self.assertEqual(normalize_vi("Xin-chào,\n bác sĩ."), "xin chào bác sĩ")

    def test_plain_hypothesis(self):
        self.assertEqual(text_from_hypothesis("xin chào"), "xin chào")


if __name__ == "__main__":
    unittest.main()
