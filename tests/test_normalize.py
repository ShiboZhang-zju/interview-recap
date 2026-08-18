import unittest

from src.normalize import normalize_text


class NormalizeTextTests(unittest.TestCase):
    def test_ascii_terms_are_case_insensitive_and_bounded(self) -> None:
        replacements = {"redis": "Redis", "http": "HTTP", "https": "HTTPS"}
        self.assertEqual(normalize_text("redis 和 HTTPS", replacements), "Redis 和 HTTPS")

    def test_chinese_replacement_and_whitespace(self) -> None:
        replacements = {"瑞迪斯": "Redis"}
        self.assertEqual(normalize_text("  瑞迪斯   持久化 。 ", replacements), "Redis 持久化。")


if __name__ == "__main__":
    unittest.main()
