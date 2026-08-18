import unittest

from src.transcript import build_segments, format_timestamp, milliseconds_to_seconds


class TranscriptTests(unittest.TestCase):
    def test_sentence_info_becomes_required_segment_schema(self) -> None:
        raw = {
            "result": [
                {
                    "text": "你好。",
                    "sentence_info": [
                        {"start": 120, "end": 2340, "spk": 1, "text": "你好。", "timestamp": [[120, 500]]}
                    ],
                }
            ]
        }
        segments = build_segments(raw, {})
        self.assertEqual(segments[0]["start"], 0.12)
        self.assertEqual(segments[0]["end"], 2.34)
        self.assertEqual(segments[0]["speaker"], "speaker_1")
        self.assertEqual(segments[0]["text"], "你好。")

    def test_timestamp_format(self) -> None:
        self.assertEqual(milliseconds_to_seconds(1234), 1.234)
        self.assertEqual(format_timestamp(61.234), "00:01:01.234")


if __name__ == "__main__":
    unittest.main()
