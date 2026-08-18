import copy
import unittest

from src.config import load_config
from src.conversation_v2 import analyze_segment, repair_conversation


def segment(
    index: int,
    start: float,
    end: float,
    speaker: str,
    text: str,
    role: str = "unknown",
) -> dict:
    return {
        "id": f"seg_{index:04d}",
        "start": start,
        "end": end,
        "speaker": speaker,
        "role": role,
        "text": text,
    }


class SegmentCleanupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_punctuation_and_long_low_density_are_invalid(self) -> None:
        punctuation = analyze_segment(segment(1, 0, 2, "speaker_0", "。，，。"), self.config)
        sparse = analyze_segment(segment(2, 0, 40, "speaker_0", "小问题？"), self.config)

        self.assertFalse(punctuation["valid"])
        self.assertIn("pure_or_repeated_punctuation", punctuation["reasons"])
        self.assertFalse(sparse["valid"])
        self.assertIn("long_duration_low_text_density", sparse["reasons"])

    def test_real_short_follow_up_is_preserved(self) -> None:
        for text in ("为什么？", "怎么做？", "复杂度呢？"):
            result = analyze_segment(segment(1, 0, 2, "speaker_0", text), self.config)
            self.assertTrue(result["valid"], text)

    def test_repetitive_noise_is_invalid(self) -> None:
        result = analyze_segment(
            segment(1, 0, 4, "speaker_0", "哒哒哒哒哒哒哒哒哒"), self.config
        )
        self.assertFalse(result["valid"])
        self.assertIn("repetitive_noise_transcription", result["reasons"])


class ConversationRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def _transcript(self, segments: list[dict], duration: float = 400.0) -> dict:
        return {
            "schema_version": "0.1.0",
            "source_audio": "work/audio/test.wav",
            "audio": {"duration": duration},
            "pipeline": {"raw_output_preserved": True},
            "speakers": [
                {"id": "speaker_0", "role": "interviewer"},
                {"id": "speaker_1", "role": "candidate"},
            ],
            "segments": segments,
            "qa_pairs": [{"id": "qa_001"}, {"id": "qa_002"}],
        }

    def test_role_is_inferred_per_segment_not_fixed_by_speaker(self) -> None:
        source = self._transcript(
            [
                segment(1, 60, 63, "speaker_0", "请你介绍一下项目？", "interviewer"),
                segment(
                    2,
                    63.2,
                    72,
                    "speaker_1",
                    "我负责系统设计和性能优化，最后把延迟降低了一半。",
                    "candidate",
                ),
                segment(3, 72.2, 74, "speaker_1", "那为什么选择这个方案？", "candidate"),
                segment(4, 74.2, 80, "speaker_0", "我当时比较了三种方案。", "interviewer"),
            ]
        )

        result = repair_conversation(self.config, source)
        by_id = {item["id"]: item for item in result["segments"]}

        self.assertEqual(by_id["seg_0003"]["speaker_id"], "speaker_1")
        self.assertEqual(by_id["seg_0003"]["role"], "interviewer")
        self.assertEqual(by_id["seg_0004"]["role"], "candidate")
        self.assertGreaterEqual(result["statistics"]["role_correction_count"], 2)

    def test_question_tree_keeps_follow_up_traceability(self) -> None:
        source = self._transcript(
            [
                segment(1, 60, 63, "speaker_0", "请你介绍一下项目？", "interviewer"),
                segment(2, 63.2, 70, "speaker_1", "我负责缓存模块。", "candidate"),
                segment(3, 70.2, 72, "speaker_0", "为什么？", "interviewer"),
                segment(4, 72.2, 76, "speaker_1", "因为读流量很大。", "candidate"),
            ]
        )
        original = copy.deepcopy(source)

        result = repair_conversation(self.config, source)

        self.assertEqual(source, original)
        self.assertEqual(len(result["question_tree"]), 1)
        root = result["question_tree"][0]
        self.assertEqual(root["id"], "Q1")
        self.assertEqual(root["children"][0]["id"], "Q1.1")
        self.assertEqual(root["children"][0]["node_type"], "follow_up")
        self.assertEqual(root["children"][0]["segment_ids"], ["seg_0003"])
        self.assertEqual(root["children"][0]["answers"][0]["segment_ids"], ["seg_0004"])

    def test_coding_silence_is_event_not_question(self) -> None:
        source = self._transcript(
            [
                segment(1, 200, 204, "speaker_0", "下面开始做题，代码用 Python。", "interviewer"),
                segment(2, 204.2, 207, "speaker_1", "好的，我先看一下题目。", "candidate"),
                segment(3, 220, 222, "speaker_0", "复杂度呢？", "interviewer"),
                segment(4, 222.2, 225, "speaker_1", "时间复杂度是线性的。", "candidate"),
            ],
            duration=300,
        )

        result = repair_conversation(self.config, source)
        events = result["events"]

        self.assertTrue(any(event["event_type"] == "coding_silence" for event in events))
        questions = [
            node
            for root in result["question_tree"]
            for node in [root, *root["children"]]
        ]
        self.assertTrue(any("复杂度" in node["text"] for node in questions))
        self.assertFalse(any("silence" in node["text"].lower() for node in questions))

    def test_clarification_is_a_child_node(self) -> None:
        source = self._transcript(
            [
                segment(1, 60, 63, "speaker_0", "请介绍一下索引更新方案？", "interviewer"),
                segment(2, 63.2, 68, "speaker_1", "我们采用增量更新。", "candidate"),
                segment(3, 68.2, 71, "speaker_0", "你是说只更新变化的索引吗？", "interviewer"),
                segment(4, 71.2, 74, "speaker_1", "对，不需要全量重建。", "candidate"),
            ]
        )

        result = repair_conversation(self.config, source)

        self.assertEqual(result["question_tree"][0]["children"][0]["node_type"], "clarification")

    def test_candidate_closing_is_not_a_question(self) -> None:
        source = self._transcript(
            [
                segment(1, 250, 253, "speaker_1", "我想问一下这个岗位具体做什么？", "candidate"),
                segment(2, 253.2, 260, "speaker_0", "主要负责搜索系统。", "interviewer"),
                segment(3, 270, 273, "speaker_1", "我这边没什么问题了。", "candidate"),
            ],
            duration=300,
        )

        result = repair_conversation(self.config, source)

        self.assertEqual(result["statistics"]["question_count"], 1)
        self.assertEqual(result["question_tree"][0]["asked_by"], "candidate")

    def test_invalid_segments_remain_traceable(self) -> None:
        source = self._transcript(
            [
                segment(1, 0, 1, "speaker_0", "。，，。", "interviewer"),
                segment(2, 2, 4, "speaker_0", "为什么？", "interviewer"),
            ]
        )

        result = repair_conversation(self.config, source)
        by_id = {item["id"]: item for item in result["segments"]}

        self.assertIn("seg_0001", by_id)
        self.assertFalse(by_id["seg_0001"]["cleanup"]["valid"])
        self.assertEqual(by_id["seg_0001"]["role"], "unknown")
        self.assertEqual(result["statistics"]["unknown_role_count"], 1)


if __name__ == "__main__":
    unittest.main()
