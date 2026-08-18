import copy
import unittest

from src.config import load_config
from src.qa import infer_roles, segment_qa


class QASegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.transcript = {
            "speakers": [{"id": "speaker_0", "role": "unknown"}, {"id": "speaker_1", "role": "unknown"}],
            "segments": [
                {"id": "seg_0001", "start": 0.0, "end": 2.0, "speaker": "speaker_0", "role": "unknown", "text": "请介绍 Redis 持久化的区别？"},
                {"id": "seg_0002", "start": 2.1, "end": 5.0, "speaker": "speaker_1", "role": "unknown", "text": "RDB 是快照，AOF 是日志。"},
                {"id": "seg_0003", "start": 5.1, "end": 7.0, "speaker": "speaker_0", "role": "unknown", "text": "如果 AOF 很大怎么处理？"},
                {"id": "seg_0004", "start": 7.1, "end": 9.0, "speaker": "speaker_1", "role": "unknown", "text": "可以后台重写。"},
                {"id": "seg_0005", "start": 9.1, "end": 11.0, "speaker": "speaker_0", "role": "unknown", "text": "下一个问题，TCP 为什么要三次握手？"},
                {"id": "seg_0006", "start": 11.1, "end": 14.0, "speaker": "speaker_1", "role": "unknown", "text": "为了确认双方收发能力。"},
            ],
        }

    def test_role_inference(self) -> None:
        roles = infer_roles(self.transcript["segments"])
        self.assertEqual(roles["speaker_0"], "interviewer")
        self.assertEqual(roles["speaker_1"], "candidate")

    def test_follow_up_and_new_topic(self) -> None:
        result = segment_qa(self.config, copy.deepcopy(self.transcript))
        self.assertEqual(len(result["qa_pairs"]), 2)
        self.assertEqual(len(result["qa_pairs"][0]["follow_ups"]), 1)
        self.assertEqual(result["qa_pairs"][0]["candidate_answers"][1]["for_question"], "follow_up_1")
        self.assertEqual(result["qa_pairs"][1]["topic"]["label"], "网络")

    def test_contiguous_same_speaker_segments_are_one_turn(self) -> None:
        transcript = {
            "speakers": copy.deepcopy(self.transcript["speakers"]),
            "segments": [
                {"id": "seg_0001", "start": 0.0, "end": 1.0, "speaker": "speaker_0", "role": "unknown", "text": "请介绍 Redis 持久化机制，"},
                {"id": "seg_0002", "start": 1.0, "end": 2.0, "speaker": "speaker_0", "role": "unknown", "text": "以及 RDB 和 AOF 的区别？"},
                {"id": "seg_0003", "start": 2.2, "end": 4.0, "speaker": "speaker_1", "role": "unknown", "text": "RDB 是快照，AOF 是日志。"},
                {"id": "seg_0004", "start": 4.2, "end": 5.0, "speaker": "speaker_0", "role": "unknown", "text": "如果 AOF 文件越来越大，"},
                {"id": "seg_0005", "start": 5.0, "end": 6.0, "speaker": "speaker_0", "role": "unknown", "text": "你会怎么处理？"},
                {"id": "seg_0006", "start": 6.2, "end": 7.0, "speaker": "speaker_1", "role": "unknown", "text": "可以后台重写。"},
            ],
        }

        result = segment_qa(self.config, transcript)

        self.assertEqual(len(result["qa_pairs"]), 1)
        qa = result["qa_pairs"][0]
        self.assertEqual(qa["main_question"]["segment_ids"], ["seg_0001", "seg_0002"])
        self.assertEqual(qa["follow_ups"][0]["segment_ids"], ["seg_0004", "seg_0005"])
        self.assertEqual(qa["candidate_answers"][0]["for_question"], "main")
        self.assertEqual(qa["candidate_answers"][1]["for_question"], "follow_up_1")
        self.assertEqual(qa["segment_ids"], [f"seg_{index:04d}" for index in range(1, 7)])


if __name__ == "__main__":
    unittest.main()
