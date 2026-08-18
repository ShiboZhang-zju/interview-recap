import copy
import json
import unittest
from pathlib import Path

from src.analysis import (
    AnalysisError,
    AnalysisProvider,
    NoneProvider,
    create_provider,
    normalize_provider_result,
    run_analysis,
)
from src.report import render_analysis_report


ROOT = Path(__file__).resolve().parents[1]


def load_conversation() -> dict:
    return json.loads((ROOT / "examples/synthetic_interview.v2.json").read_text(encoding="utf-8"))


class CapturingRemoteProvider(AnalysisProvider):
    name = "openai-compatible"
    remote = True

    def __init__(self) -> None:
        super().__init__(model="test-model", endpoint="https://example.invalid/v1/chat/completions")
        self.payload = None
        self.payloads = []

    def analyze(self, payload: dict) -> dict:
        self.payload = payload
        self.payloads.append(payload)
        questions = []
        for question in payload["questions"]:
            evidence = list(question["question_segment_ids"])
            for answer in question["answers"]:
                evidence.extend(answer["segment_ids"])
            questions.append(
                {
                    "question_id": question["question_id"],
                    "summary": "回答与问题相关。",
                    "scores": {
                        "relevance": 4,
                        "technical_depth": 3,
                        "clarity": 4,
                        "structure": 3,
                    },
                    "strengths": ["给出了具体方案"],
                    "risks": [],
                    "suggestions": ["补充权衡"],
                    "improved_answer": "先说明目标，再说明方案与权衡。",
                    "evidence_segment_ids": list(dict.fromkeys(evidence)),
                }
            )
        return {
            "overview": {
                "summary": "整体回答可读。",
                "strengths": ["有具体例子"],
                "risks": [],
                "suggested_focus": ["技术权衡"],
            },
            "questions": questions,
            "review_plan": [
                {
                    "priority": "high",
                    "topic": "缓存",
                    "action": "复习一致性策略",
                    "question_ids": [payload["questions"][0]["question_id"]]
                    if payload["questions"]
                    else [],
                }
            ],
            "warnings": [],
        }


class AnalysisTests(unittest.TestCase):
    def test_local_ollama_provider_does_not_require_remote_consent(self) -> None:
        import yaml

        config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
        provider = create_provider(config, provider_name="ollama", model="qwen-test")
        self.assertFalse(provider.remote)
        self.assertEqual(provider.endpoint, "http://127.0.0.1:11434/api/chat")

        config["analysis"]["providers"]["ollama"]["base_url"] = "http://model.example.com"
        with self.assertRaisesRegex(AnalysisError, "HTTPS"):
            create_provider(config, provider_name="ollama", model="qwen-test")

        config["analysis"]["providers"]["ollama"]["base_url"] = "ftp://127.0.0.1"
        with self.assertRaisesRegex(AnalysisError, "HTTP"):
            create_provider(config, provider_name="ollama", model="qwen-test")

    def test_none_provider_produces_schema_valid_traceable_analysis(self) -> None:
        conversation = load_conversation()
        analysis = run_analysis(conversation, NoneProvider())

        self.assertEqual(analysis["provider"]["name"], "none")
        self.assertFalse(analysis["provider"]["remote"])
        self.assertFalse(analysis["privacy"]["sent_audio"])
        self.assertFalse(analysis["privacy"]["sent_full_transcript"])
        self.assertEqual(len(analysis["questions"]), 2)
        self.assertEqual(analysis["questions"][0]["assessment_status"], "not_evaluated")
        self.assertIn("seg_0001", analysis["questions"][0]["evidence_segment_ids"])

        try:
            import jsonschema
        except ImportError:
            return
        schema = json.loads((ROOT / "schemas/analysis-v1.schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(instance=analysis, schema=schema)

    def test_remote_provider_requires_consent_and_receives_only_redacted_tree(self) -> None:
        conversation = load_conversation()
        conversation = copy.deepcopy(conversation)
        conversation["source_audio"] = "/private/interview.wav"
        conversation["segments"][0]["text"] += " 联系邮箱 candidate@example.com"
        conversation["question_tree"][0]["text"] += " 联系邮箱 candidate@example.com"
        conversation["question_tree"][0]["answers"][0]["text"] += " 手机 13800138000"
        provider = CapturingRemoteProvider()

        with self.assertRaisesRegex(AnalysisError, "--allow-remote"):
            run_analysis(conversation, provider)
        self.assertIsNone(provider.payload)

        analysis = run_analysis(conversation, provider, allow_remote=True)
        serialized = json.dumps(provider.payload, ensure_ascii=False)
        self.assertNotIn("candidate@example.com", serialized)
        self.assertNotIn("13800138000", serialized)
        self.assertNotIn("source_audio", provider.payload)
        self.assertNotIn("segments", provider.payload)
        self.assertEqual(analysis["privacy"]["redaction_counts"]["email"], 1)
        self.assertEqual(analysis["privacy"]["redaction_counts"]["phone"], 1)
        self.assertTrue(analysis["privacy"]["remote_consent"])

    def test_large_question_tree_is_batched_and_reassembled(self) -> None:
        conversation = load_conversation()
        provider = CapturingRemoteProvider()
        analysis = run_analysis(
            conversation,
            provider,
            allow_remote=True,
            maximum_questions_per_request=1,
        )

        self.assertEqual(len(provider.payloads), 2)
        self.assertEqual(analysis["source"]["request_count"], 2)
        self.assertEqual([item["question_id"] for item in analysis["questions"]], ["Q1", "Q1.1"])
        self.assertIn("Semantic analysis used 2 requests.", analysis["warnings"])

    def test_provider_cannot_reference_unknown_segment(self) -> None:
        conversation = load_conversation()
        provider = CapturingRemoteProvider()
        raw = provider.analyze(
            {
                "questions": [
                    {
                        "question_id": "Q1",
                        "question_segment_ids": ["seg_0001"],
                        "answers": [],
                    },
                    {
                        "question_id": "Q1.1",
                        "question_segment_ids": ["seg_0003"],
                        "answers": [],
                    },
                ]
            }
        )
        raw["questions"][0]["evidence_segment_ids"] = ["seg_not_real"]
        with self.assertRaisesRegex(AnalysisError, "不属于该问答"):
            normalize_provider_result(raw, conversation, evaluated=True)

    def test_report_layout_is_deterministic_and_keeps_source_ids(self) -> None:
        conversation = load_conversation()
        analysis = run_analysis(conversation, NoneProvider())
        report = render_analysis_report(conversation, analysis)

        self.assertIn("# Interview Recap", report)
        self.assertIn("### Q1 · 数据库与缓存", report)
        self.assertIn("为什么？", report)
        self.assertIn("seg_0004", report)
        self.assertIn("发送音频：False", report)

    def test_report_escapes_html_from_transcript_and_provider(self) -> None:
        conversation = load_conversation()
        conversation["question_tree"][0]["text"] = "<script>alert(1)</script>"
        analysis = run_analysis(conversation, NoneProvider())
        analysis["overview"]["summary"] = "<img src=x onerror=alert(1)>"
        report = render_analysis_report(conversation, analysis)

        self.assertNotIn("<script>", report)
        self.assertNotIn("<img", report)
        self.assertIn("&lt;script&gt;", report)
        self.assertIn("&lt;img", report)

    def test_report_escapes_markdown_from_provider(self) -> None:
        conversation = load_conversation()
        analysis = run_analysis(conversation, NoneProvider())
        analysis["overview"]["summary"] = "[click](javascript:alert(1))\n# injected"
        analysis["overview"]["strengths"] = ["**untrusted**"]
        report = render_analysis_report(conversation, analysis)

        self.assertNotIn("[click](javascript:alert(1))", report)
        self.assertNotIn("\n# injected", report)
        self.assertIn(r"\[click\](javascript:alert(1))<br># injected", report)
        self.assertIn(r"\*\*untrusted\*\*", report)


if __name__ == "__main__":
    unittest.main()
