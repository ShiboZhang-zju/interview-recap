import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.analysis import AnalysisError
from src.config import load_config
from src.pipeline import run_pipeline
from src.session import PIPELINE_STAGES, SessionError, default_session_id


ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def transcribe(self, audio_path: str | Path) -> dict:
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic ASR failure")
        return {
            "schema_version": "0.1.0",
            "created_at": "2026-08-18T00:00:00+00:00",
            "source_audio": None,
            "runtime": {},
            "models": {},
            "result": [
                {
                    "sentence_info": [
                        {
                            "text": "请介绍一下缓存方案？",
                            "start": 0,
                            "end": 3000,
                            "spk": 0,
                        },
                        {
                            "text": "我使用版本号处理缓存失效。",
                            "start": 3200,
                            "end": 7000,
                            "spk": 1,
                        },
                    ]
                }
            ],
        }


def fake_preprocess(config, input_path, output_path=None, overwrite=False):
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"synthetic wav")
    return target, {
        "source_path": str(Path(input_path).resolve()),
        "output_path": str(target),
        "reused": False,
    }


class SessionPipelineTests(unittest.TestCase):
    def make_config(self, directory: str) -> dict:
        config = copy.deepcopy(load_config(ROOT / "config.yaml"))
        config["_root"] = Path(directory).resolve()
        return config

    def test_default_session_id_is_stable_and_path_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "我的 面试.wav"
            source.write_bytes(b"audio")
            first = default_session_id(source)
            second = default_session_id(source)
            self.assertEqual(first, second)
            self.assertNotIn(" ", first)
            self.assertTrue(first.startswith("我的-面试-"))

    @mock.patch("src.pipeline.preprocess_audio", side_effect=fake_preprocess)
    def test_full_run_writes_manifest_and_resume_reuses_every_stage(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interview.wav"
            source.write_bytes(b"audio")
            config = self.make_config(directory)
            engine = FakeEngine()

            first = run_pipeline(config, source, engine=engine, session_id="resume-case")
            second = run_pipeline(
                config,
                source,
                engine=engine,
                session_id="resume-case",
                resume=True,
            )

            self.assertEqual(engine.calls, 1)
            self.assertEqual(preprocess.call_count, 1)
            self.assertTrue(first["report_markdown"].is_file())
            self.assertEqual(second["analysis_provider"], "none")
            manifest = json.loads(second["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            for stage in PIPELINE_STAGES:
                self.assertEqual(manifest["stages"][stage]["status"], "completed")
                self.assertTrue(manifest["stages"][stage]["reused"])

            config["analysis"]["maximum_answer_chars"] = 1200
            third = run_pipeline(
                config,
                source,
                engine=engine,
                session_id="resume-case",
                resume=True,
            )
            changed = json.loads(third["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(engine.calls, 1)
            self.assertEqual(preprocess.call_count, 1)
            self.assertTrue(changed["stages"]["repair"]["reused"])
            self.assertFalse(changed["stages"]["analyze"]["reused"])

            config["audio"]["sample_rate"] = 8000
            fourth = run_pipeline(
                config,
                source,
                engine=engine,
                session_id="resume-case",
                resume=True,
            )
            audio_changed = json.loads(fourth["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(engine.calls, 2)
            self.assertEqual(preprocess.call_count, 2)
            self.assertTrue(preprocess.call_args.kwargs["overwrite"])
            self.assertFalse(audio_changed["stages"]["preprocess"]["reused"])

            with self.assertRaisesRegex(SessionError, "--resume"):
                run_pipeline(config, source, engine=engine, session_id="resume-case")

    @mock.patch("src.pipeline.preprocess_audio", side_effect=fake_preprocess)
    def test_failed_stage_is_recorded_and_can_resume(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interview.wav"
            source.write_bytes(b"audio")
            config = self.make_config(directory)

            with self.assertRaisesRegex(RuntimeError, "synthetic ASR failure"):
                run_pipeline(
                    config,
                    source,
                    engine=FakeEngine(fail=True),
                    session_id="failure-case",
                )
            manifest_path = Path(directory) / "work/sessions/failure-case/manifest.json"
            failed = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["stages"]["transcribe"]["status"], "failed")

            good_engine = FakeEngine()
            result = run_pipeline(
                config,
                source,
                engine=good_engine,
                session_id="failure-case",
                resume=True,
            )
            self.assertEqual(good_engine.calls, 1)
            self.assertEqual(preprocess.call_count, 1)
            resumed = json.loads(result["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(resumed["status"], "completed")

    @mock.patch("src.pipeline.preprocess_audio", side_effect=fake_preprocess)
    def test_overwrite_recovers_from_corrupt_manifest(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interview.wav"
            source.write_bytes(b"audio")
            manifest = Path(directory) / "work/sessions/corrupt-case/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("not json", encoding="utf-8")

            result = run_pipeline(
                self.make_config(directory),
                source,
                overwrite=True,
                engine=FakeEngine(),
                session_id="corrupt-case",
            )

            repaired = json.loads(result["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(repaired["status"], "completed")
            self.assertEqual(preprocess.call_count, 1)

    @mock.patch("src.pipeline.preprocess_audio", side_effect=fake_preprocess)
    def test_resume_rejects_non_object_manifest(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interview.wav"
            source.write_bytes(b"audio")
            manifest = Path(directory) / "work/sessions/invalid-case/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("[]", encoding="utf-8")

            with self.assertRaisesRegex(SessionError, "顶层必须是 object"):
                run_pipeline(
                    self.make_config(directory),
                    source,
                    resume=True,
                    engine=FakeEngine(),
                    session_id="invalid-case",
                )

            preprocess.assert_not_called()

    @mock.patch("src.pipeline.preprocess_audio", side_effect=fake_preprocess)
    def test_remote_analysis_is_rejected_before_expensive_stages(self, preprocess) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "interview.wav"
            source.write_bytes(b"audio")
            config = self.make_config(directory)

            with self.assertRaisesRegex(AnalysisError, "--allow-remote"):
                run_pipeline(
                    config,
                    source,
                    engine=FakeEngine(),
                    session_id="remote-without-consent",
                    analysis_provider_name="openai-compatible",
                    analysis_model="test-model",
                )

            preprocess.assert_not_called()
            manifest = Path(directory) / "work/sessions/remote-without-consent/manifest.json"
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
