import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from src.cli import build_parser, environment_report, main, render_environment_report


ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_run_parser_accepts_resume_and_analysis_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "input/interview.wav",
                "--resume",
                "--session-id",
                "demo-session",
                "--analysis-provider",
                "openai-compatible",
                "--analysis-model",
                "test-model",
                "--allow-remote",
            ]
        )
        self.assertTrue(args.resume)
        self.assertEqual(args.session_id, "demo-session")
        self.assertEqual(args.analysis_provider, "openai-compatible")
        self.assertTrue(args.allow_remote)

    def test_environment_report_gives_actionable_install_command(self) -> None:
        with (
            mock.patch(
                "src.cli.check_audio_tools",
                return_value={"ffmpeg": None, "ffprobe": None, "say": None},
            ),
            mock.patch("src.cli._package_version", return_value=None),
            mock.patch(
                "src.cli.shutil.disk_usage",
                return_value=SimpleNamespace(free=20 * 1024**3),
            ),
        ):
            report = environment_report()

        self.assertFalse(report["ready_for_asr"])
        self.assertTrue(any("FFmpeg" in item for item in report["recommendations"]))
        self.assertTrue(any(".[asr]" in item for item in report["recommendations"]))
        rendered = render_environment_report(report)
        self.assertIn("ACTION_REQUIRED", rendered)
        self.assertIn("Recommendations:", rendered)

    def test_expected_cli_error_is_concise_without_traceback(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(SystemExit, "2"):
            main(
                [
                    "analyze",
                    str(ROOT / "examples/synthetic_interview.v2.json"),
                    "--provider",
                    "openai-compatible",
                    "--model",
                    "test-model",
                ]
            )
        message = stderr.getvalue()
        self.assertIn("--allow-remote", message)
        self.assertNotIn("Traceback", message)

    def test_environment_report_survives_broken_torch_import(self) -> None:
        def package_version(name: str) -> str | None:
            return "2.2.0" if name == "torch" else None

        with (
            mock.patch(
                "src.cli.check_audio_tools",
                return_value={"ffmpeg": None, "ffprobe": None, "say": None},
            ),
            mock.patch("src.cli._package_version", side_effect=package_version),
            mock.patch(
                "src.cli.shutil.disk_usage",
                return_value=SimpleNamespace(free=20 * 1024**3),
            ),
            mock.patch.dict(sys.modules, {"torch": None}),
        ):
            report = environment_report()

        self.assertIn("import_error", report["torch"])
        self.assertTrue(any("无法导入" in item for item in report["recommendations"]))


if __name__ == "__main__":
    unittest.main()
