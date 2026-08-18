import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.funasr_engine import resolve_cached_model


class ModelCacheResolutionTests(unittest.TestCase):
    def test_prefers_modelscope_master_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = (
                Path(temp_dir)
                / "models"
                / "iic--speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
                / "snapshots"
                / "master"
            )
            snapshot.mkdir(parents=True)
            (snapshot / "config.yaml").write_text("model: test\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"MODELSCOPE_CACHE": temp_dir}):
                resolved = resolve_cached_model("paraformer-zh")
            self.assertEqual(resolved, str(snapshot.resolve()))

    def test_missing_snapshot_keeps_remote_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(os.environ, {"MODELSCOPE_CACHE": temp_dir}):
                resolved = resolve_cached_model("fsmn-vad")
            self.assertEqual(resolved, "fsmn-vad")


if __name__ == "__main__":
    unittest.main()
