import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from src.config import (
    BUNDLED_CONFIG,
    BUNDLED_RESOURCE_ROOT,
    SOURCE_CONFIG,
    load_config,
    project_path,
)


ROOT = Path(__file__).resolve().parents[1]


class PublicPackageTests(unittest.TestCase):
    def test_bundled_config_matches_repository_defaults(self) -> None:
        source = yaml.safe_load(SOURCE_CONFIG.read_text(encoding="utf-8"))
        bundled = yaml.safe_load(BUNDLED_CONFIG.read_text(encoding="utf-8"))

        self.assertEqual(source, bundled)

    def test_wheel_defaults_write_to_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            working_directory = Path(directory).resolve()
            with mock.patch("src.config.Path.cwd", return_value=working_directory):
                config = load_config(BUNDLED_CONFIG)

            self.assertEqual(config["_root"], working_directory)
            self.assertEqual(
                project_path(config, config["knowledge"]["topics_file"]),
                BUNDLED_RESOURCE_ROOT / "knowledge/topics.yaml",
            )

    def test_synthetic_example_is_valid_v2_json(self) -> None:
        example = json.loads(
            (ROOT / "examples/synthetic_interview.v2.json").read_text(encoding="utf-8")
        )
        schema = json.loads(
            (ROOT / "schemas/conversation-v2.schema.json").read_text(encoding="utf-8")
        )

        self.assertEqual(example["schema_version"], "0.2.0")
        self.assertEqual(schema["properties"]["schema_version"]["const"], "0.2.0")
        self.assertEqual(example["statistics"]["question_count"], 2)
        self.assertEqual(example["question_tree"][0]["children"][0]["id"], "Q1.1")

        try:
            import jsonschema
        except ImportError:
            return
        jsonschema.validate(instance=example, schema=schema)


if __name__ == "__main__":
    unittest.main()
