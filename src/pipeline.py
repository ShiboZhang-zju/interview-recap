from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audio import SUPPORTED_EXTENSIONS, preprocess_audio
from .config import display_path
from .conversation_v2 import repair_conversation, repair_file, save_repaired, v2_output_path
from .funasr_engine import FunASREngine, raw_output_path, save_raw
from .qa import segment_qa
from .transcript import build_structured, save_structured, structured_output_path


def transcribe_stage(
    config: dict[str, Any], audio_path: str | Path, engine: FunASREngine | None = None
) -> tuple[dict[str, Any], Path]:
    active_engine = engine or FunASREngine(config)
    raw_payload = active_engine.transcribe(audio_path)
    path = save_raw(raw_payload, raw_output_path(config, audio_path))
    return raw_payload, path


def structure_stage(config: dict[str, Any], raw_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    raw = Path(raw_path).expanduser().resolve()
    payload = json.loads(raw.read_text(encoding="utf-8"))
    structured = build_structured(config, payload)
    json_path, markdown_path = save_structured(structured, structured_output_path(config, raw))
    return structured, json_path, markdown_path


def segment_stage(config: dict[str, Any], structured_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    path = Path(structured_path).expanduser().resolve()
    transcript = json.loads(path.read_text(encoding="utf-8"))
    transcript = segment_qa(config, transcript)
    json_path, markdown_path = save_structured(transcript, path)
    return transcript, json_path, markdown_path


def repair_stage(config: dict[str, Any], structured_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    return repair_file(config, structured_path)


def run_pipeline(
    config: dict[str, Any],
    input_path: str | Path,
    overwrite: bool = False,
    engine: FunASREngine | None = None,
) -> dict[str, Any]:
    prepared, audio_metadata = preprocess_audio(config, input_path, overwrite=overwrite)
    raw_payload, raw_path = transcribe_stage(config, prepared, engine=engine)
    structured = build_structured(config, raw_payload)
    structured["preprocessing"] = audio_metadata
    structured = segment_qa(config, structured)
    json_path, markdown_path = save_structured(structured, structured_output_path(config, raw_path))
    repaired = repair_conversation(config, structured)
    repaired["derived_from"]["structured_path"] = display_path(config, json_path)
    v2_json_path, v2_markdown_path = save_repaired(repaired, v2_output_path(json_path))
    return {
        "prepared_audio": prepared,
        "raw_json": raw_path,
        "structured_json": json_path,
        "structured_markdown": markdown_path,
        "structured_v2_json": v2_json_path,
        "structured_v2_markdown": v2_markdown_path,
        "segments": len(structured["segments"]),
        "qa_pairs": len(structured["qa_pairs"]),
        "v2_valid_segments": repaired["statistics"]["valid_segment_count"],
        "v2_main_questions": repaired["statistics"]["main_question_count"],
    }


def discover_audio(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
