from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio import probe_audio
from .config import display_path, project_path
from .normalize import load_replacements, normalize_text


def milliseconds_to_seconds(value: Any) -> float:
    return round(float(value or 0) / 1000.0, 3)


def _first_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") or []
    while isinstance(result, list) and result:
        result = result[0]
    return result if isinstance(result, dict) else {}


def _speaker_name(raw: Any) -> str:
    if raw is None or str(raw).strip() == "":
        return "speaker_unknown"
    label = str(raw).strip()
    if label.startswith("speaker_"):
        return label
    return f"speaker_{label}"


def _fallback_range(item: dict[str, Any], result: dict[str, Any]) -> tuple[float, float]:
    timestamp = item.get("timestamp") or result.get("timestamp") or []
    if timestamp and isinstance(timestamp[0], (list, tuple)):
        return milliseconds_to_seconds(timestamp[0][0]), milliseconds_to_seconds(timestamp[-1][-1])
    return 0.0, 0.0


def build_segments(payload: dict[str, Any], replacements: dict[str, str]) -> list[dict[str, Any]]:
    result = _first_result(payload)
    sentences = result.get("sentence_info") or []
    if not sentences and result.get("text"):
        sentences = [{"text": result["text"], "timestamp": result.get("timestamp"), "spk": None}]

    segments: list[dict[str, Any]] = []
    for index, item in enumerate(sentences, start=1):
        fallback_start, fallback_end = _fallback_range(item, result)
        start = milliseconds_to_seconds(item["start"]) if item.get("start") is not None else fallback_start
        end = milliseconds_to_seconds(item["end"]) if item.get("end") is not None else fallback_end
        text = normalize_text(str(item.get("text") or ""), replacements)
        if not text:
            continue
        segments.append(
            {
                "id": f"seg_{index:04d}",
                "start": start,
                "end": max(start, end),
                "speaker": _speaker_name(item.get("spk")),
                "role": "unknown",
                "text": text,
            }
        )
    return segments


def structured_output_path(config: dict[str, Any], raw_path: str | Path) -> Path:
    name = Path(raw_path).name
    stem = name[: -len(".funasr.json")] if name.endswith(".funasr.json") else Path(name).stem
    return project_path(config, config["project"]["structured_dir"]) / f"{stem}.json"


def build_structured(config: dict[str, Any], raw_payload: dict[str, Any]) -> dict[str, Any]:
    replacements = load_replacements(config)
    segments = build_segments(raw_payload, replacements)
    source_audio = raw_payload.get("source_audio")
    source_path = project_path(config, source_audio) if source_audio else None
    audio_metadata = probe_audio(source_path) if source_path and source_path.exists() else None
    speakers = sorted({segment["speaker"] for segment in segments})
    return {
        "schema_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_audio": source_audio,
        "audio": audio_metadata,
        "pipeline": {
            "runtime": raw_payload.get("runtime", {}),
            "models": raw_payload.get("models", {}),
            "raw_output_preserved": True,
            "timestamp_unit": "seconds",
        },
        "speakers": [{"id": speaker, "role": "unknown"} for speaker in speakers],
        "segments": segments,
        "qa_pairs": [],
    }


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def render_markdown(transcript: dict[str, Any]) -> str:
    lines = ["# 技术面试转写", ""]
    lines.append(f"- 音频：`{transcript.get('source_audio') or 'unknown'}`")
    lines.append(f"- 说话人：{len(transcript.get('speakers', []))}")
    lines.append(f"- Segments：{len(transcript.get('segments', []))}")
    lines.extend(["", "## 完整转写", ""])
    for segment in transcript.get("segments", []):
        stamp = f"{format_timestamp(segment['start'])}–{format_timestamp(segment['end'])}"
        role = segment.get("role", "unknown")
        lines.append(f"**[{stamp}] {segment['speaker']} ({role})**  ")
        lines.append(segment["text"])
        lines.append("")

    qa_pairs = transcript.get("qa_pairs") or []
    if qa_pairs:
        lines.extend(["## Q&A 初分段", ""])
        for qa in qa_pairs:
            topic = qa.get("topic", {})
            lines.append(f"### {qa['id']} · {topic.get('label', '未分类')}")
            lines.append("")
            lines.append(f"- 与上一题关系：{topic.get('relation_to_previous', 'unknown')}")
            lines.append(f"- 主问题：{qa['main_question']['text']}")
            for follow_up in qa.get("follow_ups", []):
                lines.append(f"- Follow-up：{follow_up['text']}")
            for answer in qa.get("candidate_answers", []):
                lines.append(f"- Candidate answer ({answer['for_question']})：{answer['text']}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_structured(transcript: dict[str, Any], json_path: str | Path) -> tuple[Path, Path]:
    path = Path(json_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path = path.with_suffix(".md")
    markdown_path.write_text(render_markdown(transcript), encoding="utf-8")
    return path, markdown_path
