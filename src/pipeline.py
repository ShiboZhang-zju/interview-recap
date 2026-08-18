from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .analysis import (
    AnalysisError,
    create_provider,
    run_analysis,
    save_analysis,
    validate_provider_configuration,
)
from .audio import SUPPORTED_EXTENSIONS, preprocess_audio
from .config import display_path, project_path
from .conversation_v2 import repair_conversation, repair_file, save_repaired
from .funasr_engine import FunASREngine, raw_output_path, save_raw
from .qa import segment_qa
from .report import save_analysis_report
from .session import PipelineSession
from .transcript import build_structured, save_structured, structured_output_path


ProgressCallback = Callable[[str, str, dict[str, Any]], None]


def _notify(
    callback: ProgressCallback | None,
    stage: str,
    event: str,
    **details: Any,
) -> None:
    if callback:
        callback(stage, event, details)


def _signature(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _knowledge_signature(config: dict[str, Any], key: str) -> dict[str, Any]:
    configured = config.get("knowledge", {}).get(key)
    if not configured:
        return {"path": None, "sha256": None}
    path = project_path(config, configured)
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {"path": str(configured), "sha256": digest}


def _analysis_redaction_terms(config: dict[str, Any]) -> list[str]:
    settings = config.get("analysis", {})
    terms = [str(item).strip() for item in settings.get("custom_redaction_terms", [])]
    configured_file = settings.get("redaction_terms_file")
    if configured_file:
        path = project_path(config, configured_file)
        if not path.is_file():
            raise AnalysisError(f"redaction terms 文件不存在: {path}")
        terms.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return list(dict.fromkeys(term for term in terms if term))


def transcribe_stage(
    config: dict[str, Any],
    audio_path: str | Path,
    engine: FunASREngine | None = None,
    output_path: str | Path | None = None,
) -> tuple[dict[str, Any], Path]:
    active_engine = engine or FunASREngine(config)
    raw_payload = active_engine.transcribe(audio_path)
    path = save_raw(raw_payload, output_path or raw_output_path(config, audio_path))
    return raw_payload, path


def structure_stage(
    config: dict[str, Any], raw_path: str | Path
) -> tuple[dict[str, Any], Path, Path]:
    raw = Path(raw_path).expanduser().resolve()
    payload = json.loads(raw.read_text(encoding="utf-8"))
    structured = build_structured(config, payload)
    json_path, markdown_path = save_structured(structured, structured_output_path(config, raw))
    return structured, json_path, markdown_path


def segment_stage(
    config: dict[str, Any], structured_path: str | Path
) -> tuple[dict[str, Any], Path, Path]:
    path = Path(structured_path).expanduser().resolve()
    transcript = json.loads(path.read_text(encoding="utf-8"))
    transcript = segment_qa(config, transcript)
    json_path, markdown_path = save_structured(transcript, path)
    return transcript, json_path, markdown_path


def repair_stage(
    config: dict[str, Any], structured_path: str | Path
) -> tuple[dict[str, Any], Path, Path]:
    return repair_file(config, structured_path)


def analysis_stage(
    config: dict[str, Any],
    conversation_path: str | Path,
    *,
    provider_name: str | None = None,
    model: str | None = None,
    allow_remote: bool = False,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    source = Path(conversation_path).expanduser().resolve()
    conversation = json.loads(source.read_text(encoding="utf-8"))
    provider = create_provider(config, provider_name=provider_name, model=model)
    settings = config.get("analysis", {})
    validate_provider_configuration(
        provider,
        allow_remote=allow_remote,
        redact=bool(settings.get("redact", True)),
        require_http_client=True,
    )
    analysis = run_analysis(
        conversation,
        provider,
        allow_remote=allow_remote,
        redact=bool(settings.get("redact", True)),
        custom_redaction_terms=_analysis_redaction_terms(config),
        maximum_answer_chars=int(settings.get("maximum_answer_chars", 2400)),
        maximum_payload_chars=int(settings.get("maximum_payload_chars", 160000)),
        maximum_questions_per_request=int(settings.get("maximum_questions_per_request", 12)),
    )
    if output_dir is None:
        directory = project_path(config, config["project"]["reports_dir"])
        stem = source.name[: -len(".v2.json")] if source.name.endswith(".v2.json") else source.stem
        analysis_path = directory / f"{stem}.analysis.json"
        report_path = directory / f"{stem}.report.md"
    else:
        directory = Path(output_dir).expanduser().resolve()
        analysis_path = directory / "analysis.json"
        report_path = directory / "report.md"
    save_analysis(analysis, analysis_path)
    save_analysis_report(conversation, analysis, report_path)
    return analysis, analysis_path, report_path


def run_pipeline(
    config: dict[str, Any],
    input_path: str | Path,
    overwrite: bool = False,
    engine: FunASREngine | None = None,
    *,
    resume: bool = False,
    session_id: str | None = None,
    progress: ProgressCallback | None = None,
    analysis_provider_name: str | None = None,
    analysis_model: str | None = None,
    allow_remote_analysis: bool = False,
) -> dict[str, Any]:
    redaction_terms = _analysis_redaction_terms(config)
    analysis_settings = config.get("analysis", {})
    analysis_provider = create_provider(
        config, provider_name=analysis_provider_name, model=analysis_model
    )
    validate_provider_configuration(
        analysis_provider,
        allow_remote=allow_remote_analysis,
        redact=bool(analysis_settings.get("redact", True)),
        require_http_client=True,
    )
    session = PipelineSession(
        config,
        input_path,
        session_id=session_id,
        resume=resume,
        overwrite=overwrite,
    )
    prepared_path = session.path("audio", "interview.16k.wav")
    raw_path = session.path("raw", "interview.funasr.json")
    json_path = session.path("structured", "interview.json")
    markdown_path = json_path.with_suffix(".md")
    v2_json_path = session.path("structured", "interview.v2.json")
    v2_markdown_path = v2_json_path.with_suffix(".md")
    analysis_path = session.path("analysis", "analysis.json")
    report_path = session.path("reports", "report.md")
    stage_signatures = {
        "preprocess": _signature(config.get("audio", {})),
        "transcribe": _signature(
            {"models": config.get("models", {}), "inference": config.get("inference", {})}
        ),
        "structure": _signature(
            {
                "qa": config.get("qa", {}),
                "normalization": _knowledge_signature(config, "normalization_file"),
                "topics": _knowledge_signature(config, "topics_file"),
            }
        ),
        "repair": _signature(
            {
                "conversation_v2": config.get("conversation_v2", {}),
                "topics": _knowledge_signature(config, "topics_file"),
            }
        ),
        "analyze": _signature(
            {
                "analysis": config.get("analysis", {}),
                "redaction_terms": redaction_terms,
                "provider": analysis_provider.name,
                "model": analysis_provider.model,
                "allow_remote": allow_remote_analysis,
            }
        ),
    }
    upstream_reran = False

    stage = "preprocess"
    try:
        if not upstream_reran and session.can_reuse(stage, stage_signatures[stage]):
            session.mark_reused(stage)
            prepared = session.outputs(stage)[0]
            audio_metadata = session.manifest["stages"][stage].get("metadata", {})
            _notify(progress, stage, "reused", output=prepared)
        else:
            upstream_reran = True
            session.start_stage(stage)
            _notify(progress, stage, "started", input=Path(input_path).expanduser().resolve())
            prepared, audio_metadata = preprocess_audio(
                config,
                input_path,
                output_path=prepared_path,
                # The session manifest already decided that this stage cannot be
                # reused. Always replace its private output so a changed audio
                # configuration is actually applied during --resume.
                overwrite=True,
            )
            session.complete_stage(
                stage, [prepared], audio_metadata, signature=stage_signatures[stage]
            )
            _notify(progress, stage, "completed", output=prepared)
    except Exception as exc:
        session.fail_stage(stage, exc)
        _notify(progress, stage, "failed", error=str(exc))
        raise

    stage = "transcribe"
    try:
        if not upstream_reran and session.can_reuse(stage, stage_signatures[stage]):
            session.mark_reused(stage)
            raw_path = session.outputs(stage)[0]
            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            _notify(progress, stage, "reused", output=raw_path)
        else:
            upstream_reran = True
            session.start_stage(stage)
            _notify(progress, stage, "started", input=prepared)
            raw_payload, raw_path = transcribe_stage(
                config, prepared, engine=engine, output_path=raw_path
            )
            session.complete_stage(
                stage,
                [raw_path],
                {"result_items": len(raw_payload.get("result") or [])},
                signature=stage_signatures[stage],
            )
            _notify(progress, stage, "completed", output=raw_path)
    except Exception as exc:
        session.fail_stage(stage, exc)
        _notify(progress, stage, "failed", error=str(exc))
        raise

    stage = "structure"
    try:
        if not upstream_reran and session.can_reuse(stage, stage_signatures[stage]):
            session.mark_reused(stage)
            outputs = session.outputs(stage)
            json_path = outputs[0]
            markdown_path = outputs[1]
            structured = json.loads(json_path.read_text(encoding="utf-8"))
            _notify(progress, stage, "reused", output=json_path)
        else:
            upstream_reran = True
            session.start_stage(stage)
            _notify(progress, stage, "started", input=raw_path)
            structured = build_structured(config, raw_payload)
            structured["preprocessing"] = audio_metadata
            structured = segment_qa(config, structured)
            json_path, markdown_path = save_structured(structured, json_path)
            session.complete_stage(
                stage,
                [json_path, markdown_path],
                {
                    "segments": len(structured["segments"]),
                    "qa_pairs": len(structured["qa_pairs"]),
                },
                signature=stage_signatures[stage],
            )
            _notify(progress, stage, "completed", output=json_path)
    except Exception as exc:
        session.fail_stage(stage, exc)
        _notify(progress, stage, "failed", error=str(exc))
        raise

    stage = "repair"
    try:
        if not upstream_reran and session.can_reuse(stage, stage_signatures[stage]):
            session.mark_reused(stage)
            outputs = session.outputs(stage)
            v2_json_path = outputs[0]
            v2_markdown_path = outputs[1]
            repaired = json.loads(v2_json_path.read_text(encoding="utf-8"))
            _notify(progress, stage, "reused", output=v2_json_path)
        else:
            upstream_reran = True
            session.start_stage(stage)
            _notify(progress, stage, "started", input=json_path)
            repaired = repair_conversation(config, structured)
            repaired["derived_from"]["structured_path"] = display_path(config, json_path)
            v2_json_path, v2_markdown_path = save_repaired(repaired, v2_json_path)
            session.complete_stage(
                stage,
                [v2_json_path, v2_markdown_path],
                repaired["statistics"],
                signature=stage_signatures[stage],
            )
            _notify(progress, stage, "completed", output=v2_json_path)
    except Exception as exc:
        session.fail_stage(stage, exc)
        _notify(progress, stage, "failed", error=str(exc))
        raise

    stage = "analyze"
    try:
        if not upstream_reran and session.can_reuse(stage, stage_signatures[stage]):
            session.mark_reused(stage)
            outputs = session.outputs(stage)
            analysis_path = outputs[0]
            report_path = outputs[1]
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            _notify(progress, stage, "reused", output=analysis_path)
        else:
            upstream_reran = True
            session.start_stage(stage)
            _notify(progress, stage, "started", input=v2_json_path)
            settings = config.get("analysis", {})
            analysis = run_analysis(
                repaired,
                analysis_provider,
                allow_remote=allow_remote_analysis,
                redact=bool(settings.get("redact", True)),
                custom_redaction_terms=redaction_terms,
                maximum_answer_chars=int(settings.get("maximum_answer_chars", 2400)),
                maximum_payload_chars=int(settings.get("maximum_payload_chars", 160000)),
                maximum_questions_per_request=int(
                    settings.get("maximum_questions_per_request", 12)
                ),
            )
            save_analysis(analysis, analysis_path)
            save_analysis_report(repaired, analysis, report_path)
            session.complete_stage(
                stage,
                [analysis_path, report_path],
                {
                    "provider": analysis["provider"]["name"],
                    "remote": analysis["provider"]["remote"],
                    "question_count": len(analysis["questions"]),
                },
                signature=stage_signatures[stage],
            )
            _notify(progress, stage, "completed", output=report_path)
    except Exception as exc:
        session.fail_stage(stage, exc)
        _notify(progress, stage, "failed", error=str(exc))
        raise

    session.complete()
    return {
        "session_id": session.session_id,
        "session_dir": session.root,
        "manifest": session.manifest_path,
        "prepared_audio": prepared,
        "raw_json": raw_path,
        "structured_json": json_path,
        "structured_markdown": markdown_path,
        "structured_v2_json": v2_json_path,
        "structured_v2_markdown": v2_markdown_path,
        "analysis_json": analysis_path,
        "report_markdown": report_path,
        "analysis_provider": analysis["provider"]["name"],
        "segments": len(structured["segments"]),
        "qa_pairs": len(structured["qa_pairs"]),
        "v2_valid_segments": repaired["statistics"]["valid_segment_count"],
        "v2_main_questions": repaired["statistics"]["main_question_count"],
    }


def discover_audio(directory: str | Path) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
