from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any

from .config import project_path


MANIFEST_SCHEMA_VERSION = "0.1.1"
PIPELINE_STAGES = ("preprocess", "transcribe", "structure", "repair", "analyze")


class SessionError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_stem(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._\u4e00-\u9fff-]+", "-", value).strip("-._")
    return normalized[:64] or "interview"


def default_session_id(input_path: str | Path) -> str:
    source = Path(input_path).expanduser().resolve()
    path_digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{_safe_stem(source.stem)}-{path_digest}"


def _input_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


class PipelineSession:
    """Durable state for a resumable, private local pipeline run."""

    def __init__(
        self,
        config: dict[str, Any],
        input_path: str | Path,
        *,
        session_id: str | None = None,
        resume: bool = False,
        overwrite: bool = False,
    ) -> None:
        if resume and overwrite:
            raise SessionError("--resume 和 --overwrite 不能同时使用")

        self.input_path = Path(input_path).expanduser().resolve()
        if not self.input_path.is_file():
            raise SessionError(f"输入音频不存在: {self.input_path}")
        self.session_id = session_id or default_session_id(self.input_path)
        if self.session_id != _safe_stem(self.session_id):
            raise SessionError("session id 只能包含字母、数字、中文、点、下划线和连字符")

        sessions_dir = config.get("project", {}).get("sessions_dir", "work/sessions")
        self.root = project_path(config, sessions_dir) / self.session_id
        self.manifest_path = self.root / "manifest.json"
        self.resume = resume
        self.overwrite = overwrite
        self._started: dict[str, float] = {}

        existing = None if overwrite else self._load_existing()
        identity = _input_identity(self.input_path)
        if existing and not (resume or overwrite):
            raise SessionError(
                f"会话已存在: {self.root}；使用 --resume 继续，或使用 --overwrite 重新运行"
            )
        if existing and resume:
            if existing.get("input") != identity:
                raise SessionError("输入文件在会话创建后发生变化；请使用 --overwrite 重新运行")
            self.manifest = existing
            self.manifest["status"] = "running"
            self.manifest["updated_at"] = _now()
        else:
            created_at = _now()
            self.manifest = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "session_id": self.session_id,
                "created_at": created_at,
                "updated_at": created_at,
                "status": "running",
                "input": identity,
                "stages": {stage: {"status": "pending"} for stage in PIPELINE_STAGES},
            }
        self.root.mkdir(parents=True, exist_ok=True)
        self._save()

    def _load_existing(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionError(f"无法读取会话 manifest: {self.manifest_path}") from exc
        if not isinstance(value, dict):
            raise SessionError(f"会话 manifest 顶层必须是 object: {self.manifest_path}")
        if value.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise SessionError(
                f"不支持的 manifest schema: {value.get('schema_version') or 'unknown'}"
            )
        return value

    def _save(self) -> None:
        self.manifest["updated_at"] = _now()
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    def path(self, directory: str, filename: str) -> Path:
        path = self.root / directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def relative_path(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(self.root))
        except ValueError:
            return str(resolved)

    def outputs(self, stage: str) -> list[Path]:
        values = self.manifest["stages"].get(stage, {}).get("outputs", [])
        return [
            (self.root / value).resolve() if not Path(value).is_absolute() else Path(value)
            for value in values
        ]

    def can_reuse(self, stage: str, signature: str | None = None) -> bool:
        if not self.resume:
            return False
        state = self.manifest["stages"].get(stage, {})
        outputs = self.outputs(stage)
        signature_matches = signature is None or state.get("signature") == signature
        return (
            state.get("status") == "completed"
            and signature_matches
            and bool(outputs)
            and all(path.is_file() for path in outputs)
        )

    def mark_reused(self, stage: str) -> None:
        state = self.manifest["stages"][stage]
        state["reused"] = True
        state["last_reused_at"] = _now()
        self._save()

    def start_stage(self, stage: str) -> None:
        if stage not in PIPELINE_STAGES:
            raise SessionError(f"未知 pipeline stage: {stage}")
        stage_index = PIPELINE_STAGES.index(stage)
        for downstream in PIPELINE_STAGES[stage_index + 1 :]:
            state = self.manifest["stages"].get(downstream, {})
            if state.get("status") != "pending":
                self.manifest["stages"][downstream] = {
                    "status": "pending",
                    "invalidated_by": stage,
                }
        self._started[stage] = monotonic()
        self.manifest["stages"][stage] = {
            "status": "running",
            "started_at": _now(),
            "reused": False,
        }
        self._save()

    def complete_stage(
        self,
        stage: str,
        outputs: list[str | Path],
        metadata: dict[str, Any] | None = None,
        signature: str | None = None,
    ) -> None:
        elapsed = monotonic() - self._started.pop(stage, monotonic())
        state: dict[str, Any] = {
            "status": "completed",
            "completed_at": _now(),
            "duration_seconds": round(max(0.0, elapsed), 3),
            "reused": False,
            "outputs": [self.relative_path(path) for path in outputs],
        }
        if metadata:
            state["metadata"] = metadata
        if signature:
            state["signature"] = signature
        self.manifest["stages"][stage] = state
        self._save()

    def fail_stage(self, stage: str, exc: BaseException) -> None:
        elapsed = monotonic() - self._started.pop(stage, monotonic())
        self.manifest["stages"][stage] = {
            "status": "failed",
            "failed_at": _now(),
            "duration_seconds": round(max(0.0, elapsed), 3),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        self.manifest["status"] = "failed"
        self._save()

    def complete(self) -> None:
        self.manifest["status"] = "completed"
        self.manifest["completed_at"] = _now()
        self._save()
