from __future__ import annotations

import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import display_path, project_path


class FunASRError(RuntimeError):
    pass


MODELSCOPE_MODEL_ALIASES = {
    "paraformer-zh": "iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "fsmn-vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "ct-punc-c": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "cam++": "iic/speech_campplus_sv_zh-cn_16k-common",
}


def _is_model_snapshot(path: Path) -> bool:
    return path.is_dir() and any(
        (path / filename).is_file() for filename in ("config.yaml", "configuration.json")
    )


def resolve_cached_model(reference: str, hub: str = "ms") -> str:
    """Prefer a complete ModelScope snapshot so cached models work offline."""
    configured = Path(reference).expanduser()
    if configured.exists():
        return str(configured.resolve())
    if hub != "ms":
        return reference

    repo_id = MODELSCOPE_MODEL_ALIASES.get(reference, reference)
    if "/" not in repo_id:
        return reference
    cache_root = Path(
        os.environ.get("MODELSCOPE_CACHE", str(Path.home() / ".cache" / "modelscope"))
    ).expanduser()
    safe_id = repo_id.replace("/", "--")
    repository_dirs = (
        cache_root / "models" / safe_id,
        cache_root / "models" / repo_id,
        cache_root / "hub" / repo_id,
        cache_root / repo_id,
    )
    for repository in repository_dirs:
        master = repository / "snapshots" / "master"
        if _is_model_snapshot(master):
            return str(master.resolve())
        snapshots = repository / "snapshots"
        if snapshots.is_dir():
            candidates = sorted(
                (path for path in snapshots.iterdir() if _is_model_snapshot(path)),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return str(candidates[0].resolve())
        if _is_model_snapshot(repository):
            return str(repository.resolve())
    return reference


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def load_hotwords(config: dict[str, Any]) -> list[str]:
    configured = config["inference"].get("hotwords_file")
    if not configured:
        return []
    path = project_path(config, configured)
    if not path.exists():
        return []
    words: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if word and not word.startswith("#"):
            words.append(word)
    return words


class FunASREngine:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._model: Any = None
        self._local_cache_used = False

    def load(self) -> None:
        if self._model is not None:
            return
        models = self.config["models"]
        # ModelScope reads these settings at import time. setdefault preserves any
        # explicit choice made by an advanced user in their shell environment.
        os.environ.setdefault(
            "MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS", str(models.get("download_workers", 8))
        )
        os.environ.setdefault(
            "MODELSCOPE_DOWNLOAD_PART_SIZE_MB", str(models.get("download_part_size_mb", 64))
        )
        os.environ.setdefault(
            "MODELSCOPE_DOWNLOAD_PARALLEL_THRESHOLD_MB",
            str(models.get("download_parallel_threshold_mb", 50)),
        )
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise FunASRError('未安装 FunASR；请先运行 python -m pip install -e ".[asr]"') from exc

        hub = models.get("hub", "ms")
        configured_models = {
            "asr": str(models["asr"]),
            "vad": str(models["vad"]),
            "punctuation": str(models["punctuation"]),
            "speaker": str(models["speaker"]),
        }
        if models.get("prefer_local_cache", True):
            resolved_models = {
                name: resolve_cached_model(reference, hub)
                for name, reference in configured_models.items()
            }
        else:
            resolved_models = configured_models
        self._local_cache_used = any(
            resolved_models[name] != configured_models[name] for name in configured_models
        )

        kwargs: dict[str, Any] = {
            "model": resolved_models["asr"],
            "vad_model": resolved_models["vad"],
            "vad_kwargs": {"max_single_segment_time": int(models["vad_max_segment_ms"])},
            "punc_model": resolved_models["punctuation"],
            "spk_model": resolved_models["speaker"],
            "device": models.get("device", "cpu"),
            "hub": hub,
            "disable_update": True,
        }
        try:
            self._model = AutoModel(**kwargs)
        except Exception as exc:  # model hubs raise several exception types
            raise FunASRError(f"FunASR 模型加载失败: {exc}") from exc

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        self.load()
        source = Path(audio_path).expanduser().resolve()
        inference = self.config["inference"]
        generate_kwargs: dict[str, Any] = {
            "input": str(source),
            "batch_size_s": int(inference.get("batch_size_s", 60)),
            "batch_size_threshold_s": int(inference.get("batch_size_threshold_s", 30)),
            "merge_vad": bool(inference.get("merge_vad", True)),
            "merge_length_s": int(inference.get("merge_length_s", 15)),
        }
        hotwords = load_hotwords(self.config)
        if hotwords:
            generate_kwargs["hotword"] = " ".join(hotwords)
        speaker_count = inference.get("speaker_count")
        if speaker_count is not None:
            generate_kwargs["preset_spk_num"] = int(speaker_count)

        try:
            result = self._model.generate(**generate_kwargs)
        except Exception as exc:
            raise FunASRError(f"FunASR 推理失败: {exc}") from exc

        models = self.config["models"]
        return {
            "schema_version": "0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_audio": display_path(self.config, source),
            "runtime": {
                "funasr": importlib.metadata.version("funasr"),
                "torch": importlib.metadata.version("torch"),
                "device": models.get("device", "cpu"),
                "hub": models.get("hub", "ms"),
                "local_cache_used": self._local_cache_used,
            },
            "models": {
                "asr": models["asr"],
                "vad": models["vad"],
                "punctuation": models["punctuation"],
                "speaker": models["speaker"],
            },
            "parameters": {
                "batch_size_s": generate_kwargs["batch_size_s"],
                "batch_size_threshold_s": generate_kwargs["batch_size_threshold_s"],
                "merge_vad": generate_kwargs["merge_vad"],
                "merge_length_s": generate_kwargs["merge_length_s"],
                "speaker_count": speaker_count,
                "hotwords": hotwords,
            },
            "result": _jsonable(result),
        }


def raw_output_path(config: dict[str, Any], audio_path: str | Path) -> Path:
    stem = Path(audio_path).name
    if stem.endswith(".16k.wav"):
        stem = stem[: -len(".16k.wav")]
    else:
        stem = Path(stem).stem
    return project_path(config, config["project"]["raw_dir"]) / f"{stem}.funasr.json"


def save_raw(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
