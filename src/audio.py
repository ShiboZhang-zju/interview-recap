from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import display_path, project_path


SUPPORTED_EXTENSIONS = {".m4a", ".mp3", ".wav"}


class AudioError(RuntimeError):
    pass


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, text=True, capture_output=True)
    except FileNotFoundError as exc:
        raise AudioError(f"找不到命令: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise AudioError(f"音频命令失败: {detail}") from exc


def probe_audio(path: str | Path) -> dict[str, Any]:
    audio_path = Path(path).expanduser().resolve()
    result = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,format_name,size:stream=index,codec_name,sample_rate,channels,channel_layout",
            "-select_streams",
            "a:0",
            "-of",
            "json",
            str(audio_path),
        ]
    )
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    return {
        "duration": float(fmt["duration"]) if fmt.get("duration") else None,
        "format": fmt.get("format_name"),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else audio_path.stat().st_size,
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": stream.get("channels"),
        "channel_layout": stream.get("channel_layout"),
    }


def preprocessed_path(config: dict[str, Any], input_path: str | Path) -> Path:
    source = Path(input_path)
    directory = project_path(config, config["project"]["work_audio_dir"])
    return directory / f"{source.stem}.16k.wav"


def preprocess_audio(
    config: dict[str, Any],
    input_path: str | Path,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, dict[str, Any]]:
    source = Path(input_path).expanduser().resolve()
    if not source.is_file():
        raise AudioError(f"输入音频不存在: {source}")
    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(f"暂不支持 {source.suffix}；支持 m4a/mp3/wav")

    target = Path(output_path).expanduser().resolve() if output_path else preprocessed_path(config, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite and target.stat().st_mtime >= source.stat().st_mtime:
        return target, {"source": probe_audio(source), "output": probe_audio(target), "reused": True}

    audio = config["audio"]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    filters = [str(item) for item in audio.get("filters", []) if str(item).strip()]
    if filters:
        command.extend(["-af", ",".join(filters)])
    command.extend(
        [
            "-vn",
            "-ac",
            str(audio["channels"]),
            "-ar",
            str(audio["sample_rate"]),
            "-c:a",
            str(audio["codec"]),
            str(target),
        ]
    )
    _run(command)
    metadata = {
        "source_path": display_path(config, source),
        "output_path": display_path(config, target),
        "source": probe_audio(source),
        "output": probe_audio(target),
        "filters": filters,
        "reused": False,
    }
    return target, metadata


def check_audio_tools() -> dict[str, Any]:
    return {
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "say": shutil.which("say"),
    }
