from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .config import project_path


CANDIDATE_VOICE_FILTER = "aresample=16000,asetrate=12800,aresample=16000,atempo=1.25"

SMOKE_UTTERANCES = (
    ("Tingting", "请介绍一下 Redis 的持久化机制，以及 RDB 和 AOF 的区别？", None),
    # Pitch-shifting a second copy of the natural Mandarin voice keeps the
    # words ASR-friendly while giving CAM++ a stable, distinct timbre.
    ("Tingting", "Redis 支持 RDB 快照和 AOF 日志。RDB 恢复快，AOF 数据更完整。", CANDIDATE_VOICE_FILTER),
    ("Tingting", "如果 AOF 文件越来越大，你会怎么处理？", None),
    ("Tingting", "可以执行 AOF 重写。后台生成新的日志文件，最后合并新的写入。", CANDIDATE_VOICE_FILTER),
)


def make_smoke_audio(config: dict[str, Any], overwrite: bool = False) -> Path:
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        raise RuntimeError("生成 smoke 音频需要 macOS say 和 ffmpeg")
    target = project_path(config, config["project"]["input_dir"]) / "smoke_technical_interview.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return target

    with tempfile.TemporaryDirectory(prefix="interview-smoke-") as temp_dir:
        temp = Path(temp_dir)
        clips: list[Path] = []
        for index, (voice, text, audio_filter) in enumerate(SMOKE_UTTERANCES):
            aiff = temp / f"speech-{index}.aiff"
            clip = temp / f"speech-{index}.wav"
            subprocess.run(["say", "-v", voice, "-r", "170", "-o", str(aiff), text], check=True)
            # `say` voices may emit different sample rates/channel layouts. The
            # concat demuxer requires identical stream parameters, so normalize
            # every utterance before joining it with silence.
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(aiff)]
            if audio_filter:
                command.extend(["-af", audio_filter])
            command.extend(["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(clip)])
            subprocess.run(command, check=True)
            clips.append(clip)
        silence = temp / "silence.wav"
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1.5", str(silence),
            ],
            check=True,
        )
        concat_file = temp / "concat.txt"
        concat_lines: list[str] = []
        for clip in clips:
            concat_lines.extend([f"file '{clip}'", f"file '{silence}'"])
        concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target),
            ],
            check=True,
        )
    return target
