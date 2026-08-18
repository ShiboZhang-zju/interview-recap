from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .audio import check_audio_tools, preprocess_audio
from .config import DEFAULT_CONFIG, display_path, ensure_project_dirs, load_config
from .funasr_engine import FunASREngine
from .pipeline import (
    discover_audio,
    repair_stage,
    run_pipeline,
    segment_stage,
    structure_stage,
    transcribe_stage,
)
from .smoke import make_smoke_audio


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_report() -> dict[str, Any]:
    tools = check_audio_tools()
    packages = {
        "funasr": _package_version("funasr"),
        "torch": _package_version("torch"),
        "torchaudio": _package_version("torchaudio"),
        "PyYAML": _package_version("PyYAML"),
        "soundfile": _package_version("soundfile"),
    }
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "tools": tools,
        "packages": packages,
        "checks": {
            "apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64",
            "python_supported": sys.version_info >= (3, 10) and sys.version_info < (3, 13),
            "ffmpeg_available": bool(tools["ffmpeg"] and tools["ffprobe"]),
            "funasr_installed": bool(packages["funasr"]),
        },
    }
    if packages["torch"]:
        import torch

        report["torch"] = {
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        }
    return report


def _print_paths(config: dict[str, Any], result: dict[str, Any]) -> None:
    printable = {
        key: display_path(config, value) if isinstance(value, Path) else value for key, value in result.items()
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="本地技术面试录音复盘 pipeline")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.yaml 路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("env-check", help="检查 Python、FFmpeg 与模型依赖")

    preprocess = subparsers.add_parser("preprocess", help="音频转 16 kHz mono WAV")
    preprocess.add_argument("input")
    preprocess.add_argument("--output")
    preprocess.add_argument("--overwrite", action="store_true")

    transcribe = subparsers.add_parser("transcribe", help="运行 FunASR/VAD/标点/说话人")
    transcribe.add_argument("audio")

    structure = subparsers.add_parser("structure", help="FunASR raw JSON 转 structured transcript")
    structure.add_argument("raw_json")

    segment = subparsers.add_parser("segment", help="对 structured transcript 做 Q&A 初分段")
    segment.add_argument("structured_json")

    repair = subparsers.add_parser("repair-v2", help="生成 V2 对话结构派生结果，不修改 V1")
    repair.add_argument("structured_json")

    run = subparsers.add_parser("run", help="对单个音频运行完整 pipeline")
    run.add_argument("input")
    run.add_argument("--overwrite", action="store_true")

    batch = subparsers.add_parser("batch", help="批量处理目录中的 m4a/mp3/wav")
    batch.add_argument("directory", nargs="?", default="input")
    batch.add_argument("--overwrite", action="store_true")

    smoke = subparsers.add_parser("make-smoke-audio", help="生成双说话人技术问答合成音频")
    smoke.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_project_dirs(config)

    if args.command == "env-check":
        print(json.dumps(environment_report(), ensure_ascii=False, indent=2))
    elif args.command == "preprocess":
        target, metadata = preprocess_audio(config, args.input, args.output, args.overwrite)
        _print_paths(config, {"output": target, "metadata": metadata})
    elif args.command == "transcribe":
        payload, path = transcribe_stage(config, args.audio)
        _print_paths(config, {"raw_json": path, "result_items": len(payload.get("result") or [])})
    elif args.command == "structure":
        transcript, json_path, markdown_path = structure_stage(config, args.raw_json)
        _print_paths(
            config,
            {"structured_json": json_path, "markdown": markdown_path, "segments": len(transcript["segments"])},
        )
    elif args.command == "segment":
        transcript, json_path, markdown_path = segment_stage(config, args.structured_json)
        _print_paths(
            config,
            {"structured_json": json_path, "markdown": markdown_path, "qa_pairs": len(transcript["qa_pairs"])},
        )
    elif args.command == "repair-v2":
        transcript, json_path, markdown_path = repair_stage(config, args.structured_json)
        stats = transcript["statistics"]
        _print_paths(
            config,
            {
                "structured_v2_json": json_path,
                "structured_v2_markdown": markdown_path,
                "valid_segments": stats["valid_segment_count"],
                "main_questions": stats["main_question_count"],
                "follow_ups": stats["follow_up_count"],
                "coding_events": stats["coding_event_count"],
            },
        )
    elif args.command == "run":
        _print_paths(config, run_pipeline(config, args.input, args.overwrite))
    elif args.command == "batch":
        paths = discover_audio(args.directory)
        if not paths:
            raise SystemExit(f"目录中没有 m4a/mp3/wav: {args.directory}")
        engine = FunASREngine(config)
        results = [run_pipeline(config, path, args.overwrite, engine=engine) for path in paths]
        for result in results:
            _print_paths(config, result)
    elif args.command == "make-smoke-audio":
        path = make_smoke_audio(config, args.overwrite)
        _print_paths(config, {"smoke_audio": path})


if __name__ == "__main__":
    main()
