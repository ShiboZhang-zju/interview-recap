from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import AnalysisError
from .audio import AudioError, check_audio_tools, preprocess_audio
from .config import DEFAULT_CONFIG, display_path, ensure_project_dirs, load_config
from .funasr_engine import FunASREngine, FunASRError, resolve_cached_model
from .pipeline import (
    analysis_stage,
    discover_audio,
    repair_stage,
    run_pipeline,
    segment_stage,
    structure_stage,
    transcribe_stage,
)
from .smoke import make_smoke_audio
from .session import SessionError


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _major_version(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value.split(".", 1)[0])
    except ValueError:
        return None


def environment_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    tools = check_audio_tools()
    packages = {
        "funasr": _package_version("funasr"),
        "torch": _package_version("torch"),
        "torchaudio": _package_version("torchaudio"),
        "PyYAML": _package_version("PyYAML"),
        "soundfile": _package_version("soundfile"),
        "httpx": _package_version("httpx"),
        "setuptools": _package_version("setuptools"),
    }
    disk = shutil.disk_usage(Path.cwd())
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
        "storage": {
            "working_directory": str(Path.cwd().resolve()),
            "free_gib": round(disk.free / (1024**3), 2),
        },
    }
    accelerator = "cpu"
    torch_import_error: str | None = None
    if packages["torch"]:
        try:
            import torch

            mps = getattr(torch.backends, "mps", None)
            mps_built = bool(mps and mps.is_built())
            mps_available = bool(mps and mps.is_available())
            cuda_available = bool(torch.cuda.is_available())
            report["torch"] = {
                "mps_built": mps_built,
                "mps_available": mps_available,
                "cuda_available": cuda_available,
            }
            if cuda_available:
                accelerator = "cuda"
            elif mps_available:
                accelerator = "mps"
        except Exception as exc:
            torch_import_error = f"{type(exc).__name__}: {exc}"
            report["torch"] = {"import_error": torch_import_error}

    if config:
        hub = str(config["models"].get("hub", "ms"))
        model_cache: dict[str, Any] = {}
        for name in ("asr", "vad", "punctuation", "speaker"):
            configured = str(config["models"][name])
            resolved = resolve_cached_model(configured, hub)
            model_cache[name] = {
                "configured": configured,
                "resolved": resolved,
                "cached_locally": resolved != configured or Path(resolved).exists(),
            }
        report["model_cache"] = model_cache

    analysis_config = config.get("analysis", {}) if config else {}
    analysis_provider = str(analysis_config.get("provider", "none"))
    provider_settings = analysis_config.get("providers", {})
    if analysis_provider == "openai-compatible":
        selected_settings = provider_settings.get("openai_compatible", {})
    elif analysis_provider == "ollama":
        selected_settings = provider_settings.get("ollama", {})
    else:
        selected_settings = {}
    analysis_model = selected_settings.get("model")
    analysis_ready = analysis_provider == "none" or bool(packages["httpx"] and analysis_model)
    report["analysis"] = {
        "provider": analysis_provider,
        "model_configured": bool(analysis_model),
        "http_client_installed": bool(packages["httpx"]),
        "ready": analysis_ready,
    }

    recommendations: list[str] = []
    if not report["checks"]["python_supported"]:
        recommendations.append("请使用 Python 3.10、3.11 或 3.12；推荐 Python 3.11。")
    if not report["checks"]["ffmpeg_available"]:
        if platform.system() == "Darwin":
            recommendations.append("安装 FFmpeg：brew install ffmpeg")
        else:
            recommendations.append("安装 FFmpeg/FFprobe，例如：sudo apt install ffmpeg")
    if not report["checks"]["funasr_installed"]:
        recommendations.append('安装 ASR 依赖：python -m pip install -e ".[asr]"')
    if torch_import_error:
        recommendations.append(f"Torch 已安装但无法导入：{torch_import_error}")
    if disk.free < 8 * 1024**3:
        recommendations.append("当前磁盘可用空间低于 8 GiB，首次下载模型前建议释放空间。")
    setuptools_major = _major_version(packages["setuptools"])
    if sys.version_info >= (3, 12) and setuptools_major is not None and setuptools_major < 68:
        recommendations.append(
            "Python 3.12 检测到旧版 setuptools；请运行 python -m pip install --upgrade setuptools wheel。"
        )
    if analysis_provider != "none" and not packages["httpx"]:
        recommendations.append('安装分析依赖：python -m pip install -e ".[analysis]"')
    if analysis_provider != "none" and not analysis_model:
        recommendations.append(f"为 analysis provider {analysis_provider} 配置 model。")
    report["recommended_device"] = accelerator
    report["ready_for_structure"] = bool(report["checks"]["python_supported"])
    report["ready_for_asr"] = bool(
        report["checks"]["python_supported"]
        and report["checks"]["ffmpeg_available"]
        and report["checks"]["funasr_installed"]
    )
    report["ready_for_analysis"] = analysis_ready
    report["recommendations"] = recommendations or ["环境检查通过，可以运行本地 pipeline。"]
    return report


def render_environment_report(report: dict[str, Any]) -> str:
    status = "READY" if report["ready_for_asr"] else "ACTION_REQUIRED"
    lines = [
        f"Interview Recap environment: {status}",
        f"Python: {report['python']} ({'supported' if report['checks']['python_supported'] else 'unsupported'})",
        f"FFmpeg: {'available' if report['checks']['ffmpeg_available'] else 'missing'}",
        f"FunASR: {'installed' if report['checks']['funasr_installed'] else 'missing'}",
        f"Recommended device: {report['recommended_device']}",
        f"Analysis provider: {report['analysis']['provider']} "
        f"({'ready' if report['ready_for_analysis'] else 'action required'})",
        f"Free disk: {report['storage']['free_gib']} GiB",
        "",
        "Recommendations:",
    ]
    lines.extend(f"- {item}" for item in report["recommendations"])
    return "\n".join(lines)


def _progress(stage: str, event: str, details: dict[str, Any]) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    suffix = ""
    if details.get("output"):
        suffix = f" -> {details['output']}"
    elif details.get("input"):
        suffix = f" <- {details['input']}"
    elif details.get("error"):
        suffix = f": {details['error']}"
    print(f"[{timestamp}] {stage}: {event}{suffix}", file=sys.stderr, flush=True)


def _print_paths(config: dict[str, Any], result: dict[str, Any]) -> None:
    printable = {
        key: display_path(config, value) if isinstance(value, Path) else value
        for key, value in result.items()
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="interview-recap", description="本地技术面试录音复盘 pipeline"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config.yaml 路径")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_check = subparsers.add_parser("env-check", help="检查 Python、FFmpeg 与模型依赖")
    env_check.add_argument("--format", choices=("json", "text"), default="json")

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

    analyze = subparsers.add_parser("analyze", help="对 V2 JSON 运行可选语义分析并生成报告")
    analyze.add_argument("conversation_v2_json")
    analyze.add_argument(
        "--provider",
        choices=("none", "ollama", "openai-compatible"),
        help="覆盖 config.yaml 中的 analysis.provider",
    )
    analyze.add_argument("--model", help="覆盖 provider model")
    analyze.add_argument(
        "--allow-remote",
        action="store_true",
        help="明确同意发送本地脱敏后的 Question Tree 文本",
    )
    analyze.add_argument("--output-dir", help="分析 JSON 和 Markdown 报告输出目录")

    run = subparsers.add_parser("run", help="对单个音频运行完整 pipeline")
    run.add_argument("input")
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--resume", action="store_true", help="从 manifest 中最后完成的阶段继续")
    run.add_argument("--session-id", help="自定义本地会话目录名称")
    run.add_argument("--quiet", action="store_true", help="不在 stderr 输出阶段进度")
    run.add_argument(
        "--analysis-provider",
        choices=("none", "ollama", "openai-compatible"),
        help="默认 none；远程 provider 还需要 --allow-remote",
    )
    run.add_argument("--analysis-model")
    run.add_argument("--allow-remote", action="store_true")

    batch = subparsers.add_parser("batch", help="批量处理目录中的 m4a/mp3/wav")
    batch.add_argument("directory", nargs="?", default="input")
    batch.add_argument("--overwrite", action="store_true")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--quiet", action="store_true")
    batch.add_argument("--analysis-provider", choices=("none", "ollama", "openai-compatible"))
    batch.add_argument("--analysis-model")
    batch.add_argument("--allow-remote", action="store_true")

    smoke = subparsers.add_parser("make-smoke-audio", help="生成双说话人技术问答合成音频")
    smoke.add_argument("--overwrite", action="store_true")
    return parser


def _run_cli(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_project_dirs(config)

    if args.command == "env-check":
        report = environment_report(config)
        if args.format == "text":
            print(render_environment_report(report))
        else:
            print(json.dumps(report, ensure_ascii=False, indent=2))
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
            {
                "structured_json": json_path,
                "markdown": markdown_path,
                "segments": len(transcript["segments"]),
            },
        )
    elif args.command == "segment":
        transcript, json_path, markdown_path = segment_stage(config, args.structured_json)
        _print_paths(
            config,
            {
                "structured_json": json_path,
                "markdown": markdown_path,
                "qa_pairs": len(transcript["qa_pairs"]),
            },
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
    elif args.command == "analyze":
        analysis, analysis_path, report_path = analysis_stage(
            config,
            args.conversation_v2_json,
            provider_name=args.provider,
            model=args.model,
            allow_remote=args.allow_remote,
            output_dir=args.output_dir,
        )
        _print_paths(
            config,
            {
                "analysis_json": analysis_path,
                "report_markdown": report_path,
                "provider": analysis["provider"]["name"],
                "remote": analysis["provider"]["remote"],
                "questions": len(analysis["questions"]),
            },
        )
    elif args.command == "run":
        _print_paths(
            config,
            run_pipeline(
                config,
                args.input,
                args.overwrite,
                resume=args.resume,
                session_id=args.session_id,
                progress=None if args.quiet else _progress,
                analysis_provider_name=args.analysis_provider,
                analysis_model=args.analysis_model,
                allow_remote_analysis=args.allow_remote,
            ),
        )
    elif args.command == "batch":
        paths = discover_audio(args.directory)
        if not paths:
            raise SystemExit(f"目录中没有 m4a/mp3/wav: {args.directory}")
        engine = FunASREngine(config)
        results = [
            run_pipeline(
                config,
                path,
                args.overwrite,
                engine=engine,
                resume=args.resume,
                progress=None if args.quiet else _progress,
                analysis_provider_name=args.analysis_provider,
                analysis_model=args.analysis_model,
                allow_remote_analysis=args.allow_remote,
            )
            for path in paths
        ]
        for result in results:
            _print_paths(config, result)
    elif args.command == "make-smoke-audio":
        path = make_smoke_audio(config, args.overwrite)
        _print_paths(config, {"smoke_audio": path})


def main(argv: list[str] | None = None) -> None:
    try:
        _run_cli(argv)
    except (
        AnalysisError,
        AudioError,
        FunASRError,
        SessionError,
        OSError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
