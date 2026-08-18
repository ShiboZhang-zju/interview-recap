from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SOURCE_CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
BUNDLED_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
BUNDLED_CONFIG = BUNDLED_RESOURCE_ROOT / "config.yaml"
DEFAULT_CONFIG = SOURCE_CONFIG if SOURCE_CONFIG.is_file() else BUNDLED_CONFIG


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    required = ("project", "audio", "models", "inference", "knowledge", "qa")
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"config 缺少字段: {', '.join(missing)}")

    config["_config_path"] = config_path
    # A wheel ships a read-only default config inside the package. Its relative
    # output paths must resolve from the user's working directory, never from
    # site-packages. Repository and user-supplied configs remain self-contained.
    config["_root"] = Path.cwd().resolve() if config_path == BUNDLED_CONFIG else config_path.parent
    config["_resource_root"] = BUNDLED_RESOURCE_ROOT
    return config


def project_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()

    local_path = (config["_root"] / path).resolve()
    if local_path.exists():
        return local_path

    resource_path = (config.get("_resource_root", BUNDLED_RESOURCE_ROOT) / path).resolve()
    if path.parts and path.parts[0] == "knowledge" and resource_path.exists():
        return resource_path
    return local_path


def ensure_project_dirs(config: dict[str, Any]) -> None:
    for key in (
        "input_dir",
        "sessions_dir",
        "work_audio_dir",
        "raw_dir",
        "structured_dir",
        "reports_dir",
    ):
        if key not in config["project"]:
            continue
        project_path(config, config["project"][key]).mkdir(parents=True, exist_ok=True)


def display_path(config: dict[str, Any], path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(config["_root"]))
    except ValueError:
        return str(resolved)
