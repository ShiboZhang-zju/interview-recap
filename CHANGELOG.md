# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and the project follows Semantic Versioning.

## [Unreleased]

### Added

- Per-interview session workspaces with atomic stage manifests and `--resume` support.
- Progress events on stderr and expanded actionable `env-check` diagnostics.
- Provider-neutral semantic analysis with `none`, Ollama, and OpenAI-compatible providers.
- Question-tree-only payload minimization and local redaction before remote analysis.
- Explicit `--allow-remote` consent gate, environment-only API credentials, and HTTPS enforcement.
- Validated analysis JSON with question/evidence traceability and a deterministic report renderer.
- Public analysis JSON Schema and synthetic regression coverage.

### Changed

- Relaxed exact Torch/Torchaudio pins to compatible Python 3.10–3.12 version ranges.
- Full `run` and `batch` executions now isolate artifacts under `work/sessions/`; independent stage commands remain compatible.

## [0.1.0] - 2026-08-18

### Added

- Local FFmpeg audio preprocessing.
- FunASR transcription with VAD, punctuation, and speaker diarization.
- Auditable raw and normalized transcript formats.
- V1 heuristic Q&A segmentation.
- V2 segment cleanup, per-segment role repair, interview phase detection, coding events, and Question Tree output.
- CLI commands for every independent stage.
- Synthetic unit tests for cleanup, role repair, phase handling, and traceability.
- Privacy-safe open-source packaging, documentation, schema, and CI configuration.
