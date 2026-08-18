# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and the project follows Semantic Versioning.

## [Unreleased]

### Planned

- Optional provider-neutral semantic analysis layer.
- Local redaction before any explicitly authorized remote analysis.
- Additional cross-platform smoke fixtures.

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
