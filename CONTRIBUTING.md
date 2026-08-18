# Contributing

Thank you for helping improve Interview Recap.

## Privacy first

Never include real interview recordings, raw transcripts, structured transcripts, generated reports, credentials, or personal information in an issue, pull request, test fixture, screenshot, or CI log.

Use synthetic content. If a bug depends on a real recording, reduce it to the smallest fully sanitized text fixture before sharing it.

## Development setup

Core structure work does not require the ASR stack:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

For end-to-end ASR work:

```bash
python -m pip install -e ".[asr,dev]"
interview-recap env-check
```

FFmpeg and FFprobe are system dependencies and must be installed separately.

## Design rules

- Preserve raw model output before normalization.
- Make cleanup and structure repair derived and auditable.
- Keep `speaker_id` independent from conversation `role`.
- Retain source `segment_ids` in every question and answer.
- Keep the core pipeline local and deterministic.
- Any remote provider must be optional, disabled by default, and explicitly authorized.
- Keep analysis provider-neutral and validate every question/evidence ID before rendering.
- Never send audio, raw output, full segments, source paths, or speaker IDs to a provider.
- Do not hard-code rules for a particular company, recording, or timestamp.

## Tests

Run before opening a pull request:

```bash
python -m py_compile src/*.py
python -m unittest discover -s tests -v
```

New cleanup, role, phase, Question Tree, session-resume, redaction, or analysis behavior should include a synthetic regression test. Tests must not download model weights or call external services.

## Pull requests

- Keep changes focused.
- Explain the user-visible behavior and privacy impact.
- Document schema changes and bump the schema version when compatibility changes.
- Confirm that private paths remain ignored.
- Do not commit generated build artifacts.

By contributing, you agree that your contributions are licensed under Apache-2.0.
