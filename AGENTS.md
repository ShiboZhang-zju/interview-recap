# Repository guidance

This repository is a privacy-first, local post-interview transcription pipeline.

## Non-negotiable constraints

- Do not add remote API or LLM calls to the core pipeline.
- Any future remote analysis provider must be optional, disabled by default, and require explicit user consent.
- Keep semantic analysis behind a provider-neutral interface and preserve a fully local no-LLM path.
- Do not add a UI, database, Docker image, or web service unless the user explicitly requests one.
- Treat files under `input/`, `work/`, `transcripts/`, and `reports/` as private local artifacts. Never commit, attach, or upload them by default.
- Prefer existing pretrained FunASR models. Do not introduce model training.

## Engineering workflow

- Use Python 3.10–3.12. Python 3.11 in a project `.venv` is the validated Apple Silicon setup.
- Keep preprocessing, FunASR inference, structuring, and Q&A segmentation independently runnable.
- Preserve raw model output before normalization so every transformation remains auditable.
- Segment timestamps are floating-point seconds; raw FunASR timestamps remain untouched in the raw JSON.
- Add or update tests for normalization, timestamp conversion, schema, and Q&A heuristics.
- Use only synthetic or explicitly sanitized fixtures in tests and public documentation.
- Run `python -m unittest discover -s tests -v` after code changes.
