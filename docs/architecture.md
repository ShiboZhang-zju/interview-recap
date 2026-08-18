# Architecture

Interview Recap separates acoustic processing, deterministic conversation repair, and optional semantic analysis.

```text
Audio
  |
  v
FFmpeg preprocessing
  |
  v
FunASR / VAD / punctuation / speaker diarization
  |
  +--> raw JSON (immutable audit source)
  |
  v
V1 normalized transcript and baseline Q&A
  |
  v
V2 conversation structure repair
  |
  +--> segment cleanup annotations
  +--> per-segment role and confidence
  +--> interview phases
  +--> coding events
  +--> traceable Question Tree
  |
  v
V2 JSON and Markdown
  |
  v
Provider-neutral analysis (default: none)
  |
  +--> minimized + locally redacted Question Tree payload
  +--> validated analysis JSON with evidence segment IDs
  |
  v
Deterministic Markdown report
```

## Stage boundaries

### Preprocessing

`src/audio.py` converts supported recordings to 16 kHz mono PCM WAV. The source recording remains untouched.

### ASR

`src/funasr_engine.py` loads the configured FunASR components and saves raw model output before normalization. Model downloads are distinct from interview-data processing.

### V1 structuring

`src/transcript.py`, `src/normalize.py`, and `src/qa.py` convert model output into timestamped segments and a baseline flat Q&A representation.

### V2 repair

`src/conversation_v2.py` creates a new derived document. It keeps every source segment and adds cleanup, role, phase, event, and tree structures. Diarization identity is weak evidence rather than a fixed role mapping.

### Rendering

JSON is the machine-readable source of truth. Markdown is a renderer output and can be regenerated.

## Data contracts

- Raw model output: schema version `0.1.0`.
- V1 structured transcript: schema version `0.1.0`.
- V2 conversation structure: schema version `0.2.0`.

The public V2 contract is described in `schemas/conversation-v2.schema.json`. Additive fields are allowed for forward compatibility. Breaking changes require a schema-version bump.

## Optional semantic analysis

Semantic interview scoring remains outside the deterministic core. The implemented analysis layer consumes V2 JSON through a provider-neutral interface:

```text
V2 JSON
  +--> no provider: deterministic Markdown only
  +--> local provider: local analysis JSON
  +--> remote provider: explicit opt-in + minimized/redacted request

validated analysis JSON --> deterministic report renderer
```

The `none` provider performs no model call. Local Ollama and OpenAI-compatible providers share the same contract. A remote provider requires explicit consent, receives no audio or full transcript, and can only return analysis tied to known question IDs and evidence segment IDs. The provider never owns the final report layout.

## Session runtime

The full `run` command writes all private artifacts to `work/sessions/<session-id>/`. An atomic `manifest.json` records each stage as `pending`, `running`, `completed`, or `failed`, together with outputs and elapsed time. `--resume` reuses only completed stages whose outputs still exist; a changed source file is rejected. Independent stage commands retain their original output paths for backward compatibility.
