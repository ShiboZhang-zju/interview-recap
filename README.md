# Interview Recap

Privacy-first local interview transcription and conversation structure repair.

[中文文档](docs/README.zh-CN.md) · [Privacy model](docs/privacy.md) · [Architecture](docs/architecture.md)

Interview Recap turns an interview recording into an auditable transcript, a traceable question tree, and an optional coaching report. Audio preprocessing, ASR, speaker diarization, cleanup, role repair, and phase detection run locally. Semantic analysis is provider-neutral, disabled by default, and runs only after the local V2 structure is complete.

## What it produces

- Raw FunASR output preserved before normalization.
- Timestamped transcript with diarization speaker IDs.
- Per-segment `speaker_id`, conversation `role`, `role_confidence`, and interview `phase`.
- Derived cleanup decisions without deleting source segments.
- Coding silence, low-speech-density, and noise events.
- A traceable Question Tree such as `Q1`, `Q1.1`, and `Q1.2`.
- A resumable per-interview workspace with an auditable `manifest.json`.
- Optional semantic analysis through `none`, local Ollama, or an OpenAI-compatible endpoint.
- JSON for downstream tools and Markdown for human review.

The V2 structure stage is deterministic. The default `none` analysis provider performs no model call and still renders a local report. Remote analysis requires an explicit `--allow-remote` flag and receives only a locally redacted Question Tree payload—never audio or the full segment transcript.

## Pipeline

```text
recording
  -> FFmpeg preprocessing
  -> FunASR + VAD + punctuation + speaker diarization
  -> normalized V1 transcript
  -> deterministic V2 conversation repair
  -> V2 JSON + Markdown
  -> provider-neutral semantic analysis (default: none)
  -> validated analysis JSON
  -> deterministic Markdown report
```

Default local models:

- ASR: FunASR `paraformer-zh`
- VAD: `fsmn-vad`
- Punctuation: `ct-punc-c`
- Speaker diarization: `cam++`

## Requirements

- Python 3.10, 3.11, or 3.12
- FFmpeg and FFprobe
- Enough disk space for the selected FunASR model snapshots
- CPU is supported; GPU configuration is environment-specific

The current validated environment is macOS on Apple Silicon with Python 3.11 and CPU inference. The CI matrix covers Python 3.10–3.12 on Linux and macOS.

## Installation

Install FFmpeg first.

macOS:

```bash
brew install ffmpeg
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[asr]"
```

Check the environment:

```bash
interview-recap env-check --format text
```

The first ASR run may download several GiB of model data from the configured model hub. Later runs prefer complete local ModelScope snapshots when available.

For CUDA, install a matching PyTorch/Torchaudio pair from the official PyTorch index first, then install `.[asr]`. The project uses compatible version ranges instead of forcing one platform-specific Torch build. If Python 3.12 reports an old `pkg_resources` or `ImpImporter` error while building, recreate the virtual environment and upgrade `setuptools` as shown above.

## Quick start

Place a supported `.m4a`, `.mp3`, or `.wav` file under `input/`, then run:

```bash
interview-recap run input/interview.m4a
```

Full runs are isolated in a stable local session directory:

```text
work/sessions/<session-id>/
  manifest.json
  audio/interview.16k.wav
  raw/interview.funasr.json
  structured/interview.json
  structured/interview.v2.json
  analysis/analysis.json
  reports/report.md
```

Progress is written to stderr, while the final machine-readable result remains JSON on stdout. If a run is interrupted, continue without repeating completed stages:

```bash
interview-recap run input/interview.m4a --resume
```

Use `--overwrite` to intentionally rerun every stage. Session artifacts are ignored by Git by default.

## Independent stages

Every major stage can be run separately:

```bash
# Audio -> 16 kHz mono WAV
interview-recap preprocess input/interview.m4a

# Local ASR and diarization
interview-recap transcribe work/audio/interview.16k.wav

# Raw FunASR JSON -> normalized V1 transcript
interview-recap structure transcripts/raw/interview.funasr.json

# V1 heuristic Q&A segmentation
interview-recap segment transcripts/structured/interview.json

# V2 conversation structure repair; V1 remains unchanged
interview-recap repair-v2 transcripts/structured/interview.json

# Deterministic local report, with no LLM call
interview-recap analyze transcripts/structured/interview.v2.json --provider none
```

The source-checkout form remains available:

```bash
python -m src.cli repair-v2 transcripts/structured/interview.json
```

## Conversation Structure V2

V2 is a derived result. It never overwrites the raw ASR output or the V1 transcript.

### Segment cleanup

Each source segment is retained and receives a `cleanup` decision. The cleaner detects pure punctuation, repetitive noise, very low text density, and long-duration sparse text. Short questions such as “为什么？” or “复杂度呢？” are explicitly protected.

### Role repair

Speaker identity and conversation role are separate:

- `speaker_id` comes from diarization.
- `role` is `interviewer`, `candidate`, or `unknown`.
- `role_confidence` is inferred for each segment.

Diarization is weak evidence. Lexical features, neighboring segments, phase, and conversation state can repair speaker drift.

### Interview phases

Supported phases are:

```text
INTRO
PROJECT
TECH_QA
CODING
CODING_DISCUSSION
BEHAVIORAL
CANDIDATE_QUESTIONS
END
UNKNOWN
```

Long coding silence and noise become events instead of fake questions. Real oral coding follow-ups can still enter the Question Tree.

### Question Tree

Questions are organized as `main_question`, `follow_up`, or `clarification`; root questions also record whether they switch topics. Every question and answer retains its original `segment_ids` and timestamps.

A synthetic, non-private example is available at [examples/synthetic_interview.v2.json](examples/synthetic_interview.v2.json). The public JSON contract is documented in [schemas/conversation-v2.schema.json](schemas/conversation-v2.schema.json).

## Optional semantic analysis

Install the lightweight HTTP client only when using Ollama or a remote compatible API:

```bash
python -m pip install -e ".[analysis]"
```

Local Ollama does not require remote consent:

```bash
interview-recap analyze path/to/interview.v2.json \
  --provider ollama \
  --model qwen2.5:7b
```

For an OpenAI-compatible service, configure `analysis.providers.openai_compatible.base_url`, keep the API key in `INTERVIEW_RECAP_API_KEY`, and explicitly authorize the redacted request:

```bash
export INTERVIEW_RECAP_API_KEY="..."
interview-recap analyze path/to/interview.v2.json \
  --provider openai-compatible \
  --model your-model \
  --allow-remote
```

Providers return validated [analysis JSON](schemas/analysis-v1.schema.json) with `question_id` and evidence `segment_ids`. A provider cannot control the final Markdown layout; the report renderer is deterministic. Unknown question IDs, unrelated evidence IDs, invalid scores, or incomplete question sets are rejected.

Long interviews are split into bounded question batches (`analysis.maximum_questions_per_request`) and reassembled in original order. `analysis.maximum_answer_chars` and `analysis.maximum_payload_chars` bound the text sent in each request.

For company names or other project-specific identifiers, set `analysis.redaction_terms_file` to an ignored local file such as `work/redaction_terms.txt` with one term per line. Do not put private terms directly in the tracked `config.yaml`.

## Configuration

Edit [config.yaml](config.yaml) to change model references, output paths, audio preprocessing, topic keywords, and V2 thresholds.

Domain knowledge files are intentionally simple and auditable:

- [knowledge/hotwords.txt](knowledge/hotwords.txt)
- [knowledge/normalization.yaml](knowledge/normalization.yaml)
- [knowledge/topics.yaml](knowledge/topics.yaml)

When installed as a wheel, the package contains fallback defaults. Relative output paths resolve from the current working directory.

## Privacy

The default pipeline uses the `none` provider and performs no remote LLM calls. Audio, transcripts, reports, and session manifests are excluded from Git by default. Model downloads are separate from interview-data processing.

Before using a remote analysis provider, review [docs/privacy.md](docs/privacy.md) and the provider's retention policy. Transcript text can be as sensitive as the recording itself.

## Testing

Core tests do not download models or process private recordings:

```bash
python -m unittest discover -s tests -v
```

On macOS, a synthetic two-speaker smoke recording can also be generated with the built-in `say` command:

```bash
interview-recap make-smoke-audio
```

## Known limitations

- ASR text is not a verbatim legal transcript; verify quotations against the audio.
- Rare English technical terms may require hotwords and manual correction.
- Overlapping speech and very short interjections remain difficult for diarization.
- V2 uses deterministic heuristics, so complex rhetorical questions and multi-interviewer sessions may require review.
- Semantic scores are model judgments, not ground truth; inspect their evidence segment IDs.
- Long analysis payloads may need a lower `analysis.maximum_answer_chars` value or separate sessions.
- The synthetic audio generator currently depends on macOS `say`.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Never attach real interview recordings or transcripts to a public issue. Report privacy or security problems using the process in [SECURITY.md](SECURITY.md).

## License

Project source code is licensed under the [Apache License 2.0](LICENSE). Downloaded model weights and third-party dependencies are governed by their respective upstream licenses and are not redistributed by this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
