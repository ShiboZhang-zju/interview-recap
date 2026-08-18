# Interview Recap

Privacy-first local interview transcription and conversation structure repair.

[中文文档](docs/README.zh-CN.md) · [Privacy model](docs/privacy.md) · [Architecture](docs/architecture.md)

Interview Recap turns an interview recording into an auditable transcript and a traceable question tree. Audio preprocessing, ASR, speaker diarization, cleanup, role repair, phase detection, and question extraction run locally. The core pipeline does not call an LLM or upload interview data.

## What it produces

- Raw FunASR output preserved before normalization.
- Timestamped transcript with diarization speaker IDs.
- Per-segment `speaker_id`, conversation `role`, `role_confidence`, and interview `phase`.
- Derived cleanup decisions without deleting source segments.
- Coding silence, low-speech-density, and noise events.
- A traceable Question Tree such as `Q1`, `Q1.1`, and `Q1.2`.
- JSON for downstream tools and Markdown for human review.

The V2 structure stage is deterministic. An optional, provider-neutral LLM analysis layer is planned for a later release; it is not required to use the project.

## Pipeline

```text
recording
  -> FFmpeg preprocessing
  -> FunASR + VAD + punctuation + speaker diarization
  -> normalized V1 transcript
  -> deterministic V2 conversation repair
  -> V2 JSON + Markdown
  -> optional downstream semantic analysis
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

The current validated environment is macOS on Apple Silicon with Python 3.11 and CPU inference. The included CI matrix covers Python 3.10–3.12 on Linux and macOS once the repository is published.

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
python -m pip install --upgrade pip
python -m pip install -e ".[asr]"
```

Check the environment:

```bash
interview-recap env-check
```

The first ASR run may download several GiB of model data from the configured model hub. Later runs prefer complete local ModelScope snapshots when available.

## Quick start

Place a supported `.m4a`, `.mp3`, or `.wav` file under `input/`, then run:

```bash
interview-recap run input/interview.m4a
```

Generated private artifacts are written to:

```text
work/audio/               preprocessed WAV
transcripts/raw/          untouched FunASR output
transcripts/structured/   V1 and V2 JSON/Markdown
```

All of these paths are ignored by Git by default.

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

## Configuration

Edit [config.yaml](config.yaml) to change model references, output paths, audio preprocessing, topic keywords, and V2 thresholds.

Domain knowledge files are intentionally simple and auditable:

- [knowledge/hotwords.txt](knowledge/hotwords.txt)
- [knowledge/normalization.yaml](knowledge/normalization.yaml)
- [knowledge/topics.yaml](knowledge/topics.yaml)

When installed as a wheel, the package contains fallback defaults. Relative output paths resolve from the current working directory.

## Privacy

The default pipeline performs no remote LLM calls. Audio, transcripts, and reports are excluded from Git by default. Model downloads are separate from interview-data processing.

Before using any future remote analysis provider, review [docs/privacy.md](docs/privacy.md). Transcript text can be as sensitive as the recording itself.

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
- The synthetic audio generator currently depends on macOS `say`.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Never attach real interview recordings or transcripts to a public issue. Report privacy or security problems using the process in [SECURITY.md](SECURITY.md).

## License

Project source code is licensed under the [Apache License 2.0](LICENSE). Downloaded model weights and third-party dependencies are governed by their respective upstream licenses and are not redistributed by this repository. See [THIRD_PARTY.md](THIRD_PARTY.md).
