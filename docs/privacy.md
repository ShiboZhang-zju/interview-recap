# Privacy Model

Interview recordings and transcripts can contain identity, employment, compensation, proprietary project, and authentication information. Treat transcript text as sensitive as the source audio.

## Default guarantees

- The default `none` provider does not call an LLM.
- ASR and conversation repair run locally.
- Raw ASR output is preserved locally for auditability.
- A full run stores private artifacts under an ignored per-interview session directory.
- `input/`, `work/`, `transcripts/`, and `reports/` are ignored by Git.
- Tests use synthetic text and do not upload artifacts.

Model snapshot downloads may contact the configured model hub, but the project does not send interview audio or transcript text as part of model download.

## Threat boundaries

The project cannot protect data that a user:

- force-adds to Git;
- pastes into a public issue or external chat;
- stores in a cloud-synchronized project directory;
- includes in shell history, crash reports, or debug logs;
- explicitly sends to a configured remote analysis provider.

## Safe operating guidance

1. Work in a local, access-controlled directory.
2. Inspect `git status --ignored` before every release.
3. Inspect the staged diff before every commit.
4. Do not enable verbose logs containing transcript text in shared environments.
5. Delete local artifacts according to your own retention policy.
6. Verify exact quotations against the original recording.

## Optional analysis providers

The analysis layer supports `none`, local Ollama, and OpenAI-compatible endpoints. Remote execution is constrained as follows:

- disabled by default;
- explicit `--allow-remote` consent on each command;
- no audio upload;
- question-tree-only payload by default;
- local redaction before transmission;
- no credentials in config files or logs;
- clear provider/model/request metadata in the analysis result;
- documented provider retention and training implications;
- a fully local alternative remains available.

The minimized request contains question IDs, question text, associated answer text, phase/topic labels, and evidence segment IDs. It excludes the audio, full `segments` array, source path, speaker IDs, and raw FunASR output. The request content itself is not written to logs; only a SHA-256 fingerprint and category-level redaction counts are saved.

Built-in redaction covers common credentials, email addresses, mainland China mobile numbers and identity-card patterns, and IPv4 addresses. Project-specific names or identifiers should be stored one per line in an ignored local file such as `work/redaction_terms.txt`, referenced by `analysis.redaction_terms_file`. Never put real private terms in the tracked `config.yaml`. Automated redaction cannot guarantee removal of every indirect identifier, proprietary term, or contextual clue.

An external provider's privacy policy and retention behavior are outside this project's guarantees.
