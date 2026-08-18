# Privacy Model

Interview recordings and transcripts can contain identity, employment, compensation, proprietary project, and authentication information. Treat transcript text as sensitive as the source audio.

## Default guarantees

- The core Python pipeline does not call a remote LLM.
- ASR and conversation repair run locally.
- Raw ASR output is preserved locally for auditability.
- `input/`, `work/`, `transcripts/`, and `reports/` are ignored by Git.
- Tests use synthetic text and do not upload artifacts.

Model snapshot downloads may contact the configured model hub, but the project does not send interview audio or transcript text as part of model download.

## Threat boundaries

The project cannot protect data that a user:

- force-adds to Git;
- pastes into a public issue or external chat;
- stores in a cloud-synchronized project directory;
- includes in shell history, crash reports, or debug logs;
- explicitly sends to a future remote analysis provider.

## Safe operating guidance

1. Work in a local, access-controlled directory.
2. Inspect `git status --ignored` before every release.
3. Inspect the staged diff before every commit.
4. Do not enable verbose logs containing transcript text in shared environments.
5. Delete local artifacts according to your own retention policy.
6. Verify exact quotations against the original recording.

## Future remote providers

Any future remote provider must meet these requirements:

- disabled by default;
- explicit `allow_remote` consent;
- no audio upload;
- question-tree-only payload by default;
- local redaction before transmission;
- no credentials in config files or logs;
- clear provider/model/request metadata in the analysis result;
- documented provider retention and training implications;
- a fully local alternative remains available.

An external provider's privacy policy and retention behavior are outside this project's guarantees.
