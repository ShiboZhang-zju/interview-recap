# Security Policy

## Supported versions

Security and privacy fixes are applied to the latest release on the default branch while the project is in its `0.x` phase.

## Reporting a vulnerability

Do not disclose a vulnerability through a public issue when it could expose recordings, transcripts, credentials, local file paths, or remote-provider data.

Use GitHub's private vulnerability reporting or a private security advisory for the repository. Include a minimal synthetic reproduction whenever possible. Do not upload a real interview recording or transcript.

## Sensitive data expectations

The following paths are treated as private local artifacts and are ignored by Git:

- `input/`
- `work/`
- `transcripts/`
- `reports/`

The project cannot prevent a user from manually forcing ignored files into Git. Always inspect the staged diff before committing.

The current core pipeline does not call a remote LLM. If remote analysis providers are added later, they must remain disabled by default, require explicit consent, avoid logging transcript content, and document exactly which fields leave the device.

## Credentials

Never place API keys or tokens in `config.yaml`, source code, tests, or reports. Use environment variables and keep `.env` files local. If a credential is exposed, revoke or rotate it immediately before attempting history cleanup.
