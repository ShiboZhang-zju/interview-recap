# Synthetic examples

Files in this directory contain synthetic interview content only. They are safe to publish and may be used in documentation, tests, and issue reproductions.

`synthetic_interview.v2.json` demonstrates a main question, a short follow-up, traceable answers, per-segment roles, and V2 statistics without including any real person's data.

Generate an offline report without an LLM call:

```bash
interview-recap analyze examples/synthetic_interview.v2.json --provider none --output-dir work/example-report
```

Do not replace these files with real recordings or transcripts.
