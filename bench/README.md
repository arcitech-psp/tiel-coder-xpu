# Benchmark helpers

These are client-only OpenAI-compatible benchmark helpers. They do not start,
stop, or configure a server. Set `TIEL_ENDPOINT` when the tested endpoint is
not the local default.

`benchmark.json` records the bounded concurrency sweep. `pp_tg.json` records
the supplementary prompt-processing sweep. No calibration data or prompt
fixtures are included; the scripts generate their small deterministic test
inputs locally.

Credit: GPT 5.6 Luna (Codex), directed by Claude.
