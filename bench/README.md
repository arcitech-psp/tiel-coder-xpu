# Benchmark helpers

These are client-only OpenAI-compatible benchmark helpers. They do not start,
stop, or configure a server. Set `TIEL_ENDPOINT` when the tested endpoint is
not the local default.

`benchmark.json` records the bounded concurrency sweep. `pp_tg.json` records
the supplementary prompt-processing sweep. No calibration data or prompt
fixtures are included; the scripts generate their small deterministic test
inputs locally.

The 2026-09-27 fixed-K3 release comparison used the vLLM repository's
[`tfinal_bench.sh`](https://github.com/arcitech-psp/vllm-xpu-arc/blob/main/bench/tfinal_bench.sh)
and [`tfast_bench.py`](https://github.com/arcitech-psp/vllm-xpu-arc/blob/main/bench/tfast_bench.py).
It reports the median of three blocks, after one shape-priming pass, with
temperature 0 and seed 1729. The public B70 cookbook reference is linked in
the root README; its one-user, 16-bit-KV, larger-batch setup is not identical
to this release's four-slot FP8-KV setup.
