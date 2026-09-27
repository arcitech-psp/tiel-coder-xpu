# Eval summaries

Aggregate results of the internal agentic coding eval (100 tasks, 224 points: code 184, tool 20, edit 20).
Task contents and per-task records are not released.

| File | Build |
|---|---|
| `summary-gptq-a-q2.json` | Tiel-Coder XPU, this release (GPTQ-A int4 + official BF16 MTP head) |
| `summary-tiel-vllm-bf16dense.json` | Community Tiel GGUF on vLLM XPU, dense layers in BF16 |
| `summary-tiel-vllm-fp8dense.json` | Community Tiel GGUF on vLLM XPU, dense layers in FP8 |
| `summary-ct2-official-spec3.json`, `summary-ct2-official-spec3-run2.json` | Community AutoRound int4, repacked, official MTP head, 3 drafts (two runs) |

All runs used the same runner and the same 100 tasks on one Intel Arc Pro B70.

The 2026-09-27 release gate adds a ten-run fixed-K3 comparison: the new route
averaged 217.7/224 (minimum 215; code 180.3, tool 17.7, edit 19.7) and the
same-night published-build baseline averaged 217.2/224 (runs 214–219; code
180.2, tool 17.1, edit 19.9). The earlier single-run 219 remains a historical
result and is within that baseline run range.
