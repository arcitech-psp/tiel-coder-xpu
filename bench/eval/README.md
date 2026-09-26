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
