# Reproduction helpers

`quantize_tiel.py` is a sanitized reference recipe for experts-only symmetric
GPTQ, group size 128, exported as compressed-tensors packed int4. It expects a
BF16 source directory, a compatible compressed-tensors reference/configuration,
an output directory, and caller-owned calibration JSONL. The calibration data
is not included in this release.

`make_bf16_mtp.py` grafts the `mtp.*` tensors from an official BF16 reference
onto a body checkpoint while leaving the body shards unchanged.

`serve-tiel-gptq.sh` is a Docker command using the tested vLLM XPU flags. Set
`MODEL_DIR` and `VLLM_XPU_IMAGE` for your environment. The tested deployment
also used a custom vLLM XPU build and mounted XPU extensions; stock vLLM XPU
compatibility is not claimed here.
