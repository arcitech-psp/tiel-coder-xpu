#!/usr/bin/env python3
"""Bounded OpenAI-compatible throughput and smoke checks for Tiel-Coder.

The script is a client only: it never starts, stops, or configures a server.
It records results as JSON for the release plotting helper.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def http_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_text(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def endpoint_url(base: str, suffix: str) -> str:
    return base.rstrip("/") + suffix


def make_prompt(target: int) -> tuple[str, int]:
    """Create a deterministic prompt with exactly ``target`` local-token IDs."""
    seed = (
        "Review the following small software change, explain one concrete risk, "
        "and propose a minimal test. The answer should be precise. "
    )
    text = seed + ("Keep the review focused on correctness and maintainability. " * 80)
    tok_path = os.environ.get("TIEL_TOKENIZER_PATH") or str(Path(__file__).resolve().parents[1])
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tok_path, local_files_only=True)
        encode = lambda value: tokenizer.encode(value, add_special_tokens=False)
        decode = lambda value: tokenizer.decode(value, skip_special_tokens=False)
    except Exception:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(Path(tok_path) / "tokenizer.json"))
        encode = lambda value: tokenizer.encode(value).ids
        decode = lambda value: tokenizer.decode(value, skip_special_tokens=False)
    ids = encode(text)
    if len(ids) < target:
        raise RuntimeError(f"prompt seed encoded to {len(ids)} tokens; need {target}")
    exact = decode(ids[:target])
    exact_ids = encode(exact)
    if len(exact_ids) != target:
        raise RuntimeError(f"decoded prompt re-encoded to {len(exact_ids)} tokens, not {target}")
    return exact, len(exact_ids)


def parse_sse(raw: bytes) -> tuple[str, int | None, dict | None]:
    text = raw.decode("utf-8", errors="replace")
    output = []
    usage = None
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            continue
        try:
            item = json.loads(body)
        except json.JSONDecodeError:
            continue
        if item.get("usage"):
            usage = item["usage"]
        for choice in item.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content") or ""
            output.append(content)
    return "".join(output), (usage or {}).get("completion_tokens"), usage


def stream_request(
    url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    batch_started: float,
) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        endpoint_url(url, "/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    output = []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                continue
            try:
                item = json.loads(body)
            except json.JSONDecodeError:
                continue
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece or delta.get("tool_calls"):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    output.append(piece)
    finished_at = time.perf_counter()
    output_tokens = (usage or {}).get("completion_tokens")
    output_tokens = output_tokens or 0
    decode_s = (finished_at - first_token_at) if first_token_at is not None else None
    return {
        "elapsed_s": finished_at - started,
        "ttft_s": (first_token_at - started) if first_token_at is not None else None,
        "decode_s": decode_s,
        "first_token_from_batch_s": (first_token_at - batch_started) if first_token_at is not None else None,
        "finished_from_batch_s": finished_at - batch_started,
        "output_tokens": output_tokens,
        "tok_s": (output_tokens / decode_s) if output_tokens and decode_s else None,
        "usage": usage,
        "chars": len("".join(output)),
    }


def relevant_metrics(text: str) -> list[str]:
    keys = re.compile(r"(spec|draft|accept|mtp)", re.I)
    return [line for line in text.splitlines() if keys.search(line) and not line.startswith("#")]


def _counter(text: str, metric: str, position: int | None = None) -> float:
    for line in text.splitlines():
        if line.startswith("#") or not line.startswith(metric + "{"):
            continue
        if position is not None and f'position="{position}"' not in line:
            continue
        match = re.search(r"\s([-+0-9.eE]+)\s*$", line)
        if match:
            return float(match.group(1))
    return 0.0


def mtp_acceptance(before: str, after: str, draft_count: int = 3) -> list[dict]:
    """Return acceptance at each configured draft position from counter deltas."""
    drafts = _counter(after, "vllm:spec_decode_num_drafts_total") - _counter(before, "vllm:spec_decode_num_drafts_total")
    rows = []
    for position in range(draft_count):
        accepted = _counter(after, "vllm:spec_decode_num_accepted_tokens_per_pos_total", position) - _counter(
            before, "vllm:spec_decode_num_accepted_tokens_per_pos_total", position
        )
        rows.append({"draft_count": position + 1, "accepted": accepted, "drafts": drafts, "acceptance": (accepted / drafts) if drafts else None})
    return rows


def run_sweep(base: str, model: str, prompt: str, prompt_tokens: int, timeout: float) -> list[dict]:
    messages = [{"role": "user", "content": prompt}]
    results = []
    for concurrency in (1, 2, 4):
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(stream_request, base, model, messages, 400, timeout, started)
                for _ in range(concurrency)
            ]
            rows = [f.result() for f in futures]
        wall = time.perf_counter() - started
        valid = [row for row in rows if row.get("tok_s") is not None]
        total_tokens = sum(row.get("output_tokens") or 0 for row in rows)
        first_tokens = [row["first_token_from_batch_s"] for row in rows if row.get("first_token_from_batch_s") is not None]
        finished = [row["finished_from_batch_s"] for row in rows]
        decode_window = (max(finished) - min(first_tokens)) if first_tokens else None
        results.append(
            {
                "concurrency": concurrency,
                "prompt_tokens_requested": 231,
                "prompt_tokens_estimated": prompt_tokens,
                "output_tokens_requested": 400,
                "wall_s": wall,
                "streams": rows,
                "per_stream_tok_s_mean": (sum(row["tok_s"] for row in valid) / len(valid)) if valid else None,
                "aggregate_tok_s": (total_tokens / decode_window) if decode_window and total_tokens else None,
                "decode_window_s": decode_window,
            }
        )
    return results


def smoke_tests(base: str, model: str, timeout: float) -> dict:
    tool_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Use the calculator tool to add 2 and 3."}],
        "tools": [{"type": "function", "function": {"name": "calculator", "description": "Add two integers", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}],
        "tool_choice": {"type": "function", "function": {"name": "calculator"}},
        "temperature": 0,
        "max_tokens": 128,
    }
    reasoning_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reason briefly: what is 19 multiplied by 23?"}],
        "temperature": 0,
        "max_tokens": 128,
    }
    out = {}
    for name, payload in (("tool_call", tool_payload), ("reasoning", reasoning_payload)):
        try:
            response = http_json(endpoint_url(base, "/chat/completions"), payload, timeout)
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            out[name] = {
                "ok": True,
                "has_tool_calls": bool(message.get("tool_calls")),
                "has_reasoning_content": bool(message.get("reasoning_content")),
                "finish_reason": choice.get("finish_reason"),
            }
        except Exception as exc:
            out[name] = {"ok": False, "error": type(exc).__name__}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=os.environ.get("TIEL_ENDPOINT", "http://localhost:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("TIEL_MODEL", "tiel-coder"))
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output", default=os.environ.get("TIEL_BENCH_JSON", "bench/benchmark.json"))
    args = parser.parse_args()

    prompt, prompt_tokens = make_prompt(231)
    metrics_before = []
    try:
        metrics_before = relevant_metrics(http_text(endpoint_url(args.endpoint.removesuffix("/v1"), "/metrics"), 30))
    except Exception as exc:
        print(f"metrics before: {type(exc).__name__}", file=sys.stderr)
    sweep = run_sweep(args.endpoint, args.model, prompt, prompt_tokens, args.timeout)
    metrics_after = []
    try:
        metrics_after = relevant_metrics(http_text(endpoint_url(args.endpoint.removesuffix("/v1"), "/metrics"), 30))
    except Exception as exc:
        print(f"metrics after: {type(exc).__name__}", file=sys.stderr)
    result = {
        "endpoint": args.endpoint,
        "model": args.model,
        "prompt_tokens_requested": 231,
        "output_tokens_requested": 400,
        "sweep": sweep,
        "metrics": {"before": metrics_before, "after": metrics_after},
        "mtp_acceptance": mtp_acceptance("\n".join(metrics_before), "\n".join(metrics_after)),
        "smoke": smoke_tests(args.endpoint, args.model, args.timeout),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": "OpenAI-compatible streaming; concurrency 1/2/4; temperature 0.2; MTP metrics snapshot",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "sweep": sweep, "smoke": result["smoke"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
