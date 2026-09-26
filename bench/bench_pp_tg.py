#!/usr/bin/env python3
"""OpenAI-compatible PP/TG sweep for 512/2K/8K/32K prompt targets.

This is a client-only measurement. It uses max_tokens=1 to isolate prompt
processing and records single-stream and four-stream aggregate rates.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = os.environ.get("TIEL_ENDPOINT", "http://localhost:8000/v1").rstrip("/")
MODEL = os.environ.get("TIEL_MODEL", "tiel-coder")


def prompt(target: int) -> tuple[str, int]:
    seed = "Explain one correctness risk in a small software change and name one test. "
    text = seed * max(100, target // 10)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(ROOT), local_files_only=True)
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids[:target], skip_special_tokens=False)
        return decoded, len(tokenizer.encode(decoded, add_special_tokens=False))
    except Exception:
        return text, target


def request(content: str, timeout: float) -> dict:
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 1}
    req = urllib.request.Request(
        ENDPOINT + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - started
    usage = body.get("usage", {})
    return {"elapsed_s": elapsed, "prompt_tokens": usage.get("prompt_tokens"), "usage": usage}


def main() -> int:
    rows = []
    for target in (512, 2048, 8192, 32768):
        content, measured = prompt(target)
        for concurrency in (1, 4):
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                results = [f.result() for f in [pool.submit(request, content, 3600) for _ in range(concurrency)]]
            wall = time.perf_counter() - started
            total = sum(row.get("prompt_tokens") or 0 for row in results)
            rows.append({
                "prompt_tokens_requested": target,
                "prompt_tokens_estimated": measured,
                "concurrency": concurrency,
                "wall_s": wall,
                "streams": results,
                "aggregate_pp_tok_s": total / wall if wall and total else None,
                "per_stream_pp_tok_s_mean": (total / concurrency) / (sum(row["elapsed_s"] for row in results) / concurrency) if all(row.get("prompt_tokens") for row in results) else None,
            })
            print(rows[-1])
    out = ROOT / "bench" / "pp_tg.json"
    out.write_text(json.dumps({"endpoint": ENDPOINT, "model": MODEL, "rows": rows, "protocol": "max_tokens=1; concurrency 1/4"}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
