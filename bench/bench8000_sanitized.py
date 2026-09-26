"""Direct OpenAI-compatible benchmark: TTFT, decode tok/s, and concurrency."""
from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
import urllib.request

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "tiel-coder"
PROMPT = (
    'Write a Python function that parses an ISO-8601 duration string such as '
    '"P3DT4H12M" into total seconds, with input validation, and include three unit tests.'
)


def run(prompt: str, max_tokens: int = 400, thinking: bool = False) -> dict:
    body = json.dumps({
        "model": MODEL, "stream": True, "max_tokens": max_tokens,
        "temperature": 0.2, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking},
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    started = time.time()
    first = None
    text = ""
    usage = {}
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("content") or "") + (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                if piece and first is None:
                    first = time.time()
                text += piece
    ended = time.time()
    output_tokens = usage.get("completion_tokens", 0)
    return {
        "ttft_s": round((first or ended) - started, 2),
        "decode_tok_s": round((output_tokens - 1) / max(ended - (first or ended), 1e-6), 1),
        "completion_tokens": output_tokens,
        "prompt_tokens": usage.get("prompt_tokens"),
        "text": text,
    }


if __name__ == "__main__":
    warmup = run("Say hello in five words.", 16)
    print("warmup", {k: v for k, v in warmup.items() if k != "text"})
    single = run(PROMPT)
    print("single", {k: v for k, v in single.items() if k != "text"})
    for concurrency in [int(value) for value in sys.argv[1:]] or [4]:
        started = time.time()
        with cf.ThreadPoolExecutor(concurrency) as pool:
            rows = list(pool.map(lambda _: run(PROMPT), range(concurrency)))
        wall = time.time() - started
        total = sum(row["completion_tokens"] for row in rows)
        print(
            f"concurrency {concurrency}: per-request decode "
            f"{[row['decode_tok_s'] for row in rows]} tok/s, "
            f"ttft {[row['ttft_s'] for row in rows]} s, "
            f"aggregate {total / wall:.1f} tok/s over {wall:.1f} s"
        )
