#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""True time-to-first-token for a cold and a warm prefix, straight from Dynamo.

`run_pair.py` reports a TTFT measured through Roundhouse, and Roundhouse emits
a local turn's text once the worker has finished it, so that figure is the
whole turn's latency and moves with output length, not with prefill. This
probe bypasses Roundhouse: it streams a chat completion with `max_tokens=1`,
so the time to the first chunk is the prefill. Each length is sent once
cold (a unique random head defeats the prefix cache) and then again warm (the
identical prompt, which hits it).

The cold figure for the target is what a mapped source cache would skip.

    python experiments/cross-model-kv/serving/probe_ttft.py
"""
from __future__ import annotations

import json
import os
import secrets
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("DYNAMO_URL", "http://127.0.0.1:8000")
MODELS = [os.environ.get("SOURCE", "Qwen/Qwen3-8B"), os.environ.get("TARGET", "Qwen/Qwen3-14B")]
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "512,1024,2048,4096,6144").split(",")]
REPEATS = int(os.environ.get("REPEATS", "3"))
HERE = Path(__file__).resolve().parent
# About 4 characters a token for English prose; the exact count is read back from usage.
CORPUS = (HERE / "corpus.md").read_text(encoding="utf-8")


def prompt_of(tokens: int) -> str:
    text = (CORPUS + "\n") * (tokens * 4 // len(CORPUS) + 1)
    return f"[{secrets.token_hex(8)}]\n" + text[: tokens * 4]


def ttft_ms(model: str, prompt: str) -> tuple[float, int, int]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=body, headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    first, prompt_tokens, cached = None, 0, 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            event = json.loads(line[5:])
            if first is None and event.get("choices"):
                first = time.perf_counter() - t0
            usage = event.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return (first or 0.0) * 1e3, prompt_tokens, cached


def median(xs: list[float]) -> float:
    return sorted(xs)[len(xs) // 2]


def main() -> None:
    rows = []
    print(f"{'model':<16} {'tokens':>7} {'cold ms':>8} {'warm ms':>8} {'warm cached':>12}")
    for model in MODELS:
        ttft_ms(model, prompt_of(64))  # first-request CUDA/graph setup, kept out of the numbers
        for length in LENGTHS:
            colds, warms, n, cached = [], [], 0, 0
            for _ in range(REPEATS):
                p = prompt_of(length)
                c, n, _ = ttft_ms(model, p)
                w, _, cached = ttft_ms(model, p)
                colds.append(c)
                warms.append(w)
            row = {"model": model, "prompt_tokens": n, "cold_ms": median(colds), "warm_ms": median(warms), "warm_cached": cached}
            rows.append(row)
            print(f"{model:<16} {n:>7} {row['cold_ms']:>8.0f} {row['warm_ms']:>8.0f} {cached:>12}")
    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    path = out / f"probe_ttft_{int(time.time())}.json"
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
