# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared instrumentation for run_with_openjev.py / run_without_jev.py.

Both scripts talk **directly to Dynamo** (bypassing roundhouse), for the same
reason `run_local.py` does in `cache-aware-routing`: this benchmark measures
the LLM-serving layer itself (KV-cache reuse, prefill/decode split), and
roundhouse's own wire does not expose a timing breakdown per call -- adding
that would be new Rust work orthogonal to what this benchmark needs. See
`README.md`'s "Tokenomics" section for the methodology note this implies:
these numbers characterize Dynamo/vLLM's serving cost for the two workload
shapes, not roundhouse's routing overhead (already characterized separately
in cache-aware-routing's and this use case's other docs).

**Timing methodology.** Neither `/v1/completions` nor `/v1/chat/completions`
returns a prefill/decode split in a non-streaming response. Both request
paths here stream (`"stream": true, "stream_options": {"include_usage":
true}`) and time client-side:
  - `ttft_ms`  = wall-clock time from request start to the first chunk
                 carrying generated content -- a proxy for prefill time
                 (scheduling + the actual prefill pass), the same
                 first-delta-minus-decision approximation this repo's root
                 README already uses for TTFT elsewhere.
  - `decode_ms` = wall-clock time from that first chunk to the final chunk
                 (before `[DONE]`) -- everything after the first token,
                 i.e. the decode loop.
  - `total_ms`  = wall-clock time from request start to `[DONE]`.
This is a client-side approximation (it includes local HTTP/JSON overhead,
not a server-reported number), stated here so the README doesn't imply more
precision than it has.

**KV-cache numbers are not approximated.** `cached_tokens` /
`prompt_tokens` / `completion_tokens` come straight from vLLM's own
`usage`/`usage.prompt_tokens_details.cached_tokens` on the final streamed
chunk -- the same real, provider-reported number `cache-aware-routing`'s
`run.py`/`run_local.py` already rely on, not derived from timing at all.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DYNAMO_URL = "http://127.0.0.1:8000"
RESULTS_DIR = HERE / "results"


def load_tickets() -> list[dict]:
    return [json.loads(line) for line in (HERE / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


# ------------------------------------------------------ rendering (ported from
# open-jev/openjev/systemone.py's render_choice/render_score/render_noul --
# see run_score.py's own copy of this for the attribution; duplicated here
# rather than imported so this benchmark module has no import-order
# dependency on the other script.)
def render_entry(e: Any) -> str:
    if e is None:
        return ""
    if isinstance(e, str):
        return e
    return json.dumps(e, indent=2, ensure_ascii=False)


def _block(title: str, e: Any) -> str:
    body = render_entry(e)
    return f"{title}:\n{body}\n\n" if body else ""


def render_choice(state: Any, q: dict) -> tuple[str, list[str]]:
    lines = [f"- {name}: {render_entry(desc)}" for name, desc in q["criteria"].items()]
    prompt = (
        _block("State", state)
        + _block("Question", q.get("instructions"))
        + "Choose exactly one option.\nOptions:\n" + "\n".join(lines)
        + "\n\nAnswer:\n"
    )
    return prompt, list(q["criteria"])


def render_score(state: Any, q: dict) -> tuple[str, list[str]]:
    lines = [f"{i}: {render_entry(desc)}" for i, desc in enumerate(q["criteria"])]
    prompt = (
        _block("State", state)
        + _block("Question", q.get("instructions"))
        + "Rate on the following ordered levels and answer with the level number only.\nLevels:\n"
        + "\n".join(lines)
        + "\n\nAnswer:\n"
    )
    return prompt, [str(i) for i in range(len(q["criteria"]))]


def render_noul(state: Any, q: dict) -> tuple[str, list[str]]:
    prompt = _block("State", state) + _block("Question", q.get("instructions")) + "Answer yes or no.\n\nAnswer:\n"
    return prompt, ["yes", "no"]


def render_question(state: Any, q: dict) -> tuple[str, list[str]]:
    return {"choice": render_choice, "score": render_score, "noul": render_noul}[q["type"]](state, q)


# ---------------------------------------------------------------- streaming
def stream_call(path: str, body: dict) -> dict:
    """POST a streaming request to Dynamo, timing TTFT/decode/total client-side.

    Returns {text, usage, nvext, ttft_ms, decode_ms, total_ms, error}.
    `usage` and `nvext` are the raw dicts off the last chunk that carried
    them (vLLM puts `usage` on its own final chunk, after `[DONE]`'s
    preceding one; `nvext` arrives on the first chunk for the completions
    route's `prompt_logprobs` mechanism).
    """
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{DYNAMO_URL}{path}", data=payload, method="POST",
        headers={"content-type": "application/json"},
    )
    text_parts: list[str] = []
    usage: dict = {}
    nvext: dict = {}
    t0 = time.perf_counter()
    t_first: float | None = None
    t_last: float | None = None
    error: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[len("data:"):].strip()
                if payload_str == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                if chunk.get("nvext"):
                    nvext = chunk["nvext"]
                if chunk.get("usage"):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                content = ""
                if choices:
                    delta = choices[0].get("delta")
                    content = (delta or {}).get("content", "") if delta is not None else choices[0].get("text", "")
                if content:
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text_parts.append(content)
    except Exception as exc:  # noqa: BLE001 -- surfaced in the record, not raised
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    if t_first is None:
        t_first = t_end
    if t_last is None:
        t_last = t_first
    return {
        "text": "".join(text_parts),
        "usage": usage,
        "nvext": nvext,
        "ttft_ms": (t_first - t0) * 1000,
        "decode_ms": (t_last - t_first) * 1000,
        "total_ms": (t_end - t0) * 1000,
        "error": error,
    }


# --------------------------------------------------------------- KV metrics
def kv_metrics(usage: dict) -> dict:
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    cached_tokens = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0)
    new_prefill_tokens = max(0, prompt_tokens - cached_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "new_prefill_tokens": new_prefill_tokens,
        "cache_hit_pct": (100.0 * cached_tokens / prompt_tokens) if prompt_tokens else 0.0,
        "is_cold_start": cached_tokens == 0,
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
    }


def write_json(name: str, payload: Any) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def rollup(calls: list[dict]) -> dict:
    """Sum/derive the metrics that make a per-ticket or per-workload total
    meaningful, rather than leaving a reader to sum raw per-call records by hand."""
    n = len(calls)
    total_prompt = sum(c["prompt_tokens"] for c in calls)
    total_cached = sum(c["cached_tokens"] for c in calls)
    total_new_prefill = sum(c["new_prefill_tokens"] for c in calls)
    total_completion = sum(c["completion_tokens"] for c in calls)
    return {
        "calls": n,
        "total_tokens": total_prompt + total_completion,
        "total_prompt_tokens": total_prompt,
        "total_cached_tokens": total_cached,
        "total_new_prefill_tokens": total_new_prefill,
        "total_completion_tokens": total_completion,
        "cache_hit_pct": (100.0 * total_cached / total_prompt) if total_prompt else 0.0,
        "total_ttft_ms": sum(c["ttft_ms"] for c in calls),
        "total_decode_ms": sum(c["decode_ms"] for c in calls),
        "total_wall_ms": sum(c["total_ms"] for c in calls),
        "cold_start_calls": sum(1 for c in calls if c["is_cold_start"]),
    }
