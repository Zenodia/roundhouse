#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure the prefill tax of a mid-conversation model handoff through
Roundhouse, with both models of the pair served by Dynamo.

What it shows
-------------
A conversation runs N turns on the source model; every turn re-sends the
history, and the source's KV cache warms (cached% climbs). The same history
is then continued on the target model. The target holds none of that prefix,
so its first turn prefills everything from scratch -- cached=0 -- and only
afterwards warms. That cold turn is exactly what the paper's mapper removes:
a mapped source cache would let the target start warm.

Three phases, three keys (see control-plane.json):
  1. source-only : turns 1..N        on the source  (warm-up on the source)
  2. target-only : turns N+1..2N     on the target  (turn N+1 is the handoff)
  3. pair        : the whole script  under local/*  (what the router does unforced)

Per turn: input tokens, cached tokens, cache%, output tokens, and the model
that served it (read from /v1/metrics deltas, since the Responses usage
block does not name the worker). Roundhouse's own cached count comes from
the Dynamo KV events of the worker that served the turn (`LocalQuote`), not
from vLLM's self-report.

Prereqs: serve_pair.sh running, Roundhouse running with env.sh sourced,
mint_keys.py already run.

Usage:
    python experiments/cross-model-kv/serving/run_pair.py
    ROUNDHOUSE_URL=http://127.0.0.1:8080 HANDOFF_AT=4 python experiments/cross-model-kv/serving/run_pair.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_URL = os.environ.get("ROUNDHOUSE_URL", "http://127.0.0.1:8080").rstrip("/")
TURN_KEY_HEADER = "x-roundhouse-key"
HANDOFF_AT = int(os.environ.get("HANDOFF_AT", "4"))

SYSTEM_PREAMBLE = (
    "You are a precise engineering assistant answering questions about the "
    "Ledgerline service described below. Answer only from the document; if the "
    "document does not say, reply exactly 'not specified'. Be concise.\n\n"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_turns() -> list[str]:
    return [json.loads(l)["q"] for l in (HERE / "turns.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def metrics(admin_secret: str | None) -> dict:
    headers = {TURN_KEY_HEADER: admin_secret} if admin_secret else {}
    req = urllib.request.Request(f"{BASE_URL}/v1/metrics", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError):
        return {}


def model_calls(snapshot: dict) -> dict[str, int]:
    """{provider/model: calls} from a metrics snapshot, tolerant of shape."""
    out = {}
    for entry in snapshot.get("models", []) or []:
        name = f"{entry.get('provider', '?')}/{entry.get('model', '?')}"
        out[name] = int(entry.get("calls", 0))
    return out


def served_by(before: dict[str, int], after: dict[str, int]) -> str:
    changed = [m for m in after if after[m] != before.get(m, 0)]
    return changed[0] if len(changed) == 1 else (",".join(changed) or "?")


def stream_turn(secret: str, instructions: str, conversation: list[dict], cache_key: str) -> dict:
    body = json.dumps({"instructions": instructions, "input": conversation, "stream": True, "prompt_cache_key": cache_key}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/responses", data=body, method="POST", headers={"content-type": "application/json", TURN_KEY_HEADER: secret}
    )
    text, usage, error, model = [], {}, None, None
    t0 = time.perf_counter()
    ttft = None
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                et = event.get("type", "")
                if et == "response.output_text.delta":
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text.append(event.get("delta", ""))
                elif et == "response.completed":
                    r = event.get("response", {})
                    usage = r.get("usage", {}) or {}
                    model = r.get("model")
                elif et == "response.failed":
                    error = json.dumps(event.get("response", {}).get("error", event))
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}"
    except urllib.error.URLError as exc:
        error = f"connection error: {exc}. Is roundhouse running at {BASE_URL}?"
    return {"text": "".join(text), "usage": usage, "error": error, "model": model, "ttft_s": ttft, "wall_s": time.perf_counter() - t0}


def run_turns(label, secret, admin, instructions, conversation, questions, first_index, cache_key, rows):
    for i, q in enumerate(questions, start=first_index):
        conversation.append({"type": "message", "role": "user", "content": q})
        before = model_calls(metrics(admin))
        res = stream_turn(secret, instructions, conversation, cache_key)
        if res["error"]:
            print(f"{i:>4}  ERROR: {res['error']}")
            return False
        after = model_calls(metrics(admin))
        u = res["usage"]
        in_tok = int(u.get("input_tokens", 0))
        cached = int(u.get("input_tokens_details", {}).get("cached_tokens", 0))
        out_tok = int(u.get("output_tokens", 0))
        pct = 100.0 * cached / in_tok if in_tok else 0.0
        who = served_by(before, after)
        ttft = f"{res['ttft_s'] * 1e3:7.0f}" if res["ttft_s"] is not None else "      -"
        print(f"{i:>4}  {label:<12} {in_tok:>8} {cached:>8} {pct:>6.1f}% {out_tok:>7} {ttft}  {who}")
        rows.append({"turn": i, "phase": label, "input": in_tok, "cached": cached, "cache_pct": pct, "output": out_tok, "ttft_ms": res["ttft_s"] and res["ttft_s"] * 1e3, "served_by": who})
        conversation.append({"type": "message", "role": "assistant", "content": res["text"]})
    return True


def header():
    print(f"{'turn':>4}  {'phase':<12} {'in_tok':>8} {'cached':>8} {'cache%':>7} {'out':>7} {'ttft_ms':>7}  served_by")


def main() -> None:
    keys_file = HERE / "keys.local.json"
    if not keys_file.exists():
        sys.exit("keys.local.json not found -- run mint_keys.py first")
    secrets_map = load_json(keys_file)
    admin = secrets_map.get("__admin__")
    corpus = (HERE / "corpus.md").read_text(encoding="utf-8")
    instructions = SYSTEM_PREAMBLE + corpus
    turns = load_turns()
    n = HANDOFF_AT
    if len(turns) < 2 * n:
        sys.exit(f"need at least {2 * n} turns in turns.jsonl for HANDOFF_AT={n}")

    print(f"roundhouse: {BASE_URL}; shared prefix {len(corpus)} chars; handoff after turn {n}")
    rows: list[dict] = []

    # --- forced handoff: source for n turns, then the target continues ----
    print("\n=== forced handoff: source-only -> target-only ===")
    header()
    conversation: list[dict] = []
    ok = run_turns("source-only", secrets_map["source-only/dev"], admin, instructions, conversation, turns[:n], 1, "handoff-run", rows)
    if ok:
        run_turns("target-only", secrets_map["target-only/dev"], admin, instructions, conversation, turns[n : 2 * n], n + 1, "handoff-run", rows)
    handoff = next((r for r in rows if r["turn"] == n + 1), None)
    if handoff:
        print(
            f"\n  handoff turn {n + 1}: {handoff['input']} input tokens, {handoff['cached']} cached "
            f"({handoff['cache_pct']:.1f}%) on {handoff['served_by']}. The uncached "
            f"{handoff['input'] - handoff['cached']} tokens are the prefill a mapped source cache would skip."
        )

    # --- unforced: the router picks under local/* -------------------------
    print("\n=== unforced: pair (local/*) ===")
    header()
    run_turns("pair", secrets_map["pair/dev"], admin, instructions, [], turns[: 2 * n], 1, "pair-run", rows)

    out = HERE / "runs"
    out.mkdir(exist_ok=True)
    path = out / f"run_pair_{int(time.time())}.json"
    path.write_text(json.dumps({"handoff_at": n, "rows": rows, "metrics": metrics(admin)}, indent=2))
    print(f"\nwrote {path}\ndashboard: {BASE_URL}/v1/metrics/dashboard")


if __name__ == "__main__":
    main()
