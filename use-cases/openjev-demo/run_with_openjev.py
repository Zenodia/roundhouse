#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: the real Jev-style logprob-scoring path, instrumented per LLM call.

Same mechanism `run_score.py` and `crates/roundhouse-server/src/local_score.rs`
use (`prompt_logprobs` + `nvext.extra_fields: ["prompt_logprobs"]` against
Dynamo's `/v1/completions`, raw token-id prompts) -- this script calls Dynamo
directly instead of through roundhouse's `/v1/local/score`, so it can time
each call's prefill/decode split (see `bench_common.py`'s module doc for why
and how). Registers, per LLM call: KV-cache-% reused, cold-start flag,
prefill (TTFT) and decode time, end-to-end time, cached vs newly-ingested
prefill tokens, plus the logprob-derived answer -- then rolls up to per-ticket
and per-workload totals, because Jev makes one HTTP call *per candidate
option* (3+3+2 = 8 per ticket here), not one per question or per ticket, and
a fair cost comparison against `run_without_jev.py`'s one-call-per-ticket
baseline has to be made at the rolled-up level, not the raw per-call level.

Prereqs: Dynamo serving directly on :8000 (see DYNAMO_LOCAL_SERVING.md), run
from the `dynamo` venv so `tokenizers` is importable
(`source ~/dynamo/.venv/bin/activate`), and:
    export ROUNDHOUSE_LOCAL_TOKENIZER=/path/to/tokenizer.json
No roundhouse process needs to be running -- this bypasses it entirely.

Usage:
    ROUNDHOUSE_LOCAL_TOKENIZER=/path/to/tokenizer.json python3 use-cases/openjev-demo/run_with_openjev.py
"""
from __future__ import annotations

import os
import sys
import time

from bench_common import (
    kv_metrics,
    load_tickets,
    render_question,
    rollup,
    write_json,
)

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit(
        "the `tokenizers` package is not importable -- run this from the Dynamo venv, "
        "e.g. `source ~/dynamo/.venv/bin/activate`"
    )

TOKENIZER_PATH = os.environ.get("ROUNDHOUSE_LOCAL_TOKENIZER")
MODEL = os.environ.get("ROUNDHOUSE_LOCAL_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")


def score_option(tok: Tokenizer, context_ids: list[int], option: str) -> dict:
    from bench_common import stream_call

    option_ids = tok.encode(option, add_special_tokens=False).ids
    prompt_ids = context_ids + option_ids
    result = stream_call(
        "/v1/completions",
        {
            "model": MODEL,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "nvext": {"extra_fields": ["prompt_logprobs"]},
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    # Aligned from the end of the returned array, not from context length --
    # see local_score.rs's own doc comment: a cached prefix means
    # nvext.prompt_logprobs comes back shorter than the prompt, covering only
    # the uncached tail, and the option's own tokens are always that tail.
    prompt_logprobs = (result.get("nvext") or {}).get("prompt_logprobs") or []
    tail = prompt_logprobs[-len(option_ids):] if option_ids else []
    logprob_sum = 0.0
    scoring_error = None
    if len(tail) < len(option_ids):
        scoring_error = f"prompt_logprobs carried {len(prompt_logprobs)} entries, fewer than {len(option_ids)} option tokens"
    else:
        for entry, token_id in zip(tail, option_ids):
            hit = (entry or {}).get(str(token_id))
            if hit is None:
                scoring_error = f"no logprob entry for token {token_id}"
                break
            logprob_sum += hit["logprob"]

    record = {
        "option": option,
        "option_tokens": len(option_ids),
        "logprob_sum": logprob_sum,
        "logprob_mean": (logprob_sum / len(option_ids)) if option_ids else 0.0,
        "scoring_error": scoring_error,
        "ttft_ms": result["ttft_ms"],
        "decode_ms": result["decode_ms"],
        "total_ms": result["total_ms"],
        "call_error": result["error"],
        **kv_metrics(result["usage"]),
    }
    return record


def softmax(xs: list[float]) -> list[float]:
    import math
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    return [e / z for e in exps]


def run_question(tok: Tokenizer, state: str, qid: str, q: dict) -> tuple[dict, list[dict]]:
    prompt, labels = render_question(state, q)
    context_ids = tok.encode(prompt, add_special_tokens=False).ids
    calls = [score_option(tok, context_ids, label) for label in labels]
    means = [c["logprob_mean"] for c in calls]
    probs = softmax(means)
    for c, p in zip(calls, probs):
        c["probability"] = p
    best_idx = max(range(len(labels)), key=lambda i: probs[i])
    answer = {"qid": qid, "type": q["type"], "answer": labels[best_idx], "probabilities": dict(zip(labels, probs))}
    return answer, calls


def main() -> None:
    if not TOKENIZER_PATH:
        sys.exit("set ROUNDHOUSE_LOCAL_TOKENIZER=/path/to/tokenizer.json")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    tickets = load_tickets()

    print(f"run_with_openjev: {len(tickets)} tickets, real logprob scoring against {MODEL}\n")

    per_ticket: list[dict] = []
    all_calls: list[dict] = []
    workload_t0 = time.perf_counter()

    for ticket in tickets:
        ticket_calls: list[dict] = []
        answers = []
        for qid, q in ticket["questions"].items():
            answer, calls = run_question(tok, ticket["state"], qid, q)
            for c in calls:
                c["ticket_id"] = ticket["id"]
                c["question_id"] = qid
            answers.append(answer)
            ticket_calls.extend(calls)
        all_calls.extend(ticket_calls)
        ticket_rollup = rollup(ticket_calls)
        per_ticket.append({
            "ticket_id": ticket["id"],
            "answers": answers,
            "calls": ticket_calls,
            "rollup": ticket_rollup,
        })
        print(
            f"  {ticket['id']:<16} calls={ticket_rollup['calls']:>2} "
            f"prompt_tok={ticket_rollup['total_prompt_tokens']:>5} "
            f"cached={ticket_rollup['total_cached_tokens']:>5} "
            f"({ticket_rollup['cache_hit_pct']:.1f}%) "
            f"wall_ms={ticket_rollup['total_wall_ms']:.0f}"
        )

    workload_wall_ms = (time.perf_counter() - workload_t0) * 1000
    workload_rollup = rollup(all_calls)
    workload_rollup["measured_wall_ms"] = workload_wall_ms  # includes client-side sequencing overhead

    payload = {
        "workload": "with_openjev",
        "model": MODEL,
        "tickets": per_ticket,
        "rollup": workload_rollup,
    }
    out = write_json("with_openjev.json", payload)
    print(f"\nWorkload totals: {workload_rollup}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
