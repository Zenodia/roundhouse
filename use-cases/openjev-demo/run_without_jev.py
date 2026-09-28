#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark: the "without Jev" baseline -- one combined structured-JSON
generation call per ticket, no logprob flag, contrasted against
run_with_openjev.py's per-option scoring calls.

Why JSON generation and not free-form prose, and not the constrained
label-only prompting `run.py` uses: both of those would either overstate or
understate the contrast. Free-form prose reasoning would conflate "scoring
vs. generating" with "terse vs. verbose" -- a different, less controlled
question. `run.py`'s label-only prompting already decodes almost nothing
(one word), which would show barely any latency difference from Jev's
near-zero-decode scoring calls and understate what real, unconstrained
generation costs. Structured JSON output is what a real "without Jev"
agent commonly does in practice (function-calling / structured-output
patterns), asks the model to resolve the *same three decisions* Jev scores,
and still requires genuine sequential decoding (field names, punctuation,
values) -- the real point of contrast.

One call per ticket (not one per question) is the natural shape for this
baseline: a real structured-output agent asks for all needed fields in one
round trip, not three. This is also the sharpest version of the
tokenomics question this benchmark exists to answer: 1 heavier call vs.
Jev's 8 lighter ones, compared at the rolled-up per-ticket total, not
call-for-call.

Prereqs: Dynamo serving directly on :8000. No tokenizer/venv requirement --
this sends text, not raw token ids (there is no cache-hash invariant to
preserve here, unlike the local-tier turn-dispatch path).

Usage:
    python3 use-cases/openjev-demo/run_without_jev.py
"""
from __future__ import annotations

import json
import os
import re
import time

from bench_common import kv_metrics, load_tickets, render_entry, rollup, stream_call, write_json

MODEL = os.environ.get("ROUNDHOUSE_LOCAL_MODEL", "Qwen/Qwen2.5-Coder-32B-Instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "150"))


def render_combined_prompt(state: str, questions: dict) -> tuple[str, dict]:
    """Build one prompt asking for all of a ticket's questions as one JSON
    object, generically from `questions` (the same criteria Jev scores) --
    not hardcoded to this use case's three specific fields."""
    field_specs = []
    schema_fields = {}
    for qid, q in questions.items():
        if q["type"] == "choice":
            options = ", ".join(f'"{k}"' for k in q["criteria"])
            lines = "\n".join(f"  - {name}: {render_entry(desc)}" for name, desc in q["criteria"].items())
            field_specs.append(f'"{qid}" (one of {options}):\n{lines}')
            schema_fields[qid] = f"one of {list(q['criteria'])}"
        elif q["type"] == "score":
            lines = "\n".join(f"  {i}: {render_entry(desc)}" for i, desc in enumerate(q["criteria"]))
            field_specs.append(f'"{qid}" (integer level, see below):\n{lines}')
            schema_fields[qid] = f"integer 0-{len(q['criteria']) - 1}"
        else:  # noul
            field_specs.append(f'"{qid}" (true or false): {render_entry(q.get("instructions"))}')
            schema_fields[qid] = "true or false"

    schema_json = json.dumps(schema_fields, indent=2)
    prompt = (
        "You are a support-ticket triage assistant. Analyze the ticket below and respond with "
        "ONLY a single JSON object, no other text, no markdown fences, matching this shape:\n"
        f"{schema_json}\n\n"
        "Field definitions:\n" + "\n\n".join(field_specs) + "\n\n"
        f"Ticket:\n{state}\n\n"
        "Respond with the JSON object only."
    )
    return prompt, schema_fields


def parse_json_answer(text: str) -> tuple[dict | None, str | None]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, f"no JSON object found in: {text!r}"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc}): {text!r}"


def run_ticket(ticket: dict) -> dict:
    prompt, schema_fields = render_combined_prompt(ticket["state"], ticket["questions"])
    result = stream_call(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    answer, parse_error = parse_json_answer(result["text"])
    call = {
        "ticket_id": ticket["id"],
        "question_id": "combined",
        "raw_text": result["text"],
        "answer": answer,
        "parse_error": parse_error,
        "expected_fields": list(schema_fields),
        "ttft_ms": result["ttft_ms"],
        "decode_ms": result["decode_ms"],
        "total_ms": result["total_ms"],
        "call_error": result["error"],
        **kv_metrics(result["usage"]),
    }
    return call


def main() -> None:
    tickets = load_tickets()
    print(f"run_without_jev: {len(tickets)} tickets, one structured-JSON call each against {MODEL}\n")

    per_ticket: list[dict] = []
    all_calls: list[dict] = []
    workload_t0 = time.perf_counter()

    for ticket in tickets:
        call = run_ticket(ticket)
        all_calls.append(call)
        ticket_rollup = rollup([call])
        per_ticket.append({"ticket_id": ticket["id"], "calls": [call], "rollup": ticket_rollup})
        status = "OK" if call["parse_error"] is None else f"PARSE ERROR: {call['parse_error']}"
        print(
            f"  {ticket['id']:<16} prompt_tok={call['prompt_tokens']:>4} "
            f"cached={call['cached_tokens']:>4} ({call['cache_hit_pct']:.1f}%) "
            f"completion_tok={call['completion_tokens']:>3} "
            f"ttft_ms={call['ttft_ms']:.0f} decode_ms={call['decode_ms']:.0f} "
            f"-- {status}: {call['answer']}"
        )

    workload_wall_ms = (time.perf_counter() - workload_t0) * 1000
    workload_rollup = rollup(all_calls)
    workload_rollup["measured_wall_ms"] = workload_wall_ms

    payload = {
        "workload": "without_jev",
        "model": MODEL,
        "tickets": per_ticket,
        "rollup": workload_rollup,
    }
    out = write_json("without_jev.json", payload)
    print(f"\nWorkload totals: {workload_rollup}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
