#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the openjev-demo: TypeSafe-System-One-shaped ticket triage against
roundhouse's local Dynamo tier.

What this is
-------------
`open-jev` (https://github.com/ ... /open-jev, local clone at ../../open-jev
relative to this file's usual location) implements TypeSafe's "System One"
contract (https://docs.typesafe.ai) locally: given a `state` and a set of
typed `questions` (`choice` / `score` / `noul`), it answers each by scoring
pre-written option labels as continuations with a local Gemma 3 model and
reading off calibrated probabilities from the raw log-probabilities -- no
decoding.

This use case asks the same *shape* of question -- one `state` (a support
ticket), several typed questions about it (`department`, `frustration`,
`is_urgent`), read `tickets.jsonl` for the exact set, modeled directly on
`open-jev`'s own `examples/systemone-quickstart.json` -- but answers them
through roundhouse's ordinary `/v1/responses` turn API, routed to the local
Dynamo/Qwen worker this repo's cache-aware-routing use case already validates.

**This is a deliberate approximation, not a re-implementation of open-jev's
math, and GAPS.md says so.** roundhouse's wire protocols carry text, not
token-level log-probabilities, so there is no `probability` or `confidence`
field here the way open-jev's `ChoiceAnswer`/`ScoreAnswer` have one -- the
model is asked to answer with the option label on its own line, and that
line is parsed. What IS real: the question prompts share the ticket's
`state` as a common prefix (see `render_*` below, ported from
`open-jev/openjev/systemone.py`'s own renderers), so three questions about
one ticket sent as three turns of one roundhouse session is a genuine
KV-cache-reuse workload -- roundhouse's real, provider-reported `cached_tokens`
climbing turn to turn, the same measurement cache-aware-routing's `run.py`
makes, applied to a different workload shape.

Prereqs: roundhouse running with the local fleet wired to a Dynamo worker
(see README.md), and `python use-cases/openjev-demo/mint_keys.py` already run.

Usage:
    python use-cases/openjev-demo/run.py
    ROUNDHOUSE_URL=http://127.0.0.1:8080 python use-cases/openjev-demo/run.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_URL = os.environ.get("ROUNDHOUSE_URL", "http://127.0.0.1:8080").rstrip("/")
TURN_KEY_HEADER = "x-roundhouse-key"

SYSTEM_PREAMBLE = (
    "You are a support-ticket triage assistant. You will be asked several "
    "questions about the same ticket. For each question, answer with ONLY "
    "the exact option key or number from the list, on its own line, and "
    "nothing else -- no punctuation, no explanation.\n\n"
)


# ------------------------------------------------------ rendering (ported from
# open-jev/openjev/systemone.py's render_choice/render_score/render_noul, so
# the prompt shape this use case scores against matches what open-jev itself
# asks -- only the scoring mechanism differs, not the question.)
def render_entry(e: Any) -> str:
    if e is None:
        return ""
    if isinstance(e, str):
        return e
    return json.dumps(e, indent=2, ensure_ascii=False)


def _block(title: str, e: Any) -> str:
    body = render_entry(e)
    return f"{title}:\n{body}\n\n" if body else ""


def render_choice(q: dict) -> tuple[str, list[str]]:
    lines = [f"- {name}: {render_entry(desc)}" for name, desc in q["criteria"].items()]
    prompt = (
        _block("Question", q.get("instructions"))
        + "Choose exactly one option.\nOptions:\n" + "\n".join(lines)
        + "\n\nAnswer with the option key only:\n"
    )
    return prompt, list(q["criteria"])


def render_score(q: dict) -> tuple[str, list[str]]:
    lines = [f"{i}: {render_entry(desc)}" for i, desc in enumerate(q["criteria"])]
    prompt = (
        _block("Question", q.get("instructions"))
        + "Rate on the following ordered levels.\nLevels:\n" + "\n".join(lines)
        + "\n\nAnswer with the level number only:\n"
    )
    return prompt, [str(i) for i in range(len(q["criteria"]))]


def render_noul(q: dict) -> tuple[str, list[str]]:
    prompt = _block("Question", q.get("instructions")) + "Answer yes or no.\n\nAnswer:\n"
    return prompt, ["yes", "no"]


def render_question(q: dict) -> tuple[str, list[str]]:
    return {"choice": render_choice, "score": render_score, "noul": render_noul}[q["type"]](q)


def parse_answer(text: str, labels: list[str]) -> str:
    """Best-effort extraction of one of `labels` from the model's free-text reply.

    Not calibrated, not a probability -- see the module docstring. Exact
    (case-insensitive) match on the first non-empty line wins; failing that,
    the first label found anywhere in the reply; failing that, the reply is
    reported verbatim so a bad prompt fails loudly instead of silently
    defaulting to labels[0].
    """
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    norm = first_line.strip(" .:\"'`").lower()
    for label in labels:
        if norm == label.lower():
            return label
    lowered = text.lower()
    for label in labels:
        if label.lower() in lowered:
            return label
    return f"UNPARSED({first_line!r})"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tickets() -> list[dict]:
    return [json.loads(line) for line in (HERE / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def stream_turn(secret: str, instructions: str, conversation: list[dict], cache_key: str) -> dict:
    body = json.dumps(
        {"instructions": instructions, "input": conversation, "stream": True, "prompt_cache_key": cache_key}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/responses", data=body, method="POST",
        headers={"content-type": "application/json", TURN_KEY_HEADER: secret},
    )
    text_parts: list[str] = []
    usage: dict = {}
    error: str | None = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                if etype == "response.output_text.delta":
                    text_parts.append(event.get("delta", ""))
                elif etype == "response.completed":
                    usage = event.get("response", {}).get("usage", {}) or {}
                elif etype == "response.failed":
                    error = json.dumps(event.get("response", {}).get("error", event))
    except urllib.error.HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}"
    except urllib.error.URLError as exc:
        error = f"connection error: {exc}. Is roundhouse running at {BASE_URL}?"
    return {"text": "".join(text_parts), "usage": usage, "error": error}


def run_ticket(secret: str, ticket: dict) -> dict:
    tid = ticket["id"]
    print(f"\n=== ticket: {tid} ===")
    print(f"state: {ticket['state']!r}")
    print(f"{'question':>12}  {'in_tok':>8}  {'cached':>8}  {'cache%':>7}  answer")
    instructions = SYSTEM_PREAMBLE + f"State:\n{ticket['state']}\n\n"
    conversation: list[dict] = []
    answers: dict[str, str] = {}
    totals = {"input": 0, "cached": 0}
    for qid, q in ticket["questions"].items():
        prompt, labels = render_question(q)
        conversation.append({"type": "message", "role": "user", "content": prompt})
        result = stream_turn(secret, instructions, conversation, cache_key=tid)
        if result["error"]:
            print(f"{qid:>12}  ERROR: {result['error']}")
            return {"id": tid, "answers": answers, "totals": totals}
        usage = result["usage"]
        in_tok = int(usage.get("input_tokens", 0))
        cached = int(usage.get("input_tokens_details", {}).get("cached_tokens", 0))
        pct = (100.0 * cached / in_tok) if in_tok else 0.0
        totals["input"] += in_tok
        totals["cached"] += cached
        answer = parse_answer(result["text"], labels)
        answers[qid] = answer
        print(f"{qid:>12}  {in_tok:>8}  {cached:>8}  {pct:>6.1f}%  {answer}")
        conversation.append({"type": "message", "role": "assistant", "content": result["text"]})
    return {"id": tid, "answers": answers, "totals": totals}


def fetch_metrics(admin_secret: str | None) -> None:
    headers = {TURN_KEY_HEADER: admin_secret} if admin_secret else {}
    req = urllib.request.Request(f"{BASE_URL}/v1/metrics", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            snapshot = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError) as exc:
        print(f"\n/v1/metrics -> {exc}")
        return
    print("\n=== /v1/metrics snapshot ===")
    print(json.dumps({"models": snapshot.get("models"), "savings": snapshot.get("savings")}, indent=2))
    print(f"\nOpen the live dashboard at {BASE_URL}/v1/metrics/dashboard")


def main() -> None:
    keys_file = HERE / "keys.local.json"
    if not keys_file.exists():
        sys.exit("keys.local.json not found -- run `python use-cases/openjev-demo/mint_keys.py` first.")
    secrets_map = load_json(keys_file)
    admin_secret = secrets_map.get("__admin__")
    tickets = load_tickets()

    plane = load_json(HERE / "control-plane.json")
    memberships = [(k["project"], k["user"]) for k in plane.get("keys", [])]
    if not memberships:
        sys.exit("control-plane.json has no keys; run mint_keys.py")
    project, user = memberships[0]
    secret = secrets_map.get(f"{project}/{user}")
    if not secret:
        sys.exit(f"no secret for {project}/{user} in keys.local.json")

    print(f"roundhouse: {BASE_URL}")
    print(f"{len(tickets)} tickets, {len(tickets[0]['questions'])} typed questions each")
    print("Watch 'cached' climb across a ticket's questions -- they share the same state prefix.")

    results = [run_ticket(secret, t) for t in tickets]

    print("\n=== answers ===")
    for r in results:
        print(f"  {r['id']}: {r['answers']}")

    fetch_metrics(admin_secret)


if __name__ == "__main__":
    main()
