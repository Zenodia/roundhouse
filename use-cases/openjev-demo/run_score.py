#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive the openjev-demo ticket set through roundhouse's real logprob scorer.

What changed from `run.py`
---------------------------
`run.py` (still here, still valid) approximates `open-jev`'s System One
contract with a prompted free-text answer over `/v1/responses` -- documented
in GAPS.md as Gap 1: no calibrated `probability`/`confidence`, and an answer
that can fail to parse. This script is what closes that gap: it calls
roundhouse's `POST /v1/local/score` (`crates/roundhouse-server/src/
local_score.rs`), which returns the *real* sum of
`log p(option token | context)` for each candidate label, computed against
the local Dynamo worker's `nvext.prompt_logprobs` -- not a parsed guess.

The `render_choice`/`render_score`/`render_noul` prompt renderers and the
`system_one()` composition (softmax over log-probs, `score` as a
probability-weighted mean level index, `confidence` as 1 minus normalised
entropy) are a direct port of `open-jev/openjev/systemone.py`'s own
functions of the same names -- same prompts, same math, same TypeSafe
System One answer shape (`ChoiceAnswer`/`ScoreAnswer`/`NoulAnswer`). The only
difference from running `open-jev` itself is where the log-probabilities
come from: its own local Gemma process there, roundhouse's `/v1/local/score`
against the local Dynamo/Qwen worker here.

Prereqs: roundhouse running with the local fleet wired in (see README.md),
and `python use-cases/openjev-demo/mint_keys.py` already run.

Usage:
    python use-cases/openjev-demo/run_score.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_URL = os.environ.get("ROUNDHOUSE_URL", "http://127.0.0.1:8080").rstrip("/")
TURN_KEY_HEADER = "x-roundhouse-key"


# ------------------------------------------------------ rendering (ported from
# open-jev/openjev/systemone.py's render_choice/render_score/render_noul,
# verbatim in shape -- these are the exact prompts open-jev itself scores.)
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


# ------------------------------------------------------------------- maths
# Ported verbatim from open-jev/openjev/systemone.py's confidence().
def confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


def load_tickets() -> list[dict]:
    return [json.loads(line) for line in (HERE / "tickets.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def call_score(secret: str, context: str, options: list[str]) -> dict:
    body = json.dumps({"context": context, "options": options}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/local/score", data=body, method="POST",
        headers={"content-type": "application/json", TURN_KEY_HEADER: secret},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        sys.exit(f"/v1/local/score -> HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        sys.exit(f"/v1/local/score -> connection error: {exc}. Is roundhouse running at {BASE_URL} "
                  f"with ROUNDHOUSE_LOCAL_* set (real log-prob scoring is local-tier only)?")


def system_one(secret: str, state: str, questions: dict) -> dict:
    """Port of open-jev/openjev/systemone.py's system_one(), against roundhouse
    instead of a local scorer object."""
    answers: dict[str, Any] = {}
    for qid, q in questions.items():
        prompt, labels = render_question(state, q)
        result = call_score(secret, prompt, labels)
        probs = {opt["option"]: opt["probability"] for opt in result["options"]}
        ordered_probs = [probs[label] for label in labels]

        if q["type"] == "choice":
            best = max(labels, key=lambda label: probs[label])
            answers[qid] = {
                "type": "choice", "choice": best,
                "probabilities": probs, "confidence": confidence(ordered_probs),
            }
        elif q["type"] == "score":
            answers[qid] = {
                "type": "score",
                "score": sum(i * p for i, p in enumerate(ordered_probs)),
                "confidence": confidence(ordered_probs),
                "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                "probabilities": {str(i): p for i, p in enumerate(ordered_probs)},
            }
        else:  # noul
            answers[qid] = {"type": "noul", "noul": probs["yes"]}
    return answers


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
    print(json.dumps({"models": snapshot.get("models")}, indent=2))


def main() -> None:
    keys_file = HERE / "keys.local.json"
    if not keys_file.exists():
        sys.exit("keys.local.json not found -- run `python use-cases/openjev-demo/mint_keys.py` first.")
    secrets_map = json.loads(keys_file.read_text(encoding="utf-8"))
    admin_secret = secrets_map.get("__admin__")
    plane = json.loads((HERE / "control-plane.json").read_text(encoding="utf-8"))
    project, user = plane["keys"][0]["project"], plane["keys"][0]["user"]
    secret = secrets_map.get(f"{project}/{user}")
    if not secret:
        sys.exit(f"no secret for {project}/{user} in keys.local.json")

    tickets = load_tickets()
    print(f"roundhouse: {BASE_URL}")
    print(f"{len(tickets)} tickets -- real log-probability scoring via /v1/local/score")
    print("(open-jev's own render_choice/render_score/render_noul + softmax + entropy confidence, ")
    print(" ported verbatim; the only difference from running open-jev itself is where the")
    print(" log-probabilities come from.)\n")

    for ticket in tickets:
        print(f"=== ticket: {ticket['id']} ===")
        print(f"state: {ticket['state']!r}")
        answers = system_one(secret, ticket["state"], ticket["questions"])
        for qid, answer in answers.items():
            if answer["type"] == "choice":
                probs = ", ".join(f"{k}={v:.3f}" for k, v in answer["probabilities"].items())
                print(f"  {qid:>12}: {answer['choice']:<10} confidence={answer['confidence']:.3f}  ({probs})")
            elif answer["type"] == "score":
                print(f"  {qid:>12}: score={answer['score']:.3f}  confidence={answer['confidence']:.3f}")
            else:
                print(f"  {qid:>12}: p(yes)={answer['noul']:.3f}")
        print()

    fetch_metrics(admin_secret)


if __name__ == "__main__":
    main()
