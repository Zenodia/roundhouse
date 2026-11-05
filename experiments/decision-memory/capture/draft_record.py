#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The capture step: a meeting note in, a draft decision record out.

This is the piece that fixes the broken pipeline. A decision taken in a
room becomes a `status: proposed` record in `decisions/`, opened as a pull
request that the decision owner — not an engineer — approves. Until the
owner flips it to `accepted`, nothing reads it as a rule.

    # with a model (any OpenAI-compatible chat endpoint, or Roundhouse)
    python -m capture.draft_record ledgerline/meeting-notes/2026-11-04-risk-committee.md \
        --client chat:https://integrate.api.nvidia.com --model <model>

    # without a model, for the demo and the CPU test: a deterministic draft
    python -m capture.draft_record ledgerline/meeting-notes/2026-11-04-risk-committee.md --client stub

The draft is written to `decisions/DR-<next>-<slug>.md` and the path is
printed. Then:

    git checkout -b decision/DR-0011 && git add decisions && git commit -m "DR-0011: proposed"
    # open the PR; CODEOWNERS routes it to the owners; the owner edits and accepts:
    python -m ledgerline.decisions accept DR-0011
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FIXTURE_REPO = EXPERIMENT / "ledgerline"
sys.path.insert(0, str(FIXTURE_REPO))
sys.path.insert(0, str(EXPERIMENT))

from ledgerline.decisions import current, load_records  # noqa: E402
from runner.clients import make_client  # noqa: E402

PROMPT = """You turn a meeting note into a decision record for a software team's `decisions/` folder.

A decision record states ONE rule a system can follow, with its owner (the person who decided and can
revoke it), dates, scope (the source files it governs, as globs), machine-readable parameters, and the
reasoning in the deciders' own words. It replaces an older record when the note says so.

Output the record as Markdown with YAML front matter in exactly this shape, and nothing else:

{template}

Fill `id` with `{next_id}`. Set `status: proposed` (a human accepts it). Set `supersedes` to the id of
the current record this replaces, if any, else null; leave `superseded_by: null`. `parameters` must use
the same keys as the record it supersedes when the meaning is the same (so tests keep working); here are
the current accepted records' ids, titles and parameters:

{current}

If the note records no decision (only a proposal or a discussion), still produce the record with
`status: proposed` and say in the Decision section that nothing was decided yet.

Meeting note:
---
{note}
---
"""


def next_id(records) -> str:
    n = max((int(r.id.split("-")[1]) for r in records), default=0) + 1
    return f"DR-{n:04d}"


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    if len(s) > 60:
        s = s[:60].rsplit("-", 1)[0]
    return s


def stub_draft(note: str, nid: str, records) -> str:
    """A deterministic draft for the sample 2026-11-04 note, so the capture
    -> accept -> retest loop runs without a model."""
    if "14 days" not in note:
        raise SystemExit("the stub only knows the 2026-11-04 risk-committee note; use a model for others")
    return f"""---
id: {nid}
title: Refunds requested more than 14 days after capture need manager approval
status: proposed
owner: Lena Hoffmann (Head of Payments Risk)
decided_on: 2026-11-04
effective_from: 2026-11-04
supersedes: DR-0005
superseded_by: null
scope:
  - ledgerline/refunds.py
  - ledgerline/payments.py
tags: [refunds, risk, approval]
parameters:
  refund_approval_after_days: 14
  refund_cap: captured_minus_prior_refunds
source: meeting-notes/2026-11-04-risk-committee.md
---

## Decision

A refund requested more than 14 days after the payment's `captured_at`
requires an explicit manager approval flag; without it the request is
refused. The cap (captured amount minus prior refunds) is unchanged.

## Why

Q3 chargeback data shows friendly-fraud refunds clustering from day 15
after capture, down from day 30 in May, because faster shipping puts
goods in customers' hands by day 3–5 and statements by day 14.

## Constraints

- Measured from `captured_at`.
- Approval is a flag on the request, not a queue.
- Effective immediately for new refund requests.

## What it replaces

DR-0005's 30-day threshold. Everything else in DR-0005 stands.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("note", help="path to the meeting note")
    p.add_argument("--client", default="stub", help="stub | chat:<url> | responses:<url>")
    p.add_argument("--model", default=None)
    p.add_argument("--decisions", default=str(FIXTURE_REPO / "decisions"))
    p.add_argument("--out", default=None, help="output path (default decisions/DR-<next>-<slug>.md)")
    args = p.parse_args()

    note = Path(args.note).read_text(encoding="utf-8")
    records = load_records(args.decisions)
    nid = next_id(records)
    if args.client == "stub":
        draft = stub_draft(note, nid, records)
    else:
        template = (Path(args.decisions) / "TEMPLATE.md").read_text(encoding="utf-8")
        cur = "\n".join(f"- {r.id}: {r.title}  parameters={r.parameters}" for r in current(records))
        client = make_client(args.client, args.model, Path("."), [])
        reply = client.complete("You write precise decision records.", PROMPT.format(template=template, next_id=nid, current=cur, note=note), "capture")
        if reply.error:
            raise SystemExit(reply.error)
        draft = reply.text.strip()
        m = re.search(r"---\n.*?\n---\n.*", draft, re.S)
        if not m:
            raise SystemExit("model did not return a record:\n" + draft[:800])
        draft = m.group(0) if m.start() == 0 else draft[m.start():]
    fm = yaml.safe_load(draft.split("---", 2)[1])
    if fm.get("status") != "proposed":
        raise SystemExit("a drafted record must be `status: proposed`; a human accepts it")
    out = Path(args.out) if args.out else Path(args.decisions) / f"{nid}-{slug(fm['title'])}.md"
    out.write_text(draft if draft.endswith("\n") else draft + "\n", encoding="utf-8")
    print(f"drafted {out}  (status: proposed; supersedes: {fm.get('supersedes')})")
    print("next: open a PR; the owner reviews, then `python -m ledgerline.decisions accept " + nid + "`")


if __name__ == "__main__":
    main()
