<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# use-case: openwiki-adversarial

Runnable harness for the OpenWiki adversarial-documentation experiment. The design ruling
this implements is `agent-docs/PLAN-openwiki-adversarial.md`; read it first — this directory
is the *mechanism*, that document is the *why*.

> Status: **skeleton**. Design is ruled; harness is not built yet. This README fixes the layout
> and the run contract so the milestones in the plan (§10) have a place to land.

## What this scenario drives

An unmodified coding agent (the *attacker*) reaches its frontier model **through roundhouse**,
and attacks a deliberately-vulnerable [Outline](https://github.com/outline/outline) instance
produced by [marinade](https://github.com/) (local clone). We measure whether an
[OpenWiki](https://github.com/langchain-ai/openwiki)-generated wiki helps the attacker, and how
three mitigations (content redaction, access gating, roundhouse turn-steering) and finally a
code fix drive exploitation back to zero.

Roundhouse is **consumed, not modified** — this scenario is orchestration that points an agent
at a running `roundhouse-server`, mirroring `use-cases/cache-aware-routing/` (a keys file for
the control-plane arm plus scripts).

## Planned layout (created as milestones land)

```
use-cases/openwiki-adversarial/
  README.md                 # this file
  keys.local.json           # control-plane keys for the "gated" arm (Axis B), gitignored secrets
  env.example               # roundhouse + marinade + openwiki env for a run
  sut/                      # bring vulnerable Outline up/down (wraps marinade app.sh + compose)
  wiki/                     # generate full wiki; produce redacted variant + redaction-report.md
  attacker/                 # the agent task + budget/prompt, pointed at roundhouse
  scorer/                   # runs marinade exploit.sh / functional-test.sh; folds roundhouse log
  matrix.py                 # runs the C0..C4 condition matrix, N runs each
  results/                  # per-run metrics + the before/after hardening table (gitignored)
```

## External dependencies (not vendored here)

- **roundhouse-server** — built from this workspace; the attacker's model surface (Channel 1).
- **marinade** — local clone; supplies the prepared Outline seed, the three injected vuln
  patches, and the `exploit.sh` / `functional-test.sh` oracle.
- **OpenWiki** — `npm i -g openwiki`; generates the code wiki over the vulnerable Outline repo.
- **docker** — the Outline compose stack (redis + postgres + outline).
- **a logging proxy** (mitmproxy) — the attacker→Outline wire (Channel 2).

## Run contract (target, once built)

`matrix.py` will: boot roundhouse, boot vulnerable Outline behind the proxy, (per condition)
stand up the wiki surface, run the attacker at a fixed budget, then score with the marinade
oracle and pull the roundhouse session log for the trajectory. It never hands the attacker any
marinade artifact — discovery must be earned.
