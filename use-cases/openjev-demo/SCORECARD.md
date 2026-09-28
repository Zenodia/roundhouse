<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scorecard — openjev-demo

Fitness of this use case for the current roundhouse codebase. Each dimension is scored 0–3:
- **3** Strong fit — the capability exists and is exercised.
- **2** Moderate fit — the capability exists but needs wiring or configuration.
- **1** Weak fit — the capability is partially built or applies only in a constrained way.
- **0** Not applicable — the dimension does not apply, or the capability is entirely absent.

Max score: 24. See `use-cases/README.md` for band interpretations.

---

## Scores

| # | Dimension | Current | Target | Notes |
|---|---|---|---|---|
| 1 | Cache utilization | **3** | 3 | Real run (2026-09-28): cache% climbs 0% → ~62-64% by the second question, ~40-69% by the third — a strong effect despite a much shorter prefix than `cache-aware-routing`'s corpus, because `block_size: 64` is crossed easily even by a one-sentence ticket + system preamble |
| 2 | Routing coverage | **3** | 3 | Local-only policy, actually exercised end to end — the strongest dimension of this use case, and a contrast `cache-aware-routing` didn't have until this same round of work |
| 3 | Session durability | **1** | 1 | 3 turns per ticket, not a long growing conversation; not this use case's point |
| 4 | Budget sensitivity | **0** | 0 | Deliberately not applicable — $0 frontier spend by design, see README.md Notes |
| 5 | MCP control surface | **0** | 1 | Not called by `run.py`; same gap `cache-aware-routing` had at Phase 0 |
| 6 | Validate / steer loop | **0** | 0 | No trigger signals in bounded classification; unlikely to ever apply here |
| 7 | Multi-agent / multi-user | **0** | 0 | Single project/user; no Tax-B-shaped claim this use case makes (see GAPS.md) |
| 8 | Implementation readiness | **3** | 3 | Routing, cache reuse, and calibrated log-probability scoring are all real and working — `/v1/local/score`, 2026-09-28 (INTEGRATION.md Gap 1, closed) |
| | **Total** | **10 / 24** | **11 / 24** | A narrow, deliberately-scoped use case — most zeros are "not applicable by design," not "not built" |

---

## Dimension rationale

### 1. Cache utilization — 3 / 3

`tickets.jsonl`'s `state` fields are one to two sentences — far shorter than
`cache-aware-routing`'s ~4 KB corpus, and well under the `min_prefix_tokens: 1024` threshold
that gates the *frontier* `inactivity_decay` cache model. That threshold is irrelevant to this
use case's actual path, though: local-tier cache hits come from real KV block matches
(`block_size: 64`), not from `min_prefix_tokens`. A real run (2026-09-28, `PLAN.md`'s Phase 0
table) confirmed the effect is strong despite the short prefix: cache% climbs from 0% on each
ticket's first question to ~62-64% on the second and ~40-69% on the third, because the
preamble + `state` text alone is enough to cross one 64-token block. Only 3 questions per
ticket rather than `cache-aware-routing`'s 20 turns, but the *rate* of reuse is comparably
strong, which is what this dimension actually measures.

### 2. Routing coverage — 3 / 3

`control-plane.json`'s `policy.allow: ["local/*"]"` makes local the *only* admissible target,
and it is real, working routing — the exact `HttpLocalExecutor`/`EmbeddedFleet` wiring
`cache-aware-routing`'s `PROGRESS_TRACKER.md` validated end to end. This is the dimension this
use case exists to showcase.

### 3. Session durability — 1 / 3

Each ticket is 3 turns; there is no long-running, many-turn session in this use case's design.
Not a weakness relative to its purpose, just a dimension it doesn't stress.

### 4. Budget sensitivity — 0 / 3

Not applicable by design: this use case never routes to the frontier, so there is no cost
gradient to demonstrate and no correlary pairing worth sourcing real prices for. Scoring this
a 0 is not a gap to close — see `INTEGRATION.md`'s priority table, which doesn't list it.

### 5. MCP control surface — 0 / 3

`run.py` calls only `/v1/responses`, the same starting point `cache-aware-routing` had before
its own Phase 1. Unexplored here; would need its own justification (what would `declare_intent`
or `explain_last_route` add to a bounded-classification demo?) before it's worth building.

### 6. Validate / steer loop — 0 / 3

Bounded classification over a fixed label set has no obvious trigger signal (`PingPong`,
`ToolFailureStreak`, …) to fire on. Likely a permanent 0 for this use case's shape, not a gap.

### 7. Multi-agent / multi-user — 0 / 3

One project, one user. This use case makes no Tax-B-shaped claim (see GAPS.md) — there is
nothing analogous to demonstrate here the way a shared corpus prefix across users would be.

### 8. Implementation readiness — 3 / 3

Everything needed to route a real turn to a real local worker is built and proven
(`cache-aware-routing`'s work, inherited unmodified), and, as of 2026-09-28, so is calibrated,
`open-jev`-faithful scoring: `POST /v1/local/score` returns real log-probability-derived
`probability`/`confidence` values, verified against all 5 tickets (`PLAN.md`'s Phase 1 record).
"Reproduces `open-jev`'s System One contract" is no longer only approximately true — the
prompt renderers and the answer math are `open-jev`'s own, ported verbatim, scoring real
log-probabilities from the local worker rather than a parsed guess.
