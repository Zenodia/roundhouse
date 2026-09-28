<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Implementation Plan — openjev-demo

---

## Objective

Demonstrate roundhouse routing real turns to a locally served Dynamo model for a
Jev/`open-jev`-shaped bounded-decision workload (`state` + typed questions), with a real,
provider-measured KV-cache-reuse effect across a ticket's questions, and an honest account of
where the demo's answer fidelity diverges from `open-jev`'s own calibrated scoring.

---

## Deployment topology

Local-only (Shape B, no frontier fallback) — see README.md. This use case assumes
`cache-aware-routing`'s Dynamo worker is already up; it does not stand up its own.

---

## Prerequisites

| Prerequisite | Notes |
|---|---|
| `cache-aware-routing`'s Dynamo worker running | See its `DYNAMO_LOCAL_SERVING.md`; this use case reuses it as-is |
| roundhouse built with the local-fleet wiring | Present on `main` as of this session; `crates/roundhouse-server/src/local_fleet.rs` |
| `python use-cases/openjev-demo/mint_keys.py` run | Same mechanism as every other use case |

---

## Phase 0 — Prove the routing and the cache-reuse shape (done, 2026-09-28)

**Goal:** Confirm the local-only policy actually holds, and that a ticket's questions show
real cache reuse.

**Steps:**
1. `mint_keys.py`, launch roundhouse with the four `ROUNDHOUSE_LOCAL_*` variables and
   `ROUNDHOUSE_CATALOG`/`ROUNDHOUSE_CONTROL_PLANE` pointed at this use case's files.
2. `python use-cases/openjev-demo/run.py`.
3. Confirm: every ticket's first question is cold, the second and third climb; `/v1/metrics`
   attributes every call to `"provider": "dynamo"`.

**Result — real run, live Qwen2.5-Coder-32B worker:**

| ticket | question | in_tok | cached | cache% | answer |
|---|---|---|---|---|---|
| stripe-outage | department | 141 | 0 | 0.0% | technical |
| stripe-outage | frustration | 200 | 128 | 64.0% | 0 |
| stripe-outage | is_urgent | 374 | 192 | 51.3% | yes |
| plan-question | department | 148 | 0 | 0.0% | sales |
| plan-question | frustration | 207 | 128 | 61.8% | 0 |
| plan-question | is_urgent | 491 | 192 | 39.1% | yes |
| wrong-invoice | department | 145 | 0 | 0.0% | billing |
| wrong-invoice | frustration | 204 | 128 | 62.7% | 2 |
| wrong-invoice | is_urgent | 361 | 192 | 53.2% | no |
| api-error-500 | department | 148 | 0 | 0.0% | technical |
| api-error-500 | frustration | 207 | 128 | 61.8% | 1 |
| api-error-500 | is_urgent | 279 | 192 | 68.8% | yes |
| feature-request | department | 144 | 0 | 0.0% | technical |
| feature-request | frustration | 203 | 128 | 63.1% | 0 |
| feature-request | is_urgent | 295 | 192 | 65.1% | yes |

`/v1/metrics`: 15/15 calls attributed to `"provider": "dynamo"`, `"mode": "local"` — zero fell
over to frontier. Every answer parsed to a real label, zero `UNPARSED(...)`. No changes to
`run.py` were needed after the first real run.

**Gaps closed:** none new — this phase exercises `cache-aware-routing`'s existing local-tier
work against a new workload shape, and is itself the validation that the shape works.

---

## Phase 1 — Answer fidelity (done, 2026-09-28)

**Goal:** Close `INTEGRATION.md`'s Gap 1 — calibrated, `open-jev`-comparable
`probability`/`confidence` per answer instead of a parsed text label.

**What was built (Option A from INTEGRATION.md):**
1. `POST /v1/local/score` (`crates/roundhouse-server/src/local_score.rs`), a new local-tier-only
   endpoint accepting `{context, options}`, mounted only when a local fleet is configured
   **and** `ROUNDHOUSE_LOCAL_ENABLE_SCORE` is set — a second, independent opt-in (added
   2026-09-28 after the first cut mounted it unconditionally whenever a local fleet existed,
   which would have given `cache-aware-routing` an unused route for free; see `INTEGRATION.md`
   Gap 1's second addendum) so a deployment that never calls this route doesn't carry it.
2. Real log-probability scoring against Dynamo's `/v1/completions`, via
   `"prompt_logprobs": 1` + `"nvext": {"extra_fields": ["prompt_logprobs"]}` — the mechanism
   that actually works, found by ruling out two others first (`echo`, unsupported by Dynamo's
   completions route at all; and Dynamo's own experimental `/inference/v1/generate`, whose
   Rust-side plumbing is real but which no shipped `dynamo.vllm` launcher path ever makes
   routable). See `local_score.rs`'s own doc comment and `INTEGRATION.md` Gap 1's addendum for
   the full record, including the prefix-caching-truncates-the-response finding caught by an
   actual failure during validation (a cached prefix means `prompt_logprobs` comes back shorter
   than the prompt — aligned from the end of the array, not from context length, as a result).
3. `run_score.py` — `open-jev`'s own `render_choice`/`render_score`/`render_noul` and
   `system_one()` (softmax over log-probs, `score` as a probability-weighted mean level index,
   `confidence` as 1 minus normalised entropy) ported verbatim from
   `open-jev/openjev/systemone.py`, calling `/v1/local/score` instead of a local `OptionScorer`.

**Result — real run, live Qwen2.5-Coder-32B worker (2026-09-28):**

| ticket | department | frustration (score) | is_urgent (p yes) |
|---|---|---|---|
| stripe-outage | technical (conf 1.000) | 1.992 (conf 0.959) | 0.693 |
| plan-question | sales (conf 1.000) | 0.000 (conf 0.997) | 0.000 |
| wrong-invoice | billing (conf 1.000) | 1.107 (conf 0.691) | 0.000 |
| api-error-500 | technical (conf 1.000) | 1.893 (conf 0.690) | 0.693 |
| feature-request | technical (conf 1.000) | 0.000 (conf 0.999) | 0.000 |

Notably better-differentiated than `run.py`'s free-text approximation: `is_urgent` now cleanly
separates the two genuinely urgent tickets (p≈0.69) from the three that aren't (p=0.000),
rather than the free-text run's blanket `yes` on 4 of 5 tickets including the "no rush" feature
request.

**Gaps closed:** `INTEGRATION.md` Gap 1, fully.

**What Phase 1 does not close:** `/v1/local/score` bypasses the turn engine (no session, no
`/v1/metrics` accounting for these calls) — deliberate, see `GAPS.md`. `run.py`'s free-text
path is kept, not replaced, as the Option C comparison point `INTEGRATION.md` names.

---

## Phase 2 — Tokenomics benchmark: does the marketing claim hold? (done, 2026-09-28)

**Goal:** Measure TypeSafe's own "ultra-low latency, massive cost reduction (tokens + KV-cache
reuse)" claim for Jev/System One against this use case's real local worker, per LLM call, not
just assert it.

**Steps:** `run_with_openjev.py` and `run_without_jev.py` (direct to Dynamo, streaming,
instrumented per call — see `bench_common.py`), run against the same 5 tickets with the Dynamo
worker **cold-restarted before each condition** (`cache-aware-routing/shutdown_served_model.sh
--worker` then `serve_model.sh`) so neither run's KV cache could bias the other.

**Result:**

| Claim | Verdict | Evidence |
|---|---|---|
| Ultra-low latency | ✅ Confirmed — **2.3× faster** (311 ms/ticket vs 712 ms/ticket) | Zero decode time for Jev (only ever generates 1 throwaway token) vs 608 ms/ticket of genuine sequential decode for the JSON baseline |
| Massive token/cache-cost reduction | ❌ Not confirmed — **the opposite measured**: 2.1× more total tokens, 2.6× more newly-processed tokens per ticket for Jev | `/v1/local/score`'s one-call-per-option design resends nearly the full context per option; block-size-64 granularity means short contexts (e.g. `is_urgent`'s 45 tokens) never cache at all |

**Gaps closed:** none — this phase is a measurement, and it surfaced a new one (`GAPS.md`'s
new P1 row, `INTEGRATION.md` Gap 1's Option D: a batched multi-option scoring call, not built).

**Full numbers and mechanism:** `README.md`'s "Tokenomics" section; raw data in `results/`.

---

## Score trajectory

| After phase | Expected score | Key unlock |
|---|---|---|
| Phase 0 | 8 / 24 | Local-only routing proven for a new workload shape |
| Phase 1 | 10 / 24 | Calibrated answers close Implementation readiness to 3/3 |
| Phase 2 (this round) | 10 / 24 (no score dimension changes) | Measured, not assumed, tokenomics — one real gap found and tracked |
