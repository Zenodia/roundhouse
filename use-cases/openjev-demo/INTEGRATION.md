<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Integration Paths — openjev-demo

**Audience:** roundhouse repo admin / milestone planner.

**Purpose:** This use case exposes one repo-level gap worth a real decision, unlike
`cache-aware-routing`'s Gaps 1–4 which this use case inherits already resolved. Presented the
same way: options, not a single recommendation baked in.

---

## Gap 1: No token-level log-probabilities anywhere on roundhouse's wire

**What a real Jev-style decision layer needs:** the log-probability of each candidate label
given the context, so a `probability`/`confidence` can be computed the way `open-jev`'s own
`OptionScorer` does — not a free-text answer to parse.

**What exists today:** `crates/roundhouse-server/src/engine.rs`'s `LocalExecution` carries
`{text, output_tokens, reasoning_tokens}` — no logprobs anywhere in the type. `HttpLocalExecutor`
(`crates/roundhouse-server/src/local_fleet.rs`) calls Dynamo's `/v1/completions`, which *can*
return `logprobs` per token (vLLM supports `echo=true, logprobs=N`), but the executor never
asks for them and nothing in the response wire (`/v1/responses`'s `Usage`, or the Anthropic
Messages equivalent) has a field to carry them even if it did.

**Option A — A dedicated `/v1/score` (or similar) endpoint, local-tier only:**
- What changes: A new route, parallel to `/v1/responses`, accepting `{context, options}` (or
  the full System-One `{state, questions}` shape) and returning per-option log-probabilities.
  Backing it needs a new `LocalExecutor`-adjacent trait (or an extension of the existing one)
  that requests `echo=true, logprobs=1` from the worker and sums the option-token slice — the
  same computation `open-jev`'s own `OptionScorer._score_with_prefix` does, just over HTTP
  instead of an in-process batched pass.
- Enables: Real calibrated answers, matching `open-jev`'s own contract shape closely enough
  that a client could plausibly point at either.
- Costs: A genuinely new surface — new wire types, new tests, a decision about whether this
  belongs on the frontier path too (some hosted APIs support `logprobs`; OpenAI's Responses
  API does not the way its old Completions API did, so a frontier equivalent is dialect-specific
  and not free).
- Forecloses: Nothing; this is additive.

**Option B — Extend `LocalExecution`/`Usage` with an optional logprobs field, keep one endpoint:**
- What changes: `LocalExecution` gains `token_logprobs: Option<Vec<f64>>`; `HttpLocalExecutor`
  requests them when a turn asks for them (a new field on the turn request); `/v1/responses`'
  wire carries them through when present.
- Enables: The existing turn API stays the single surface; a client that wants logprobs opts in
  per turn.
- Costs: Threads an optional field through several layers (`LocalExecution` →
  `Engine::local_stream` → `FrontierChunk`/`Usage` → the wire) for a feature only this use case's
  shape needs today — real risk of a field nothing else reads, which is exactly the kind of
  unused-abstraction cost CLAUDE.md's comment-and-abstraction discipline warns about.
- Forecloses: Nothing technically, but is a heavier touch on shared code for a narrower payoff
  than Option A.

**Option C — Stay with prompted classification (current):**
- What changes: Nothing. `run.py`'s `parse_answer` is the whole answer.
- Enables: What this use case already demonstrates — real local routing, real cache reuse —
  without touching shared engine code for one use case's need.
- Costs: No calibrated probability/confidence, and answers can fail to parse
  (`UNPARSED(...)`), which `open-jev`'s own scoring cannot do by construction (it always picks
  the highest-scoring label from a fixed set).
- Forecloses: Nothing; Option A or B can still be built later and this use case switched over.

**Admin recommendation:** Option C for now — this use case's job was to prove local routing for
a Jev-shaped workload, which it does. Option A if a second use case, or a real product need,
wants calibrated scores badly enough to justify a new endpoint; it is the cleaner boundary
(new surface, not a new optional field threaded through the general turn path) and does not
foreclose Option B later if the two ever need to converge.

*2026-09-28: Implemented, as Option A — `POST /v1/local/score`
(`crates/roundhouse-server/src/local_score.rs`), mounted only when a local fleet is configured.
Three real dead ends were hit and recorded in the module's own doc comment before landing on a
working mechanism, worth summarizing here because none of the three is discoverable from the
OpenAI spec or from Dynamo's own `/inference/v1/generate` module doc:*

- *`echo: true` on `/v1/completions`: not implemented by Dynamo's `NvCreateCompletionRequest` at
  all (the field is commented out in its own source) — silently dropped, not refused.*
- *`/inference/v1/generate`, Dynamo's own experimental token-in/token-out surface built for
  exactly this: its Rust-side `prompt_logprobs` plumbing is real and tested, but no code path in
  the pinned rev's shipped `dynamo.vllm` Python launcher ever advertises the
  `vllm_inference_v1_generate` runtime capability a worker needs to be routable — every request
  answered `"no generate-capable model is registered"` regardless of frontend flags.*
- *`prompt_logprobs` set directly on `/v1/completions`'s top level: accepted by request
  validation, computed by the worker, but never serialized back onto the top-level response.*

*What actually works: `prompt_logprobs` at the top level **and**
`"nvext": {"extra_fields": ["prompt_logprobs"]}` — Dynamo's own response-field-selection
mechanism. The worker always computes it; the request has to separately opt in to it appearing,
under `response.nvext.prompt_logprobs`, not the field name the request used to ask for it.*

*One more real finding, caught by an actual failure during end-to-end validation, not
anticipated in advance: with prefix caching on (this deployment's default), a cached block is
never recomputed, so `nvext.prompt_logprobs` comes back **shorter than the prompt** — only the
uncached tail. `local_score.rs` aligns from the end of the returned array rather than from the
context length for exactly this reason; see its own doc comment for the confirming measurement
(a 75-token prompt with one 64-token block already cached returned exactly 11 entries).*

*Verified end to end against real tickets — see `PLAN.md`'s Phase 1 record and `README.md`.
One thing this route does **not** do: it bypasses the turn engine entirely (no session, no
`/v1/metrics` accounting), which is a deliberate consequence of it being a local-tier utility
call rather than a billed turn — see this file's own Option A writeup on why a new surface was
chosen over threading a field through the shared turn path.*

*2026-09-28, second addendum: mounting `local_score_router` unconditionally whenever a local
fleet was configured meant `cache-aware-routing` — which wires a local fleet purely for turn
dispatch and never calls this route — would have gotten it for free too, an unused surface with
its own extra per-option HTTP cost. Fixed with a second, independent opt-in,
`ROUNDHOUSE_LOCAL_ENABLE_SCORE`, off by default: `main.rs` now only constructs the scorer and
merges its router when both `local_fleet::from_env` returned `Some` **and**
`local_score::enabled()` is true. Verified both states against a live worker
(`GET /v1/local/score` 404s with the flag unset, answers with the flag set) — see
`README.md`'s "The `ROUNDHOUSE_LOCAL_ENABLE_SCORE` flag" section.*

*2026-09-28, third addendum — a measured tokenomics result, not a design decision: benchmarked
`/v1/local/score`'s current one-call-per-option design against a single structured-JSON
generation call, same 5 tickets, cold-restarted worker before each (`use-cases/openjev-demo/
run_with_openjev.py` / `run_without_jev.py`, `results/summary.json`). Latency won decisively
(311 ms/ticket vs 712 ms/ticket — near-zero decode beats sequential decode). **Token cost did
not**: 2.1× more total tokens and 2.6× more newly-processed tokens per ticket, because each
option call resends nearly the full context and a context under one KV block (64 tokens) never
gets cached at all regardless of how many options share it. `open-jev`'s own `OptionScorer`
avoids exactly this by prefilling once and expanding that KV cache across every option in one
batched forward pass — this route's HTTP-callable, one-request-per-option shape is a different,
more expensive trade. A batched variant (Option D below) would need to exist for the
token/cache half of the original claim to hold; today, only the latency half does. See
`GAPS.md`'s new P1 row and `README.md`'s "Tokenomics" section for the full measurement.*

**Option D (not built) — batch every option of one question into a single scoring request:**
- What changes: `/v1/local/score`'s request shape would need `context` + `options` to
  translate into **one** completions call whose `prompt_logprobs` scoring shares one prefill
  across every option — either by having Dynamo's worker expand a single KV cache across a
  batch (mirroring `open-jev`'s own in-process approach) or, more simply, by padding the shared
  context to a full block boundary so it caches identically regardless of which option follows
  it, then dispatching the per-option calls concurrently rather than sequentially (does not fix
  the token-count story, but would close most of the *latency* gap the current 306 ms/ticket of
  pure per-call network overhead costs).
- Enables: The token/cache-efficiency half of the original claim, not just the latency half.
- Costs: Real engineering — either a Dynamo-side batched-prefill capability this pinned rev may
  not expose over HTTP at all (would need investigation), or a lower-effort padding + concurrent
  dispatch change that only partially addresses it.
- Forecloses: Nothing; `/v1/local/score`'s current shape is what `run_with_openjev.py` measured
  and can stay as the honest baseline for whatever variant is built next.

**Admin recommendation:** Not started. This is now a concretely measured, well-understood gap
rather than a hypothetical one — worth prioritizing over the "wait for a second use case" logic
this document's other options use, precisely because the data already shows the current design
does not deliver half of what this whole use case exists to demonstrate.

---

## Priority signal for the roadmap

| Suggestion | Target document | Priority |
|---|---|---|
| A dedicated logprob-scoring endpoint (Gap 1, Option A), if a second use case wants it | A new `agent-docs/PLAN-*.md` or as a line item in an existing one | **Low** — one use case wanting it is not yet a pattern |
| A batched multi-option scoring call (Gap 1, Option D) — measured, 2026-09-28, to be the difference between the token/cache half of the Jev claim holding or not | Same target as Option A, or a follow-on to it | **Medium** — a measured gap, not a hypothetical one; see this file's third addendum |
| Record "local-only policy, deliberately no frontier admissibility" as a supported, intentional shape (this use case is the first to ship one) | `use-cases/README.md` or a shared `CONSTRAINTS.md`, once a second local-only use case exists | **Low for now**, same "wait for a second data point" logic `cache-aware-routing`'s own `INTEGRATION.md` uses for its Gap 3 Option B |
