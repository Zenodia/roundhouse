<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# openjev-demo — a Jev-style decision layer over roundhouse's local tier

Demonstrates roundhouse routing real turns to a **locally served Dynamo model** for a
[`open-jev`](https://github.com/) / [TypeSafe System One](https://docs.typesafe.ai)-shaped
workload: a support ticket (`state`) and a handful of typed questions about it
(`department`: choice, `frustration`: score, `is_urgent`: noul), answered by a small local
model instead of a frontier round trip. `open-jev`'s own README frames this well: a *fast,
cheap, bounded decision layer* sitting in front of (or instead of) an agent's own reasoning —
the same category TypeSafe's own "Jev builds" gallery calls out repeatedly: ticket triage,
fraud/spam classification, content-relevance scoring — bounded decisions, not open-ended
generation.

## What this is

roundhouse dispatching real turns to a real Dynamo/Qwen worker (the exact one
`cache-aware-routing` validates end to end — see that use case's `PROGRESS_TRACKER.md`),
for a workload shaped like `open-jev`'s own `/v1/systemone` contract, with two things
genuinely real about it, not simulated:

1. **KV-cache reuse specific to this shape.** Three typed questions about one ticket, sent as
   three turns of one roundhouse session, share the ticket's `state` text as a common prefix
   — the same effect `cache-aware-routing`'s Tax A demonstrates, applied across *questions*
   rather than *turns of a conversation*. See `run.py`.
2. **`open-jev`'s own scoring math, for real.** `POST /v1/local/score`
   (`crates/roundhouse-server/src/local_score.rs`, added 2026-09-28) computes the exact
   log-probability of each candidate option's tokens given the context against the local
   Dynamo worker — the same computation `open-jev`'s own `OptionScorer._score_with_prefix`
   does, over HTTP instead of an in-process batched forward pass — and `run_score.py` ports
   `open-jev`'s own `render_choice`/`render_score`/`render_noul` and `system_one()` (softmax,
   `score` as a probability-weighted mean, entropy-based `confidence`) verbatim from
   `open-jev/openjev/systemone.py`. This is not an approximation; it is the same math, scoring
   real log-probabilities from roundhouse's local tier instead of a parsed guess.

`run.py`, the original prompted-classification approach (ask the model to answer with the
label on its own line, then parse it), is kept alongside `run_score.py` as a documented
comparison point — see `INTEGRATION.md` Gap 1's "Option C" — not because it's still the only
option, but because the difference between the two is itself informative: `run_score.py`'s
calibrated answers are noticeably better differentiated (see "Real run" below).

## Why Qwen, not Gemma

`open-jev` runs Gemma 3 4B locally (MLX or PyTorch). This use case runs
`Qwen/Qwen2.5-Coder-32B-Instruct` instead — the same model `cache-aware-routing` already
validated serving on Dynamo on a single GPU — and deliberately does **not** stand up a second
Dynamo worker for Gemma. Three reasons, weighed against each other:

1. **Zero extra install or GPU memory.** Downloading Gemma 3 4B and launching a second
   `dynamo.vllm` process is a real, if modest, cost (a new model pull, a new serve process, a
   port to manage) — and on the single-GPU box this was built on, Qwen 32B's steady-state
   footprint (~77/81.6 GB, see `cache-aware-routing/GAPS.md`) leaves nowhere near enough
   headroom to also load Gemma alongside it. Running both would mean tearing one down first.
2. **The task doesn't need Gemma specifically.** This is bounded classification over a short
   list of labels — department/severity/yes-no — not code generation, and not a domain where
   Qwen2.5-Coder's coding specialization is a mismatch severe enough to matter. Any competent
   instruction-tuned model can follow "answer with the option key on its own line."
3. **`open-jev`'s own README already shows Gemma zero-shot is *not* well-calibrated** on this
   exact task shape (the `is_urgent.noul` comparison against TypeSafe's published Jev numbers:
   0.999 vs Gemma's 0.005 — "Gemma is over-confident and disagrees on the two judgement calls").
   Model choice was never the source of fidelity to real Jev in the first place; a trained head
   on frozen features is, per `open-jev`'s own README, and that's out of scope for a routing demo.

**If model fidelity to `open-jev`'s actual choice ever matters** (e.g. reproducing its own
README numbers exactly), swapping in Gemma is one command, not a redesign — see
`serve_model.sh`'s `MODEL=` override, the same mechanism `cache-aware-routing` uses for its
32B→14B fallback. `google/gemma-3-4b-it`'s ~8 GB of bf16 weights would in fact leave far more
KV-cache headroom than Qwen 32B's tight 1.25× concurrency (see that use case's `GAPS.md`), so
it is not a suitability question, only an install-cost one — which is exactly why this use
case defaults to reusing what's already running instead.

## Deployment topology (Shape B, local-only)

Unlike `cache-aware-routing` (which supports both frontier-only and mixed local+frontier
shapes), this use case is **local-only by design** — `control-plane.json`'s policy admits
`local/*` and nothing else, because routing a "fast, cheap, bounded" decision to a frontier
model would defeat the point of the workload it's demonstrating.

```
[This machine, or an SSH-tunneled laptop]
  client (run.py / run_score.py ) ──▶ roundhouse :8080
                           │  in-process EmbeddedFleet, real HttpLocalExecutor
                           │  ZMQ subscribe (KV events)
                      Dynamo worker (Qwen2.5-Coder-32B, reused from cache-aware-routing)
```

A frontier catalog entry still exists in `catalog.json` (roundhouse refuses to boot with an
empty catalog — `CatalogError::Empty`) but is never dispatched to under this policy; see that
file's `$comment`.

## Quickstart: spin up Dynamo + Qwen and run `run_score.py`, start to finish

Everything below in one place, in order, for someone who just wants to run it — the sections
further down (`Reusing an existing worker`, `The ROUNDHOUSE_LOCAL_ENABLE_SCORE flag`, `Which
script do I run?`) explain the *why* behind each step if you need it; this section is only the
*what*. If a step here fails, its explanation is one of those sections away.

```bash
# --- 0. Is Dynamo already running? Check before assuming either way. ---
curl -s -o /dev/null -w "%{http_code}\n" --max-time 3 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen2.5-Coder-32B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
# 200 -> skip to step 2. Anything else (000, connection refused) -> step 1.
```

```bash
# --- 1. Spin up Dynamo (skip if step 0 printed 200) ---
# Needs the model weights pulled and Dynamo's Python env built already -- if this is a
# fresh box that's never had either, do ENVIRONMENT_SETUP.md §6-§9 first, then come back.

# 1a. etcd + nats
cd ~/dynamo/dev && docker compose up -d && cd -

# 1b. Frontend -- run in its own terminal, leave it running
source ~/dynamo/.venv/bin/activate
python -m dynamo.frontend --router-mode kv --http-port 8000

# 1c. Worker -- a SECOND terminal, leave it running too. This is the slow step: ~1-3 min
# even with weights already cached locally. Wait for "chat endpoints enabled" in its log.
source ~/dynamo/.venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0   # CUDA-13 workaround; harmless on CUDA 12
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
  --model Qwen/Qwen2.5-Coder-32B-Instruct --block-size 64 --tensor-parallel-size 1 \
  --enable-prefix-caching --enforce-eager \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'

# 1d. Re-run step 0's curl -- must print 200 before continuing.
```

```bash
# --- 2. Mint roundhouse keys (one-time) ---
cd <roundhouse-repo-root>
python3 use-cases/openjev-demo/mint_keys.py
```
**Do this *before* step 3, not after, and don't re-run it once roundhouse (step 3) is up.**
`mint_keys.py` overwrites `control-plane.json` with a freshly generated key every time it
runs — but roundhouse only reads that file once, at boot; it does not hot-reload it. Run this
again while roundhouse is already running (e.g. re-following this Quickstart from the top
without noticing step 3 never stopped) and every request afterward gets `HTTP 401
{"code":"unknown_key"}` — a real, seen-in-practice failure, not hypothetical: roundhouse is up,
`keys.local.json` and `control-plane.json` agree with each other on disk, but neither matches
the key hash the *running process* loaded minutes or hours earlier. Fix: restart roundhouse
(step 3 again) so it re-reads the file mint_keys.py just wrote.

```bash
# --- 3. Launch roundhouse -- a THIRD terminal, leave it running ---
TOKENIZER=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct \
  -name tokenizer.json | head -1)   # or wherever your HF_HOME cache put it
INFERENCE_API_KEY=unused-for-local-only-demo \
ROUNDHOUSE_CATALOG=use-cases/openjev-demo/catalog.json \
ROUNDHOUSE_CONTROL_PLANE=use-cases/openjev-demo/control-plane.json \
ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000 \
ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
ROUNDHOUSE_LOCAL_TOKENIZER="$TOKENIZER" \
ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080 \
ROUNDHOUSE_LOCAL_ENABLE_SCORE=1 \
cargo run --release -p roundhouse-server --bin roundhouse
# Wait for "roundhouse listening" and "local fleet: real log-probability scoring enabled
# at /v1/local/score" in its log -- the second line confirms ROUNDHOUSE_LOCAL_ENABLE_SCORE
# was actually picked up, not just that roundhouse started.
```

```bash
# --- 4. Run it, in a FOURTH terminal ---
cd <roundhouse-repo-root>
python3 use-cases/openjev-demo/run_score.py
```

**What success looks like:** 5 tickets, each with 3 answers carrying a real `probability`/
`confidence` (not `UNPARSED(...)`, not an error) — `department` should resolve crisply
(confidence at or near 1.000) on every ticket. See "Real run, `run_score.py`" further down for
the actual output this produced, to compare against.

**Two failure modes, two different fixes — read the error message to tell them apart:**

- **`score_failed`/connection error mentioning `127.0.0.1:8000`** — roundhouse correctly
  reporting it can't reach Dynamo. Go back to step 0. Dynamo (step 1) is a separate set of
  processes from roundhouse (step 3); either can die independently of the other (a closed
  terminal, a box reboot, a previous session ending), and roundhouse gives no indication of
  Dynamo's state until a request actually needs it.
- **`HTTP 401 {"code":"unknown_key"}`** — Dynamo is fine; roundhouse's *in-memory* key doesn't
  match what's on disk right now, almost always because step 2 (`mint_keys.py`) got re-run
  after step 3 (roundhouse) was already up. Fix: re-run step 3 to restart roundhouse — it will
  pick up whatever `control-plane.json` currently says. See step 2's own note above for why
  this happens.

## Files

| File | What it is |
|---|---|
| `tickets.jsonl` | 5 support tickets, each with `state` + 3 typed questions (`choice`/`score`/`noul`), modeled on `open-jev`'s own `examples/systemone-quickstart.json`. |
| `catalog.json` | One frontier entry (required, unused under this policy) + the local Qwen entry, reusing `cache-aware-routing`'s validated config. |
| `control-plane.json` | One project, `policy.allow: ["local/*"]` — local-only, deliberately. |
| `mint_keys.py` | Mints `rh_turn_`/`rh_admin_` secrets — same mechanism as `cache-aware-routing`. |
| `run.py` | Prompted-classification driver: per ticket, per question, one roundhouse turn; parses the free-text answer, reports cache% growth across a ticket's questions, confirms local routing via `/v1/metrics`. Kept as the `INTEGRATION.md` "Option C" comparison point. |
| `run_score.py` | **The real thing** — calls `POST /v1/local/score` for real, calibrated log-probability answers. `open-jev`'s own prompt renderers and `system_one()` math, ported verbatim. |
| `bench_common.py` | Shared streaming/timing/KV-metrics instrumentation for the two benchmark scripts below. |
| `run_with_openjev.py` | Benchmark: the real logprob-scoring path, instrumented per LLM call (KV-cache %, cold start, prefill/decode split, tokens). Talks directly to Dynamo, not through roundhouse — see "Tokenomics" below. |
| `run_without_jev.py` | Benchmark: one structured-JSON-generation call per ticket (no logprob flag) as the "without Jev" baseline, same instrumentation. |
| `results/*.json` | Raw per-call and per-ticket metrics from the last benchmark run of each script, plus `summary.json`'s consolidated comparison. |
| `pull_model.sh` / `serve_model.sh` | Only needed if no Dynamo worker is already running — see "Reusing an existing worker" below. |

## Reusing an existing worker

This use case points at `http://127.0.0.1:8000` (Dynamo's HTTP frontend) and
`tcp://127.0.0.1:20080` (its KV-events stream) — the same worker `cache-aware-routing` uses.
If that worker is already serving, this use case needs **no additional GPU setup at all** — the
"Quickstart" section above already includes the check (its step 0) and the spin-up commands
(its step 1) inline, so this section only adds the parts that aren't part of the normal path:

**Why the failure is confusing.** roundhouse itself can be running fine on `:8080` while Dynamo
on `:8000` is not (a process died, the box rebooted, a previous session ended) — and when that
happens, `run_score.py`/`run.py` fail with a *roundhouse* error that's easy to misread as a
roundhouse bug:
```
/v1/local/score -> HTTP 400: {"error":{"code":"score_failed","message":"http://127.0.0.1:8000/v1/completions:
  request failed: error sending request for url (http://127.0.0.1:8000/v1/completions)"}}
```
That message is roundhouse correctly reporting that *it* can't reach Dynamo — not that
roundhouse itself is broken. See the Quickstart's step 0 for the one-line check, and re-run it
any time this error shows up, since either process (roundhouse or Dynamo) can die independently
of the other with no warning from the one still running.

**Restarting only the frontend** (e.g. to pick up a new roundhouse env var without reloading
weights) — use `cache-aware-routing/shutdown_served_model.sh --frontend`, not Ctrl+C on
`serve_model.sh`'s own terminal, and not by closing the Quickstart's step-1b terminal either.
See that script's own header for why the difference matters: `serve_model.sh` (and a manually
backgrounded frontend+worker pair in one shell) ties both to one shell's exit, so killing either
from inside that shell takes both down together and costs several minutes reloading 32B of
weights for a change that only touched the frontend.

**First time on this box, nothing installed yet?** Follow `ENVIRONMENT_SETUP.md` end to end
(clones Dynamo, builds its Python bindings, downloads the model weights) or
`cache-aware-routing/DYNAMO_LOCAL_SERVING.md` for the same steps with more evidence/rationale
attached — this use case does not duplicate either, then come back here.

## The `ROUNDHOUSE_LOCAL_ENABLE_SCORE` flag

`/v1/local/score` is gated by its own opt-in variable, separate from the four
`ROUNDHOUSE_LOCAL_*` variables that wire the local fleet itself, and **off by default**.
Two use cases, two different needs, made this the right boundary rather than one flag:

- `cache-aware-routing` wires a local fleet purely for ordinary turn dispatch
  (`HttpLocalExecutor`, one request per turn) and has no use for per-option
  log-probability scoring — mounting the route anyway would be surface nobody there asked
  for, with its own extra per-option HTTP round trips (`local_score.rs`'s `score()` makes
  one completions call *per option*, not one batched pass — see that function's own doc
  comment on the cost of doing this over a network boundary).
- `openjev-demo` is the one use case that actually calls `/v1/local/score`, so it is the one
  that sets `ROUNDHOUSE_LOCAL_ENABLE_SCORE=1`.

Unset, the route does not exist (404) — not "exists and refuses." Set (any non-empty value,
following the same presence/absence convention every other `ROUNDHOUSE_LOCAL_*` variable
uses), `main.rs` logs `"local fleet: real log-probability scoring enabled at
/v1/local/score"` at boot so the decision is visible, not just inferred from behavior.

## Run it

The commands are in "Quickstart" above (mint keys → launch roundhouse → run a script) — this
section is the two things worth knowing *about* those commands rather than a second copy of
them:

- **`ROUNDHOUSE_LOCAL_ENABLE_SCORE` is required for `run_score.py`.** It's a separate,
  off-by-default opt-in from the four `ROUNDHOUSE_LOCAL_*` variables (see
  `crates/roundhouse-server/src/local_score.rs`). Without it, `/v1/local/score` simply does not
  exist (404) — `run.py`, which doesn't call that route, still works unchanged either way. See
  "The `ROUNDHOUSE_LOCAL_ENABLE_SCORE` flag" below for why this is its own switch rather than
  following automatically from the endpoint being set.
- **`ROUNDHOUSE_FRONTIER_UPSTREAM` is deliberately left unset.** This deployment never
  dispatches to the frontier, so the offline echo stub standing in for it is fine; see
  `main.rs`'s own log line at boot ("no frontier upstream configured ... serving the offline
  echo stub").

`run_score.py` and `run.py` can both be run against the same roundhouse process, one after the
other — `run_score.py` for real calibrated scoring, `run.py` as the prompted-classification
comparison point. See "Which script do I run?" below for the full script comparison, including
the two benchmark scripts this section doesn't cover.

## Which script do I run?

Four scripts, two different purposes — routing demos (go through roundhouse, prove the
integration) versus a tokenomics benchmark (bypass roundhouse, measure Dynamo/vLLM's serving
cost directly). Reach for the one that matches what you're actually trying to show:

| Script | Goes through roundhouse? | What it's for | Needs |
|---|---|---|---|
| `run.py` | ✅ `/v1/responses` | Prove roundhouse can route a Jev-shaped workload to the local worker at all, with a free-text answer parsed client-side. The `INTEGRATION.md` "Option C" comparison point — approximate answers, but the simplest path. | roundhouse running (Step 2, `ROUNDHOUSE_LOCAL_ENABLE_SCORE` not required) |
| `run_score.py` | ✅ `/v1/local/score` | The real integration demo: real, calibrated `probability`/`confidence` per answer, `open-jev`'s own math ported verbatim, dispatched through roundhouse. Use this to show "roundhouse can serve a genuine Jev-style decision layer," not just an approximation. | roundhouse running **with** `ROUNDHOUSE_LOCAL_ENABLE_SCORE=1` (Step 2 as written above) |
| `run_with_openjev.py` | ❌ direct to Dynamo | **Benchmark, not a demo.** Same real logprob-scoring mechanism as `run_score.py`, but instrumented per LLM call (KV-cache %, cold-start, prefill/decode split, token counts) and dumped to `results/with_openjev.json`. Use this (paired with `run_without_jev.py`) to measure the tokenomics claim, not to demonstrate roundhouse routing. | Dynamo serving on `:8000` directly — **no roundhouse process needed**; Dynamo venv active (`source ~/dynamo/.venv/bin/activate`) and `ROUNDHOUSE_LOCAL_TOKENIZER` set |
| `run_without_jev.py` | ❌ direct to Dynamo | The tokenomics benchmark's baseline half: one structured-JSON-generation call per ticket, no logprob flag, same instrumentation, dumped to `results/without_jev.json`. Meaningless on its own — always run alongside `run_with_openjev.py`. | Dynamo serving on `:8000` directly — no roundhouse, no tokenizer/venv needed (sends text, not token ids) |

**If you're demonstrating the integration** (roundhouse routes to a local Jev-style scorer):
`run_score.py`, optionally alongside `run.py` to show the free-text fallback it replaces.

**If you're measuring cost/latency** (does Jev's "ultra-low latency, massive cost reduction"
claim hold up): run the benchmark pair below, **not** `run_score.py`/`run.py` — see
"Tokenomics" further down for why this needs its own scripts and its own clean-restart
protocol rather than reusing the roundhouse-routed ones.

```bash
# Tokenomics benchmark — cold-restart Dynamo before EACH condition, or the second
# run's KV cache will be biased by whatever the first run already warmed.
source ~/dynamo/.venv/bin/activate
export ROUNDHOUSE_LOCAL_TOKENIZER=/path/to/tokenizer.json

use-cases/cache-aware-routing/shutdown_served_model.sh --worker
# ... relaunch the worker (see DYNAMO_LOCAL_SERVING.md) and wait for "chat endpoints enabled" ...
python3 use-cases/openjev-demo/run_with_openjev.py     # -> results/with_openjev.json

use-cases/cache-aware-routing/shutdown_served_model.sh --worker
# ... relaunch the worker again, wait for ready ...
python3 use-cases/openjev-demo/run_without_jev.py      # -> results/without_jev.json
```

## Expected output

**`run_score.py`** (the real thing): every answer carries a real `probability`/`confidence`
computed from the worker's own log-probabilities — `department` should resolve crisply
(confidence near 1.0) for every ticket here, `frustration` and `is_urgent` should show genuine
spread rather than always landing on the same value.

**`run.py`** (comparison point):
- Each ticket's **first** question is cold (`cache%` near 0 — nothing about this ticket has
  been seen yet).
- The **second and third** questions for the same ticket should show `cache%` climbing
  sharply, because they share the `state` block roundhouse already has in cache from the
  first question — the real, provider-measured KV-cache-reuse effect
  `cache-aware-routing` demonstrates, applied here across *questions* rather than *turns of
  a conversation*.
- `/v1/metrics` at the end should show every call attributed to `"provider": "dynamo"` — proof
  the local-only policy actually held and nothing fell over to frontier. (`run_score.py`'s own
  calls do **not** appear here — `/v1/local/score` bypasses the turn engine by design, see
  GAPS.md.)
- Answers are the model's own free-text classification, not calibrated probabilities — expect
  some tickets to parse as `UNPARSED(...)` if the model didn't follow the "answer with the
  label only" instruction exactly; that failure mode is itself informative (see GAPS.md).

### Real run, `run_score.py` (2026-09-28)

```
=== ticket: stripe-outage ===
state: 'Our Stripe connection has been failing for three days and we are losing sales every hour. We need this fixed urgently.'
    department: technical  confidence=1.000  (billing=0.000, technical=1.000, sales=0.000)
   frustration: score=1.992  confidence=0.959
     is_urgent: p(yes)=0.693

=== ticket: plan-question ===
    department: sales      confidence=1.000
   frustration: score=0.000  confidence=0.997
     is_urgent: p(yes)=0.000

=== ticket: wrong-invoice ===
    department: billing    confidence=1.000
   frustration: score=1.107  confidence=0.691
     is_urgent: p(yes)=0.000

=== ticket: api-error-500 ===
    department: technical  confidence=1.000
   frustration: score=1.893  confidence=0.690
     is_urgent: p(yes)=0.693

=== ticket: feature-request ===
    department: technical  confidence=1.000
   frustration: score=0.000  confidence=0.999
     is_urgent: p(yes)=0.000
```

All 5/5 `department` calls correct and crisp (confidence 1.000). `is_urgent` cleanly separates
the two genuinely urgent tickets (p≈0.69) from the three that aren't (p=0.000) — notably
**better calibrated** than `run.py`'s free-text version below, which answered `yes` on 4 of 5
tickets including the "no rush" feature request. `frustration` gives a real continuous score
(1.992 for the angriest ticket, 0.000 for the calmest) rather than `run.py`'s single integer
pick. Getting here took three ruled-out mechanisms before finding the one that works — see
`local_score.rs`'s own doc comment and `INTEGRATION.md` Gap 1's addendum for the full record.

### Real run, `run.py` (2026-09-28, kept for comparison)

Against the live Qwen2.5-Coder-32B worker, no fixes needed:

```
=== ticket: stripe-outage ===
    question    in_tok    cached   cache%  answer
  department       141         0     0.0%  technical
 frustration       200       128    64.0%  0
   is_urgent       374       192    51.3%  yes

=== ticket: wrong-invoice ===
    question    in_tok    cached   cache%  answer
  department       145         0     0.0%  billing
 frustration       204       128    62.7%  2
   is_urgent       361       192    53.2%  no
```
(5 tickets total; full table in `PLAN.md`'s Phase 0 record.) Every one of the 15 questions
parsed to a real label — zero `UNPARSED(...)`. `/v1/metrics`:
```json
{"provider": "dynamo", "model": "Qwen/Qwen2.5-Coder-32B-Instruct", "calls": 15, "mode": "local",
 "tokens": {"input": 3547, "cached_input": 1600, "uncached_input": 1947, "output": 1429}}
```
All 15 calls, zero to frontier — the local-only policy held for every question of every ticket.
`department` correctly differentiated `billing`/`technical`/`sales` across all five tickets
(including the ambiguous `plan-question` ticket, correctly called `sales`); `is_urgent`
disagreed with a human's likely call on the `feature-request` ticket (answered `yes`, arguably
should be `no` for a "no rush" request) — the same zero-shot-miscalibration failure mode
`open-jev`'s own README documents for Gemma on `is_urgent`, not a bug in this demo.

## Tokenomics: what the numbers actually say (measured, 2026-09-28)

TypeSafe's own positioning for Jev/System One claims **ultra-low latency** and **massive cost
reductions** (tokens and KV-cache reuse) versus an ordinary LLM call. This section measures
both claims against the same 5 tickets, on the same local Qwen worker, with a clean Dynamo
restart (cold KV cache) before each condition so neither run could free-ride on the other's
warm cache. **One claim held decisively; the other did not, and the reason why is itself the
interesting result.**

**Methodology.** `run_with_openjev.py` and `run_without_jev.py` talk directly to Dynamo
(bypassing roundhouse, like `cache-aware-routing`'s `run_local.py`), streaming every request so
prefill time (time-to-first-token) and decode time (everything after) can be measured
client-side — see `bench_common.py`'s module doc for exactly what this does and does not
capture precisely. `cached_tokens`/`prompt_tokens`/`completion_tokens` are real numbers from
vLLM's own `usage` object, not derived from timing.

### ✅ Ultra-low latency — confirmed, ~2.3× faster

| | with_openjev (8 calls/ticket) | without_jev (1 call/ticket) |
|---|---|---|
| Wall time per ticket | **311 ms** | **712 ms** |
| — of which decode | 0 ms | 608 ms |
| — of which prefill (TTFT) | 306 ms | 77 ms |

Jev's decode time is genuinely zero — every scoring call only ever generates the one throwaway
token needed to trigger `prompt_logprobs`, so there is no sequential decode loop at all. The
JSON-generation baseline spends essentially all its time in decode (27 tokens, ~22.5 ms/token,
inherently sequential). Even though Jev makes 8× more HTTP round trips per ticket (so its own
*prefill* time is actually higher in absolute terms — 306 ms vs. 77 ms, pure network/scheduling
overhead × 8), eliminating decode entirely still wins on total wall time by more than 2×. This
is the real mechanism behind "ultra-low latency": prefill is one parallel pass over the whole
prompt; decode is inherently sequential, one token at a time. Removing decode removes the
slow part.

### ⚠️ "Massive cost reduction in tokens / KV-cache reuse" — not confirmed as measured

| | with_openjev | without_jev |
|---|---|---|
| Total tokens per ticket (prompt + completion) | **584.6** | **282.2** |
| New (uncached) tokens actually processed per ticket | **328.6** | **128.6** |
| KV-cache hit rate | 44.4% | 60.2% |

By raw token volume, **the Jev path used about 2.1× more tokens per ticket, and 2.6× more
newly-processed (uncached) tokens** — the opposite of the marketing claim. This is not a flaw
in the Jev *concept*; it's a real cost of this implementation's specific shape. Inspecting one
ticket's raw calls shows exactly why:

```
q=department   option=billing    prompt_tok=75  cached= 0  new_prefill=75  cold=True   <- 1st option, cold
q=department   option=technical  prompt_tok=75  cached=64  new_prefill=11  cold=False  <- reuses 1 full block
q=department   option=sales      prompt_tok=75  cached=64  new_prefill=11  cold=False  <- reuses 1 full block
q=is_urgent    option=yes        prompt_tok=45  cached= 0  new_prefill=45  cold=True   <- never crosses 1 block
q=is_urgent    option=no         prompt_tok=45  cached= 0  new_prefill=45  cold=True   <- never crosses 1 block
```

Two compounding costs, both structural, neither a bug:
1. **`/v1/local/score` makes one HTTP call per candidate option**, each resending nearly the
   full context — `local_score.rs`'s own doc comment already names this trade-off ("one request
   per option, not one batched pass"). `open-jev`'s own local `OptionScorer` avoids this
   entirely: it prefills the context **once** and expands that one KV cache across every
   option in a single batched forward pass — no redundant resend, ever, regardless of block
   alignment. This benchmark's implementation is the HTTP-callable version of that idea, not
   the same one, and this is the cost of that difference.
2. **Block-size granularity wastes the tail.** With `block_size: 64`, only *complete* 64-token
   blocks are ever cached — a 45-token `is_urgent` context never reaches one full block, so
   neither option is ever cached, ever, on any ticket. A 75-token `department` context crosses
   one block boundary, so only the *first* option pays full cold-prefill cost; the other two
   reuse the cached block and pay only for their own trailing ~11 tokens — real reuse, but only
   because that particular context happened to be long enough.

**What this means, honestly:** the latency claim is real and was measured directly. The
token/cache-cost claim, as marketed, would need a genuinely batched local-tier scoring
implementation — sharing one prefill across all of a question's options in a single call,
the way `open-jev`'s own scorer does — to hold up; `/v1/local/score`'s current one-call-per-option
design does not deliver it, and reports that honestly rather than only reporting the number
that matches the expected story. See `GAPS.md` for this as a tracked gap and `INTEGRATION.md`
for what a batched variant would take.

### How to read the two tables together — three numbers that look "worse" for Jev, and why

It's natural to look at 306 ms of TTFT, 44.4% cache hit, and 2.1× the token count and conclude
Jev came out behind overall. That reading conflates two different things this benchmark
deliberately measured separately: **what is intrinsically true about Jev's approach** (scoring
instead of generating), and **what is an artifact of how `/v1/local/score` happens to be
implemented today** (one HTTP call per option). Only the first is a property of the *concept*;
the second is a property of *this route*, and a batched implementation would remove it without
touching the first at all.

- **TTFT (306 ms vs. 77 ms) is a sum of 8 calls, not a slower prefill.** Jev issues 8 separate
  HTTP requests per ticket, and each one pays its own fixed network/scheduling overhead before
  Dynamo even starts computing — that overhead is what gets summed into 306 ms.  Divide it back
  out (306 ms / 8 ≈ 38 ms per call) and each individual Jev prefill is actually *faster* than the
  JSON baseline's single 77 ms call, not slower. The 306 ms total is a **call-count** tax, not a
  **prefill-speed** tax — and it is exactly the cost `INTEGRATION.md`'s Option D (batching all of
  a question's options into one call) would remove, since it would collapse 8 round trips into 1.
- **The 44.4% cache-hit rate and the 2.1–2.6× token counts are the one part of the marketing
  claim this implementation genuinely does not deliver, and the raw calls above show exactly
  why**: resending the full context once per option, and a short context (`is_urgent`'s 45
  tokens) never crossing even one cacheable 64-token block. Neither of those is a property of
  "scoring options via log-probabilities" as an idea — `open-jev`'s own scorer never resends
  anything, because it prefills once and reuses that KV state across every option in-process.
  They are properties of shipping that idea as N independent, stateless HTTP calls instead of
  one batched one.
- **The one number that *is* a direct, unavoidable consequence of the Jev concept itself is
  decode time — 0 ms vs. 608 ms.** No implementation change makes a text-generation baseline's
  decode loop disappear, and no implementation change is needed to make Jev's decode loop not
  exist in the first place: it was never going to generate more than the one throwaway token
  `prompt_logprobs` requires. That is the real, structural, "ultra-low latency" win, and it is
  the only one of the three headline numbers that would survive a smarter implementation
  unchanged rather than being *caused by* the current implementation's shortcuts.

Put differently: **latency wins because of what Jev fundamentally is; the token/cache numbers
lose because of how this specific endpoint is built, not because of what Jev is.** Building
Option D would be expected to close most of the TTFT gap (fewer round trips) and flip the
token/cache numbers in Jev's favor (no redundant resend) — without changing the decode-time
result at all, since that one was never in question.

Raw per-call data: `results/with_openjev.json`, `results/without_jev.json`. Consolidated
numbers: `results/summary.json`.

## Notes

- `keys.local.json` holds real secrets and is gitignored.
- Rate cards in `catalog.json` are placeholders and irrelevant to this use case's actual
  point (local routing works with $0 frontier spend, by design — the dashboard's savings
  figures don't mean much here the way they do in `cache-aware-routing`).
- See `SCORECARD.md` for the fitness score, `GAPS.md` for gaps, `PLAN.md` for phases,
  `INTEGRATION.md` for the repo-level implications if the `/v1/local/score` pattern gets
  adopted more broadly (Gap 1, closed 2026-09-28).
