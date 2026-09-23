<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Progress Tracker — cache-aware-routing

Tracks this use case against the three ways a turn can actually be served, each proven or not
proven on its own evidence rather than assumed from the others:

- **(1) Frontier-only** — every turn dispatched to NVIDIA's `inference-api.nvidia.com`, the
  hosted frontier model, through roundhouse.
- **(2) Local-only** — every turn dispatched to the Dynamo-served `Qwen/Qwen2.5-Coder-32B-Instruct`
  worker, through roundhouse.
- **(3) Mixed** — `AffinityPolicy` choosing per turn between (1) and (2) on the same deployment,
  the actual co-optimization story this use case exists to demonstrate.

---

## Overview

Checked in two separate rounds — (1) was already run and verified in an earlier round of this
session; this round's actual new work was (2), end to end, for the first time.

| Phase | What it means | Checked | When | Primary evidence |
|---|---|---|---|---|
| **(1) Frontier-only** | 100% of turns → NVIDIA frontier API, cache% is the provider's own reported number | ✅ **Verified** | Earlier this session | `run.py` against the real `inference-api.nvidia.com` (Anthropic Messages dialect) |
| **(2) Local-only** | 100% of turns → the local Dynamo/Qwen worker, cache% from real KV events | ✅ **Verified end to end this round** | This round (2026-09-23) | `run_local.py` (Dynamo direct, no roundhouse) run earlier; **this round adds** a real `/v1/responses` session routed and measured *through roundhouse itself* — the new `HttpLocalExecutor`/`EmbeddedFleet` wiring |
| **(3) Mixed** | Same deployment, same policy, `AffinityPolicy` picks frontier or local per turn | ⚠️ **Not yet run** | — | Both paths individually proven; no run has yet put both candidates in one policy and watched the router choose |

---

## (1) Frontier-only — verified

**What exists:** `catalog.json` names one frontier model (`aws/anthropic/bedrock-claude-opus-4-8`
via NVIDIA's `anthropic_messages` dialect); `run.py` drives it through roundhouse's
`/v1/responses`.

**What was fixed this round (2026-09-23):** `catalog.json` was actually broken on current `main`
— `"provider": "nvidia"` named no `"providers"` entry, which the M10.1 provider registry refuses
at boot. Added the `providers.nvidia` block and switched to the Anthropic-Messages dialect
matching the NVIDIA endpoint supplied for this session.

**Evidence:** `cached_tokens` is the real number NVIDIA's API reports back over HTTP (traced to
`crates/roundhouse-fleet/src/anthropic_messages.rs`'s `cache_control` breakpoint and
`stream.rs`'s parsing of `cache_read_input_tokens`), not an estimate. Nothing new was needed here
this round beyond the schema fix — this path was already real.

---

## (2) Local-only — verified, and newly closes the routing gap

**What existed before this round:** Dynamo itself installed from source and serving
`Qwen/Qwen2.5-Coder-32B-Instruct` on a single H100 80GB (`DYNAMO_LOCAL_SERVING.md`), and
`run_local.py` proving real Tax A/B *directly against Dynamo* — but roundhouse had no way to
route a turn there at all (`LocalExecutor` was `EchoLocalExecutor`, a canned string; `serve()`
wired no `EmbeddedFleet`).

**What was implemented this round:**

| Piece | File | What it does |
|---|---|---|
| `HttpLocalExecutor` | `crates/roundhouse-server/src/local_fleet.rs` (new) | Real `LocalExecutor` impl. `POST {endpoint}/v1/completions` with `"prompt": [<raw token ids>]` — not `/v1/chat/completions` with text, because the ids must be the exact ones the KV-block hashes were computed over |
| `RuntimeTokenizer` | same file | Enum over `ByteTokenizer` / `HfTokenizer`, chosen at boot, so `serve()` stays written once instead of being duplicated per tokenizer choice |
| `local_fleet::from_env` | same file | Reads four `ROUNDHOUSE_LOCAL_*` vars, builds a real `EmbeddedFleet` (`use_kv_events: true`), registers the worker, returns everything `serve()` needs — `None` (today's unchanged behavior) if unset |
| Wiring | `crates/roundhouse-server/src/main.rs` | `serve()` now takes `local_setup: Option<LocalFleetSetup>`; swaps `EchoLocalExecutor`→`HttpLocalExecutor`, `ByteTokenizer`→`HfTokenizer`, calls `.with_fleet(...)`, and extends `reachable_candidates()` with a real local `Candidate` when configured |
| Dependency | `crates/roundhouse-server/Cargo.toml` | Added `reqwest` (workspace-pinned, rustls-only — no OpenSSL added to the shipped binary) |

**Config (all four required together, none required if unset):**
```bash
ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000
ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
ROUNDHOUSE_LOCAL_TOKENIZER=/path/to/tokenizer.json
ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080
```

**Verification performed:**
1. `cargo check --workspace` — clean.
2. `cargo test -p roundhouse-server` (lib + bin + every integration test file) — **369 lib tests,
   21 `main.rs` unit tests, and every integration suite including `mocker_cache_hits.rs` pass, 0
   failures.**
3. **Real end-to-end run** against the live Dynamo worker: launched `roundhouse` with the four
   `ROUNDHOUSE_LOCAL_*` vars set and `control-plane.json`'s policy narrowed to `["local/*"]"`,
   sent a real turn through `/v1/responses`, confirmed via `/v1/metrics`:
   ```json
   "models": [{"provider": "dynamo", "model": "Qwen/Qwen2.5-Coder-32B-Instruct", "calls": 1}]
   ```
4. **Real cache-hit proof through roundhouse's own routing** (not `run_local.py`'s direct-to-Dynamo
   bypass) — a 4-turn session over `corpus.md`:
   ```
   turn 1: in=1461 cached=0    cache%=0.0    <- cold
   turn 2: in=1485 cached=1408 cache%=94.8
   turn 3: in=1510 cached=1472 cache%=97.5
   turn 4: in=1582 cached=1472 cache%=93.0
   ```
   This exercises the real path end to end: roundhouse's tokenizer → block hashes → `EmbeddedFleet`
   quote → `HttpLocalExecutor` dispatch → Dynamo's real KV events → the next turn's quote reading
   them back. Nothing here is `run_local.py`'s vLLM-reported number; it is roundhouse's own
   `cached_input_tokens` on the `/v1/responses` wire, sourced from the local `LocalQuote`.

**What is still a placeholder, not a routing blocker:** `catalog.json`'s
`local_quality: {"Qwen/Qwen2.5-Coder-32B-Instruct": 0.72}` is hand-written (Option B from
`INTEGRATION.md` Gap 4), and `correlaries` is still empty, so the *savings dashboard* reports $0
even though routing itself works. See (3) below and `PLAN.md` Phase 3.

### Reproducible steps — (2), end to end through roundhouse

Prerequisite: Dynamo already serving per `DYNAMO_LOCAL_SERVING.md` (etcd/nats up,
`serve_model.sh serve` running, confirmed with its own `curl localhost:8000/v1/chat/completions`
smoke test).

```bash
# 1. Build the binary (from the repo root)
cargo build -p roundhouse-server --bin roundhouse

# 2. Point control-plane.json's policy at local only, to make the proof unambiguous
#    (the default "allow": ["*"] also works once (3) is being demonstrated instead)
python3 - <<'EOF'
import json
p = "use-cases/cache-aware-routing/control-plane.json"
cp = json.load(open(p))
cp["projects"][0]["policy"]["allow"] = ["local/*"]
json.dump(cp, open(p, "w"), indent=2)
EOF
python3 use-cases/cache-aware-routing/mint_keys.py

# 3. Launch roundhouse with the local fleet configured
TOKENIZER=$(find /ephemeral/cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct \
  -name tokenizer.json | head -1)
INFERENCE_API_KEY=unused-for-local-only-run \
ROUNDHOUSE_CATALOG=use-cases/cache-aware-routing/catalog.json \
ROUNDHOUSE_CONTROL_PLANE=use-cases/cache-aware-routing/control-plane.json \
ROUNDHOUSE_FRONTIER_UPSTREAM=openai_responses \
ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000 \
ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
ROUNDHOUSE_LOCAL_TOKENIZER="$TOKENIZER" \
ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080 \
./target/debug/roundhouse

# 4. In another terminal, replay a growing conversation over corpus.md through
#    roundhouse's /v1/responses (the same shape run.py/run_local.py use), reading
#    ROUNDHOUSE_API_KEY from keys.local.json, and watch cached_tokens climb.

# 5. Confirm which provider actually served it
curl -s http://127.0.0.1:8080/v1/metrics -H "x-roundhouse-key: $ADMIN_KEY" | python3 -m json.tool
#   -> "models": [{"provider": "dynamo", "model": "Qwen/Qwen2.5-Coder-32B-Instruct", ...}]
```

**Result, run 2026-09-23** (4-turn session over `corpus.md`, roundhouse's own reported
`cached_tokens` from `/v1/responses`):

```
turn 1: in=1461 cached=0    cache%=0.0    <- cold, no worker had this prefix yet
turn 2: in=1485 cached=1408 cache%=94.8
turn 3: in=1510 cached=1472 cache%=97.5
turn 4: in=1582 cached=1472 cache%=93.0
```

### Raw evidence — the Dynamo/vLLM worker's own log, KV-cache reuse

The lines below are vLLM's periodic scheduler log (`/tmp/serve_32b.log` on the box this ran on,
unedited except for selection), `Prefix cache hit rate` being vLLM's own real-time accounting of
its prefix cache — **cumulative since the worker started**, not per-turn, so it reflects
whatever traffic hit the worker at that point (including earlier `run_local.py` sessions), not
only the roundhouse-routed session above.

First contact with a growing prefix (from the earlier direct-to-Dynamo `run_local.py` proof,
0% → 95.5% as the cumulative hit rate climbs):

```
INFO ... Engine 000: Avg prompt throughput: 149.0 tokens/s, ... GPU KV cache usage: 3.8%, Prefix cache hit rate: 0.0%
INFO ... Engine 000: Avg prompt throughput: 49.9 tokens/s,  ... GPU KV cache usage: 5.2%, Prefix cache hit rate: 87.0%
INFO ... Engine 000: Avg prompt throughput: 60.4 tokens/s,  ... GPU KV cache usage: 6.6%, Prefix cache hit rate: 92.9%
INFO ... Engine 000: Avg prompt throughput: 38.2 tokens/s,  ... GPU KV cache usage: 4.6%, Prefix cache hit rate: 94.2%
INFO ... Engine 000: Avg prompt throughput: 34.8 tokens/s,  ... GPU KV cache usage: 6.0%, Prefix cache hit rate: 95.2%
INFO ... Engine 000: Avg prompt throughput: 45.5 tokens/s,  ... GPU KV cache usage: 0.0%, Prefix cache hit rate: 95.5%
```

The window covering this round's roundhouse-routed session (the `168.6 tokens/s` prompt-throughput
spike is the 4-turn corpus session above landing on the worker; the hit rate is already warm from
everything before it, and stays high rather than resetting):

```
INFO ... Engine 000: Avg prompt throughput: 2.2 tokens/s,   ... Running: 0 reqs, Prefix cache hit rate: 95.4%
INFO ... Engine 000: Avg prompt throughput: 0.0 tokens/s,   ... Running: 0 reqs, Prefix cache hit rate: 95.4%
INFO ... Engine 000: Avg prompt throughput: 4.3 tokens/s,   ... Running: 0 reqs, Prefix cache hit rate: 95.4%
INFO ... Engine 000: Avg prompt throughput: 0.0 tokens/s,   ... Running: 0 reqs, Prefix cache hit rate: 95.4%
INFO ... Engine 000: Avg prompt throughput: 168.6 tokens/s, ... Running: 0 reqs, Prefix cache hit rate: 93.8%
INFO ... Engine 000: Avg prompt throughput: 0.0 tokens/s,   ... Running: 0 reqs, Prefix cache hit rate: 93.8%
```

The dip from 95.4% to 93.8% rather than a rise is the honest artifact worth naming: this round's
first turn (turn 1 above, `cached=0`) was a genuinely cold prefix for the KV-event indexer even
though vLLM's own cache had seen a similar-shaped corpus before — `corpus.md`'s exact byte content
is identical run to run, but the ad hoc `control-plane.json` policy patch and process restart in
step 2-3 above meant this was a fresh `EmbeddedFleet`/indexer instance with no prior blocks
indexed, so turn 1 paid the same real cold-start cost `run_local.py`'s own turn 1 paid. Turns 2-4's
94.8%/97.5%/93.0% (the `roundhouse`-reported numbers, not vLLM's cumulative one) are the number
that actually demonstrates the mechanism working *through roundhouse's own quote/dispatch path*,
independent of vLLM's own rolling stat.

---

## (3) Mixed — not yet demonstrated

**What is true today:** both candidates are individually real and individually proven. Nothing
in the implementation prevents both from being in the same `reachable` list and the same
`AffinityPolicy` decision — `main.rs` already appends the local candidate to `reachable` whenever
`local_setup` is configured, alongside whatever the catalog's frontier entries produce.

**What has not been run:** a session where `control-plane.json`'s policy admits both
`"local/*"` and the frontier provider (i.e. the default `["*"]`, not the `["local/*"]"` this
round's local-only proof used to force the answer), watching `AffinityPolicy` actually alternate
or pick per cache state, and reading `explain_last_route` / `DecisionRecord` to confirm *why*.

**What would make it a real demonstration, not just a reachable configuration:**
1. Run with the default `policy.allow: ["*"]"` (already the case in `control-plane.json` — no
   change needed) and the four `ROUNDHOUSE_LOCAL_*` vars set.
2. Drive a session and inspect each turn's `DecisionRecord` (or add an `explain_last_route` MCP
   call per PLAN.md Phase 1) to show which target won and why — quality floor, cache warmth, cost.
3. Source real `correlaries` and a defensible `quality_prior` (PLAN.md Phase 3) so the savings
   dashboard's dollar figures mean something once a session actually mixes.

This is the next concrete step, not a Rust gap — the wiring for it already exists after this
round's work.

---

## Files touched this round (2026-09-23)

**Rust:**
- `crates/roundhouse-server/src/local_fleet.rs` (new)
- `crates/roundhouse-server/src/main.rs` (wiring)
- `crates/roundhouse-server/src/lib.rs` (module registration)
- `crates/roundhouse-server/Cargo.toml` (`reqwest` dependency)

**Use case docs and config:**
- `catalog.json` — provider-registry schema fix, `anthropic_messages` dialect, `local_quality`
- `README.md`, `GAPS.md`, `PLAN.md`, `INTEGRATION.md` — dated addenda
- `DYNAMO_LOCAL_SERVING.md` (new) — Dynamo install/serve reproduction
- `run_local.py` (new) — local-only, roundhouse-bypassing cache% proof
- `serve_model.sh` — single-GPU defaults, `dev/docker-compose.yml` path fix, FlashInfer workaround
- `PROGRESS_TRACKER.md` (this file, new)
