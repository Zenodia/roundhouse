<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cross-model KV cache transfer: baseline and plan

Work-branch material (`agent-docs/` never merges to `main`). The lasting
parts go into `docs/src/guides/cache-reuse.md` and
`docs/src/appendix/limitations.md` before a PR.

Paper: *Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear
Mapping for Prefill Reuse*, arXiv 2608.03893 (Heo et al., 2026-08-04).

## What the baseline commit does

`main` at `4a0ea46` ships no local fleet: `EchoLocalExecutor` is the only
`LocalExecutor`, and `docs/src/appendix/limitations.md` says so. This branch
ports the real local tier from `zcharpy/wiring-local-llm` onto that `main`
and generalizes it from one worker to N, which is the shape the paper's
experiment needs: a source model and a target model of one family, both
resident, both routable.

| Piece | File | Change |
|---|---|---|
| `local_fleet.rs` | `crates/roundhouse-server/src/local_fleet.rs` | Ported. `HttpLocalExecutor` no longer holds a model name; `from_env` reads N workers and registers each on one `EmbeddedFleet`. |
| `LocalExecutor::execute` | `crates/roundhouse-server/src/engine.rs` | Gains a `model: &str` argument. A Dynamo frontend serves every model behind one port, so endpoint alone cannot name the worker. Echo and the four test executors updated. |
| `EngineConfig::extra_local_models` | `engine.rs` | `Vec<LocalModelConfig { model, quality_prior }>`. Empty keeps the single-model path byte-identical. |
| Quoting loop in `plan` | `engine.rs` | One `FleetQuery` and one `LocalQuote` per local model; each candidate priced at its own prior. `connect_local` takes the quote whose `target()` the router chose. |
| `local_models`, `extra_local_quote_skip` | `engine/selection.rs` | The per-model policy check, so a `min_quality` sitting between the pair's priors admits one half and skips quoting the other. |
| Wiring | `main.rs`, `lib.rs`, `Cargo.toml` | `serve()` takes `Option<LocalFleetSetup>`; `RuntimeTokenizer` and the fleet's executor replace the echo pair; every worker is appended to `reachable` for the boot cross-checks. |

### Configuration

Primary worker (unchanged from `wiring-local-llm`):

```bash
ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000
ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
ROUNDHOUSE_LOCAL_TOKENIZER=/path/to/tokenizer.json
ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080
# optional: ROUNDHOUSE_LOCAL_BLOCK_SIZE=64 ROUNDHOUSE_LOCAL_ROUTING_GROUP=default
# local_base_ttft_ms comes from catalog.json, not an env var
```

Second and further workers, `i = 2, 3, ...` with no gaps:

```bash
ROUNDHOUSE_LOCAL_WORKER_2_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct
ROUNDHOUSE_LOCAL_WORKER_2_ENDPOINT=http://127.0.0.1:8000      # may equal the primary's
ROUNDHOUSE_LOCAL_WORKER_2_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20081
```

Tokenizer, block size and routing group are shared across the fleet by
construction: the engine hashes one buffer one way, and the selector matches
those hashes against every worker in the group. A cross-model pair needs
exactly that anyway.

Each worker's quality prior comes from `catalog.json`'s `local_quality` map
by model name, so the pair is priced at two different qualities from one
file. Fill it for both models or the router sees them as equals.

### Verify before committing

```bash
cargo fmt --all
cargo check --workspace
cargo clippy -p roundhouse-server --all-targets
cargo test -p roundhouse-server --lib local_fleet      # the worker-list parser
cargo test -p roundhouse-server                        # the rest, incl. local_quote_skip.rs
```

## Implementation

`experiments/cross-model-kv/` — its README is the runbook, in order:

- Part A, the paper: `collect_stats.py` (one pass over 500 FineWeb-Edu
  sequences accumulating full source/target second moments with RoPE
  stripped), `fit_mapper.py` (R² source selection, float64 ridge, λ=0.01),
  `eval_transfer.py` (target / transfer / source over identical
  continuations; retention and floor-normalized retention; the appendix-D
  perplexity protocol), `bench_latency.py` (mapper vs re-prefill).
- Part B, serving: `serving/serve_pair.sh` (two vLLM workers behind one
  Dynamo frontend, split by `--gpu-memory-utilization`), `env.sh` (worker 1
  and worker 2 variables), `run_pair.py` (forced handoff source→target
  through Roundhouse; the target's cold turn is the measured tax).
- `tests/test_kvmap.py`: CPU tests on toy shapes — RoPE round trip, moments
  vs direct ridge, R² ranking, `map_cache` end to end.

Default pair Qwen3-8B → Qwen3-14B (matched 8×128, fits one H100 with the
23 GB of statistics). The formulas in `stats.py`/`mapper.py` were checked
against a direct NumPy solve before commit; the torch code itself has not
run on a GPU yet — the first smoke run is `collect_stats.py --num-seqs 20`.

## What the paper needs that this branch does not yet do

1. **The mapper itself is engine-side.** Per-head ridge regression over the
   top-k most predictive source layers, with RoPE stripped from keys before
   the fit, trained on ~500 FineWeb-Edu sequences of 1,024 tokens. Roundhouse
   routes; it never sees a KV tensor. The fit and the injection live in a
   vLLM `KVConnectorBase` implementation or a sidecar beside the Dynamo
   worker.
2. **A transfer decision.** When the router moves a session from the source
   to the target model, it knows the exact prefix the source holds
   (`matched_prefix_tokens` on the source's quote). The decision "transfer
   the mapped KV vs. re-prefill on the target" belongs with the other
   per-turn corrections in `routing/learn/corrections.rs`, priced with the
   paper's measured mapper speed (2.7–25x over re-prefill) and retention.
3. **Family metadata in the catalog.** Which local models form a matched-KV
   pair (same KV head count and head dim) is configuration. `correlaries`
   in `catalog.json` is the natural hook; it is empty today.

## Model pairs

The mapper requires matched KV shape. Qwen2.5-Coder 14B and 32B both use 8
KV heads x 128 (the 7B uses 4 KV heads and does not match). Every dense
Qwen3 model uses 8 x 128, so any Qwen3 pair works.

## Compute for a small experiment

Phase 0, offline replication in HF transformers, needs no Dynamo. Calibration
is 512k tokens of forward-only passes; accumulate the per-head Gram matrices
on the fly rather than storing KV, and the solve is seconds. Load the two
models one at a time if memory is short.

| Pair | bf16 weights | Fits on | Wall time |
|---|---|---|---|
| Qwen3-1.7B -> 4B | ~12 GB | one 24 GB card | 1–2 h |
| Qwen3-4B -> 8B | ~24 GB | one A100-40 / L40S / H100 | 2–3 h |
| Coder-14B -> 32B | 92 GB | 2x H100, or 1x H100 sequentially | half a day |

Phase 1, both models resident in Dynamo workers behind Roundhouse:

- Qwen3-4B + 8B on one H100 with a split `--gpu-memory-utilization`: fine.
- Coder-14B bf16 + Coder-32B FP8: ~60 GB, tight on one H100, short contexts.
- Coder-14B + Coder-32B both bf16: 2x H100.

The serving scripts from the earlier branch are in
`agent-docs/cross-model-kv-transfer/`; `serve_model.sh` takes `MODEL=`,
`TP=` and `GPUS=` so the second worker is a second invocation with a
different model and KV-events port.

## Follow-on: decision memory (branch `zcharpy/decision-memory`)

`experiments/decision-memory/` reframes the use case around business
decisions that outrun the code (the Ledgerline payments fixture): ten
tasks, three conditions (no-memory / oracle / memory), policy tests that
read the current decision records, a capture step from meeting notes, and
the operate loop that retests affected tasks when a record is accepted.
Its README is the runbook. The KV work above is its cost layer: the
decision records sit in the shared prefix, and a changed record is the
prefix divergence point.
