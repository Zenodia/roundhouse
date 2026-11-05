<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cross-model KV cache transfer

A minimal implementation of *Cross-Model KV Cache Transfer in LLM Families:
A Closed-Form Linear Mapping for Prefill Reuse* (arXiv 2608.03893), sized
for one H100 80 GB, plus the Roundhouse serving demo that measures the
prefill tax it removes.

Two parts, run in order:

| Part | What it proves | Needs |
|---|---|---|
| **A. Mapper** (`collect_stats.py`, `fit_mapper.py`, `eval_transfer.py`, `bench_latency.py`) | The paper's result: a per-head ridge map from a source model's KV cache to a target model's, its retention on benchmarks, and its speed against re-prefill. | Python, both models in bf16 on one GPU. No Dynamo, no Roundhouse. |
| **B. Serving** (`serving/`) | Both models behind Dynamo, Roundhouse routing between them, and the cold prefill the target pays at a handoff — the number the mapper would take to zero. | Dynamo + vLLM, the Roundhouse binary. |

Part A is the paper. Part B is the baseline measurement on the serving side;
injecting the mapped cache into a vLLM worker is the next milestone, and the
last section says what that takes.

## Default pair

`Qwen/Qwen3-8B -> Qwen/Qwen3-14B`. Both are 8 KV heads x 128 (matched KV),
36 and 40 layers, one tokenizer; 16 + 28 GB of bf16 weights, so both fit on
one card with room for the statistics. The paper's own Qwen3 pairs target
32B, which with its source does not fit one 80 GB card; see "Other pairs".

Every script takes `--source` / `--target`, so any matched-KV pair works.
The scripts refuse a pair whose KV head count or head_dim differ.

## Part A: the mapper

### A0. Environment

Needs one NVIDIA GPU (80 GB for the default pair) and Python 3.12.

```bash
cd roundhouse/experiments/cross-model-kv
# uv works where python3-venv is not installed (stock Ubuntu without sudo):
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# Optional, for the paper's flash_attention_2 re-prefill timing: pip install flash-attn --no-build-isolation
huggingface-cli login   # only if your Hub access needs it; Qwen3 is public
python -m pytest tests -q   # CPU-only math tests, ~2 s
```

Pre-download the weights so later steps do not stall on the network:

```bash
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download Qwen/Qwen3-14B
```

### A1. Collect calibration statistics

```bash
python collect_stats.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
  --num-seqs 500 --seq-len 1024 --stride 4 \
  --out runs/qwen3-8b-14b/stats.pt
```

Streams 500 FineWeb-Edu documents (the paper's calibration set), runs both
models, strips the source's RoPE from its keys, keeps every 4th position
(~128K token observations), and accumulates float32 second moments over
*all* source layers against *all* target layers. One pass serves every k.

Expect: ~23 GB of statistics on the GPU beside 44 GB of weights, a few
minutes of forward passes, and a 23 GB `stats.pt`. If GPU memory is short,
`--stats-device cpu` moves the moments to host RAM (slower accumulation).
`--text-file some.txt` replaces FineWeb-Edu with local text for an offline
box; the paper's domain ablation puts that cost at under 1 pp for
Wikipedia-register text.

A smoke run first is worth it: `--num-seqs 20` finishes in under a minute
and exercises the whole path.

### A2. Fit the mapper

```bash
python fit_mapper.py --stats runs/qwen3-8b-14b/stats.pt --k 8 \
  --out runs/qwen3-8b-14b/mapper.pt
```

Per target layer: rank source layers by head-averaged R² (K and V averaged),
take the top k, solve `W = (XᵀX + λI)⁻¹ XᵀY` on the centered moments in
float64 with λ = 0.01, recover the bias. No model is loaded. For 8B→14B at
k = 8 the mapper is 671M parameters (2.7 GB float32); the paper's sizes are
in its appendix D.

To sweep k the way the paper selects it, pass several values and the k is
inserted into each file name (`0` means all source layers):

```bash
python fit_mapper.py --stats runs/qwen3-8b-14b/stats.pt --k 1 2 4 8 12 16 0 \
  --out runs/qwen3-8b-14b/mapper.pt      # writes mapper_k1.pt, mapper_k2.pt, ...
```

`fit_mapper` writes a `.json` beside each mapper with the selected source
layers and the best single-source R² per target layer. The paper reports
56% (K) / 32% (V) from one layer on 14B→32B; a flat or near-zero table here
means the statistics are wrong, not the pair.

### A3. Evaluate retention

```bash
python eval_transfer.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
  --mapper runs/qwen3-8b-14b/mapper.pt --tasks hellaswag,arc_challenge --limit 500
python eval_transfer.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
  --mapper runs/qwen3-8b-14b/mapper.pt --tasks wikitext --limit 60
```

Each benchmark is scored three ways over identical continuation tokens:
`target` (the target's own prefill: the ceiling), `transfer` (source
prefill, mapped), `source` (the small model alone: the floor you would get
by not switching). Reported per task: `acc_norm` per mode, `retention =
transfer / target`, and floor-normalized retention `(transfer − chance) /
(target − chance)`. WikiText follows the paper's prefix-conditioned protocol
(1,024-token prefix cache, score the next 1,024) and reports perplexity per
mode and the transfer/target ratio.

Results append to `runs/qwen3-8b-14b/results.jsonl`. With `--limit 500`
HellaSwag took 5.5 minutes on an H100 (3 modes x 4 choices x 500), ARC-Challenge
4.9 minutes, WikiText (60 chunks) under 30 seconds.
Drop `--limit` for the full validation split when you want paper-grade
numbers.

What to expect: the paper's Qwen3 pairs retain 95–98% on HellaSwag at
their selected k. 8B→14B is a new pair, not one from the paper; a Tier-1
result (>90% on HellaSwag, perplexity ratio under ~1.3) says the family
behaves as the paper's Qwen3 pairs did. A Tier-2 result (retention near the
source's own accuracy) is also an answer: the paper found two such pairs
and attributes them to attention-output geometry, not to the fit.

### A4. Latency

```bash
python bench_latency.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
  --mapper runs/qwen3-8b-14b/mapper.pt --seq-lens 64,256,1024,4096,8192
```

Times the mapper applied to an existing source cache against the target
re-running prefill over the same tokens (transformer body only, no LM head,
eager mode, bf16 — the paper's setting; add `--attn flash_attention_2` for
its re-prefill kernel). The source prefill is not in the mapper's time: the
source cache is the thing being reused. Writes `latency.json`. The paper
reports 4–25x small-to-large across 64 to 32K tokens with a fixed ~14 ms
floor at short lengths.

### A5. Pick k

Fit several k (A2), run A3 at each, and take the k with the best mean
`acc_norm` over the log-likelihood tasks, breaking ties toward larger k —
the paper's rule. That mapper is the one Part B's successor would ship.

## What it took to run this end to end

Part A needed three small changes to run, none to the method:

- **Environment.** `python3 -m venv` failed (no `python3-venv`, no sudo), so
  A0 now uses `uv venv`. `pip install -r requirements.txt` resolved to
  torch 2.15 and transformers 4.57.6, inside the `<5` pin.
- **One test fixed.** `test_moments_memory_estimate_is_the_float32_footprint`
  compared GiB against the README's decimal GB: the statistics are 21.4 GiB,
  which is the README's "about 23 GB". The code was right; the test now
  divides by `1e9`. The other six CPU tests, including the `DynamicCache`
  ones, passed unchanged, so the transformers-version risk did not bite.
- **`.venv/` added to `.gitignore`.**

Steps that needed nothing: FineWeb-Edu streams from the Hub without a login,
both models load together in bf16 (the statistics ran at 21.4 GiB beside
44 GB of weights), and the 20-sequence smoke run (`--num-seqs 20`, under a
minute) exercised the whole path before the full one.

Disk: `stats.pt` is 21.4 GiB, so a smoke run plus a full run needs ~45 GB of
`runs/`; delete the smoke directory when done. The mappers are 0.3 / 1.25 /
2.5 / 5 GiB for k = 1 / 4 / 8 / 16.

Part B needed more, because it had never run against real workers. Four
defects, each fixed:

- **Scripts were not executable.** `serve_pair.sh`, `mint_keys.py` and
  `run_pair.py` were committed as mode 644, so `./serve_pair.sh` failed with
  exit 126. They are 755 now.
- **`serve_pair.sh`: two models on one endpoint.** Dynamo refuses a second
  model on an endpoint that already serves one (`Cannot register model ...
  a different model ... is already registered there`), so the second worker
  crashed and the cleanup trap took the first down with it. Each worker now
  gets its own endpoint, `dyn://dynamo.target.generate` and
  `dyn://dynamo.source.generate`; the shared namespace keeps both visible to
  the one frontend.
- **Roundhouse engine: a policy that admits only the second local model was
  refused.** The `target-only` project returned `no target this key may use
  was admissible` for every turn. The engine decided whether to quote the
  fleet from the *primary* model's identity alone, so a policy excluding the
  primary skipped every local quote, including the one for the model the
  policy allowed. The skip is now decided across the whole local fleet, and
  each model is checked under its own identity (`engine/selection.rs`,
  `engine.rs`). The guard is
  `crates/roundhouse-server/tests/extra_local_model_policy.rs`; it failed with
  `Routing(PolicyRefused)` before the fix and its control (primary admitted)
  passed both before and after.
- **`run_pair.py` could not be run twice.** It used a fixed
  `prompt_cache_key`, so a second run against a live Roundhouse deduplicated
  onto the stored responses: no GPU work, `served_by` `?`, TTFT ~30 ms. The
  key now carries a per-run id (`RUN_ID` overrides it).

Also added: `probe_ttft.py`, because the TTFT column of `run_pair.py` is not a
TTFT (see B3). `mint_keys.py` writes hashes into the tracked
`control-plane.json`, so that file shows as modified after B1.

## Part A results: Qwen3-8B → Qwen3-14B, k = 8

Run on one H100 80 GB, bf16, eager attention, transformers 4.57.6, torch
2.15. Calibration: 500 FineWeb-Edu sequences x 1,024 tokens, stride 4, so
128,000 token observations (A1 took 143 s; A2 took 48 s for four k values).
Benchmarks use the first 500 examples (WikiText: 60 chunks), so one standard
error is about 2 points on the accuracies.

| Task | target (ceiling) | **transfer** | source (floor) | retention | floor-normalized |
|---|---|---|---|---|---|
| HellaSwag `acc_norm` | 0.678 | **0.634** | 0.644 | 93.5% | 89.7% |
| ARC-Challenge `acc_norm` | 0.588 | **0.526** | 0.558 | 89.5% | 81.7% |
| WikiText perplexity (lower is better) | 7.19 | **8.38** | 8.11 | ratio 1.16 | |

Fit sanity check: the best single source layer explains R² 0.62 of a target
layer's KV on average (min 0.53, max 0.83), and the chosen source layers
follow depth (target layer 20 takes source layers 15–22).

Latency (mapper vs. target re-prefill, body only, eager, bf16):

| Tokens | mapper | re-prefill | speedup |
|---|---|---|---|
| 64 | 8.7 ms | 29.8 ms | 3.4x |
| 256 | 8.5 ms | 29.2 ms | 3.4x |
| 1,024 | 8.3 ms | 58.6 ms | 7.0x |
| 4,096 | 19.6 ms | 233.5 ms | 11.9x |
| 8,192 | 37.7 ms | 484.7 ms | 12.9x |

**What these say, plainly.** Retention against the target clears the
README's Tier-1 bar (>90% on HellaSwag, perplexity ratio under 1.3), and the
mapper is 3–13x faster than re-prefill. But the mapped cache does **not**
beat the source model alone on any task: HellaSwag is within noise (0.634 vs
0.644), ARC is 3 points below, and WikiText perplexity is higher than the
source's. This pair's source is already close to its target, so "retention"
is a weak signal here; compare against the `source` column. On this evidence,
moving a session from the 8B to the 14B with a mapped cache buys speed
against re-prefill but not quality over staying on the 8B. Only k = 8 was
evaluated; the k = 1, 4, 16 mappers are fitted but not scored (A5 is open).
Swept k and the full validation splits are the next step before relying on
these numbers.

## Part B: serving both models behind Roundhouse

This measures the handoff cost in the real stack. It needs a Dynamo + vLLM
environment, the Roundhouse release binary, and the weights from Part A. Part
A does not need any of it.

### B0. Environment, from scratch

Needs: one NVIDIA GPU with 80 GB, Docker (the user must be able to run
`docker`), `sudo` for the system packages, and about 100 GB of free disk
(weights, statistics, and Dynamo's Python environment).

**1. System packages and Rust.**

```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev \
    protobuf-compiler python3-dev cmake libzmq3-dev
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source "$HOME/.cargo/env"
```

The first seven are Dynamo's build dependencies; `libzmq3-dev` is what the
Roundhouse build needs (through `dynamo-kv-router`). You also need `uv`
(https://docs.astral.sh/uv/).

**2. Dynamo at the pinned revision.** Roundhouse's `Cargo.toml` pins
`ai-dynamo/dynamo` at `ac7b7513790ef1d619b46f805aea03c9f21200ba`; use the same
one for the workers, so the KV-event wire and block hashing match.

```bash
git clone https://github.com/ai-dynamo/dynamo.git ~/dynamo
cd ~/dynamo && git checkout ac7b7513790ef1d619b46f805aea03c9f21200ba
uv venv .venv && source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
(cd lib/bindings/python && maturin develop --uv)   # compiles the Rust bindings, ~4 min
uv pip install -e lib/gpu_memory_service
uv pip install -e '.[vllm]'                        # torch 2.11, vllm 0.26; several GB
python -c "import dynamo.vllm" && echo ok
```

If the revision in `Cargo.toml` has moved since this was written, re-read that
revision's own build docs before reusing these steps.

On CUDA 13, vLLM's FlashInfer sampler can fail to JIT-compile against
version-skewed headers and abort the worker at startup. `serve_pair.sh` sets
`VLLM_USE_FLASHINFER_SAMPLER=0` by default for this; leave it unless you have
fixed the toolchain.

**3. etcd and nats**, which Dynamo uses for discovery and events:

```bash
cd ~/dynamo && docker compose -f dev/docker-compose.yml up -d
docker ps    # dev-etcd-server-1 and dev-nats-server-1 should be Up
```

**4. The Roundhouse binary**, from the repository root:

```bash
cargo build -p roundhouse-server --bin roundhouse --release
```

**5. Weights and the experiment environment.** A0 above created the
experiment's `.venv` and pulled both models into the Hugging Face cache; Part
B reuses both. `env.sh` needs `huggingface_hub` (it finds the tokenizer in the
cache), so run it from that `.venv`. `serve_pair.sh` must run in the **Dynamo**
venv from step 2. `mint_keys.py`, `run_pair.py` and `probe_ttft.py` use only
the standard library and run in either.

### B1. Mint keys and start Roundhouse (first)

Start Roundhouse **before** the workers. It learns what each worker has cached
from the KV events the worker publishes and cannot replay events it missed
(the log says `Gap detected: expected seq 0, got 64 ... batches lost`). A
Roundhouse that starts late, or is restarted alone, believes the worker holds
nothing, and the `cached` column then reads 0% for a worker that is in fact
warm. It does not need Dynamo at boot. Observed: a Roundhouse restart after
dozens of requests lost the 8B's cache state, and every source-only turn read
0%; two smoke requests before it started did no harm. After any Roundhouse
restart, restart the pair too.

```bash
cd roundhouse/experiments/cross-model-kv/serving
source ../.venv/bin/activate
python mint_keys.py     # fills control-plane.json hashes, writes keys.local.json (gitignored)
source env.sh           # ROUNDHOUSE_LOCAL_* for worker 1, ROUNDHOUSE_LOCAL_WORKER_2_* for worker 2
../../../target/release/roundhouse
```

The boot log should print two `local fleet: real worker registered` lines, one
per model. If it refuses to start, the message names the variable; the usual
one is a missing `ROUNDHOUSE_LOCAL_WORKER_2_KV_EVENTS_ENDPOINT`.
`mint_keys.py` writes hashes into the tracked `control-plane.json`; the secrets
go to `keys.local.json`.

`control-plane.json` and `env.sh` name the default pair. For another pair, edit
the three `allow` lists in `control-plane.json` and the `local_quality` entries
in `catalog.json` to the served model names, and export `SOURCE`/`TARGET`
before `source env.sh`.

### B2. Serve the pair

In a second terminal, in the Dynamo venv:

```bash
source ~/dynamo/.venv/bin/activate
cd roundhouse/experiments/cross-model-kv/serving
./serve_pair.sh                       # Qwen3-8B (0.30 of the card) + Qwen3-14B (0.55)
# or: SOURCE=Qwen/Qwen3-4B TARGET=Qwen/Qwen3-8B SOURCE_MEM=0.2 TARGET_MEM=0.35 ./serve_pair.sh
```

One Dynamo frontend on `:8000` serves both models; each vLLM worker publishes
KV events on its own ZMQ port (`20080` source, `20081` target) and registers on
its own endpoint (Dynamo refuses two models on one). The target starts first
and the source after `STAGGER_S` (90 s) so the two `--gpu-memory-utilization`
fractions do not race. Wait until both models are listed, then smoke-test:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-14B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

### B3. Run the handoff measurement

```bash
python run_pair.py                    # HANDOFF_AT=4 by default
```

Output is a per-turn table: input tokens, cached tokens, cache %, output
tokens, time to first token, and which worker served the turn. Three
phases:

1. `source-only`, turns 1–4: the source warms (cache % climbs to >90% as the
   corpus prefix is re-sent each turn).
2. `target-only`, turns 5–8, same conversation: **turn 5 is the handoff** —
   the target has none of the prefix, so `cached=0` and TTFT jumps. That
   uncached count, printed after the phase, is the prefill a mapped source
   cache would have skipped. Turns 6–8 warm again.
3. `pair`, under `local/*`: the router's own choice per turn, unforced.

Raw rows and the `/v1/metrics` snapshot go to `serving/runs/`.

Three things about reading the table:

- **`ttft_ms` is not a time to first token.** Roundhouse returns a local
  turn's text after the worker has finished it, so the column equals total
  turn time (it was 811 ms for a 49-token answer and ~3.8 s for 256 tokens,
  with the same prompt size). Use `probe_ttft.py` for TTFT (below).
- **`cached` is Roundhouse's view of the worker's KV**, built from its KV
  events, not a count the worker reports. See the ordering note in B1.
- **The handoff is only cold if the target has not seen the prefix.** The
  corpus prefix stays in the target's cache after a run, so a second run
  without restarting the pair shows turn 5 partly warm (1,472 of 2,599 cached
  in one such run). Restart the pair between runs.

```bash
python probe_ttft.py     # true cold and warm TTFT per model, straight from Dynamo
```

It streams with `max_tokens=1` at 512 to 6K tokens, each prompt once cold (a
random head defeats the prefix cache) and once warm, median of 3.

## Part B results: Qwen3-8B + Qwen3-14B behind Roundhouse

One H100, vLLM 0.26, Dynamo `ac7b751`, `--enforce-eager`, block size 64,
`max_model_len` 8192, `HANDOFF_AT=4`; Roundhouse started before the pair.

Forced handoff, same conversation, `run_pair.py`:

| Turn | Phase | Input tokens | Cached | Worker |
|---|---|---|---|---|
| 1 | source-only | 1,487 | 0 (0%) | 8B |
| 2 to 4 | source-only | 1,767 to 2,328 | 83 to 88% | 8B |
| **5** | **target-only (handoff)** | **2,599** | **0 (0%)** | **14B** |
| 6 to 8 | target-only | 2,880 to 3,439 | 89 to 91% | 14B |

The handoff turn re-prefills all 2,599 tokens on the target, though the source
held the shared prefix. That is the work a mapped cache would skip. Under the
unforced `pair` key the router stayed on the 8B for all eight turns, so it did
not hand off on its own with these catalog numbers (`local_quality` 0.66 vs
0.72). Why was not investigated; the 8B held the cached prefix, which is the
likely reason, but that is a guess. (In an earlier run, after the pair had
served the prefix on the 14B, the router chose the 14B for all eight turns.)

Prefill cost from Dynamo (`probe_ttft.py`, TTFT in ms, median of 3):

| Prefix tokens | 8B cold | 8B warm | 14B cold | 14B warm |
|---|---|---|---|---|
| ~530 | 49 | 24 | 61 | 27 |
| ~1,024 | 82 | 27 | 104 | 31 |
| ~2,050 | 157 | 27 | 195 | 29 |
| ~4,060 | 302 | 30 | 381 | 31 |
| ~6,110 | 454 | 31 | 577 | 32 |

Cold TTFT grows about linearly with the prefix (14B: ~93 ms per 1,000 tokens);
a warm hit stays near 30 ms. So the cold prefill a handoff pays on the 14B is
about 195 ms at 2K tokens and 381 ms at 4K, against a mapper that Part A timed
at 8 to 20 ms for 1K to 4K tokens. Do not divide one by the other: the vLLM
figures include the frontend and scheduler, the mapper figure is the mapper
alone, and shipping the mapped cache into the target (the connector, below) is
not in it.

## Other pairs

| Pair | Weights (bf16) | Statistics | Fits 1x H100? |
|---|---|---|---|
| Qwen3-4B → 8B | 24 GB | 23 GB | yes, fastest |
| **Qwen3-8B → 14B** | 44 GB | 23 GB | yes (default) |
| Qwen3-14B → 32B (paper) | 92 GB | 35 GB | no: 2 cards, or collect stats with the models loaded one at a time (not implemented) |
| Qwen2.5-Coder-14B → 32B | 92 GB | 35 GB | no, as above; 7B does not match (4 KV heads) |

Any dense Qwen3 pair is matched-KV (8 x 128). Llama 3.1 8B → 70B is matched
but 70B alone exceeds one card.

## What is not here, and what it takes

**Injecting the mapped cache into the serving target.** The paper measures
the mapper in isolation ("end-to-end transfer would also include shipping
the mapped cache to the target process, which we do not measure", appendix
G). Part B measures the cost it would remove; closing the loop means:

1. A vLLM **KV connector** (`KVConnectorBase_V1`) on the target worker that,
   for a request whose prefix the source holds, receives the mapped K/V and
   writes it into the target's paged cache blocks, then reports those
   tokens as computed so the scheduler skips their prefill. LMCache's
   connector is the reference shape. The mapper's keys come out in content
   space; the connector re-applies the target's RoPE per block position
   (`kvmap.rope.apply_rope`) before writing, since vLLM stores rotated keys.
2. A **source-side export** of the same prefix's K/V (the mirror connector,
   or a sidecar that re-prefills on the source).
3. In Roundhouse, the **transfer decision**: when the router moves a
   session between the pair, it already knows the prefix the source holds
   (`matched_prefix_tokens` on the source's quote). "Map vs. re-prefill"
   belongs with the per-turn corrections in
   `crates/roundhouse-core/src/routing/learn/corrections.rs`, priced with
   the retention and latency Part A measured for this pair.

## Layout

```
kvmap/            library: rope.py cache.py stats.py mapper.py scorer.py data.py tasks.py
collect_stats.py  A1
fit_mapper.py     A2
eval_transfer.py  A3
bench_latency.py  A4
tests/            CPU tests over toy shapes
serving/          B: serve_pair.sh env.sh catalog.json control-plane.json mint_keys.py run_pair.py probe_ttft.py corpus.md turns.jsonl
```
