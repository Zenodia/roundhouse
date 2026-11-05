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

On the GPU machine:

```bash
cd roundhouse/experiments/cross-model-kv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional, for the paper's flash_attention_2 re-prefill timing: pip install flash-attn --no-build-isolation
huggingface-cli login   # only if your Hub access needs it; Qwen3 is public
python -m pytest tests -q   # CPU-only math tests; also runs on a laptop
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
HellaSwag takes roughly 10–15 minutes on an H100 (3 modes x 4 choices x 500).
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

## Part B: serving both models behind Roundhouse

This measures the handoff cost in the real stack. It needs the Roundhouse
binary from this branch (the one that registers N local workers) and a
Dynamo + vLLM environment.

### B0. Environment

1. **Dynamo at the pinned rev.** Follow
   `agent-docs/cross-model-kv-transfer/DYNAMO_LOCAL_SERVING.md` (a from-source
   install validated on a 1x H100 box): clone `ai-dynamo/dynamo` at
   `ac7b7513790ef1d619b46f805aea03c9f21200ba`, build, `uv pip install -e
   '.[vllm]'`, and bring up etcd and nats with
   `docker compose -f dev/docker-compose.yml up -d`.
2. **Roundhouse binary**, built on the GPU machine or copied there:
   `cargo build -p roundhouse-server --bin roundhouse --release`.
3. Weights pulled (A0 did this if you share the HF cache).

### B1. Serve the pair

```bash
cd roundhouse/experiments/cross-model-kv/serving
./serve_pair.sh                       # Qwen3-8B (0.30 of the card) + Qwen3-14B (0.55)
# or: SOURCE=Qwen/Qwen3-4B TARGET=Qwen/Qwen3-8B SOURCE_MEM=0.2 TARGET_MEM=0.35 ./serve_pair.sh
```

One Dynamo frontend on `:8000` serves both models; each vLLM worker
publishes KV events on its own ZMQ port (`20080` source, `20081` target).
The target starts first and the source after `STAGGER_S` (90 s) so the two
`--gpu-memory-utilization` fractions do not race. Smoke-test both:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:8000/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-14B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
```

### B2. Mint keys and start Roundhouse

```bash
python mint_keys.py                   # fills control-plane.json hashes, writes keys.local.json (gitignored)
source env.sh                         # ROUNDHOUSE_LOCAL_* for worker 1 and ROUNDHOUSE_LOCAL_WORKER_2_* for worker 2
../../../target/release/roundhouse
```

The boot log should print two `local fleet: real worker registered` lines,
one per model. If it refuses to start, the message names the variable; the
usual one is a missing `ROUNDHOUSE_LOCAL_WORKER_2_KV_EVENTS_ENDPOINT`.

`control-plane.json` and `env.sh` name the default pair. For another pair,
edit the three `allow` lists in `control-plane.json` and the
`local_quality` entries in `catalog.json` to the served model names, and
export `SOURCE`/`TARGET` before `source env.sh`.

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
serving/          B: serve_pair.sh env.sh catalog.json control-plane.json mint_keys.py run_pair.py corpus.md turns.jsonl
```
