<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Environment setup — porting this repo to a fresh GPU box

A single, ordered, from-scratch runbook for standing up everything the use cases under this
directory need, on a machine that has never seen this repo before. Written so that porting
roundhouse to a new H100 (or equivalent) box and running the use cases end to end doesn't
require re-discovering the gotchas this repo's own history already hit — every step below was
actually run and validated (`cache-aware-routing/DYNAMO_LOCAL_SERVING.md`,
`openjev-demo/PLAN.md`), not assembled from documentation alone.

**What "end to end" means here:** by the end of this document you will have (1) roundhouse
built and its own test suite passing, (2) a real Qwen model served locally by Dynamo, (3)
`cache-aware-routing`'s and `openjev-demo`'s use cases actually running against that worker —
not just configured to. Section 12 is the validation step; if it doesn't produce real output,
something earlier is wrong.

**Scope.** This covers the local-tier (Dynamo/GPU) setup every use case under `use-cases/`
needs. It does not cover frontier-only setup (no GPU needed for that — see
`cache-aware-routing/README.md`'s frontier-only steps) or NVIDIA driver / CUDA toolkit
installation itself (assumed already present — `nvidia-smi` and `nvcc --version` should both
work before you start).

---

## Prerequisites

| Prerequisite | Notes |
|---|---|
| An NVIDIA GPU with ≥80 GB memory | Validated on 1× H100 80GB. A 32B model in bf16 needs ~62 GB just for weights — see §9's KV-cache note for what fits on less. |
| Ubuntu (or a similarly apt-based Linux) | Every system-package step below assumes `apt`. |
| `sudo` access | Several steps install system packages. |
| ~100 GB free disk | ~62 GB model weights + several GB of Rust build artifacts + the Dynamo Python environment. |
| Outbound network to GitHub, crates.io, and Hugging Face | Clones two repos, resolves Rust crates, downloads model weights. |
| Rust toolchain | Installed in §2 if you don't have one; version is pinned by this repo's `rust-toolchain.toml`. |
| `uv` (https://docs.astral.sh/uv/) | Installed in §2; everything Dynamo-Python-side goes into a `uv venv`, no system `pip` needed. |

---

## 1. Clone this repo

```bash
git clone <this repo's URL> roundhouse
cd roundhouse
```

Everything below assumes you're running from the repo root unless stated otherwise.

## 2. Rust toolchain and `uv`

```bash
# Rust, if you don't already have one — this repo pins an exact version in rust-toolchain.toml
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"

# uv, for the Dynamo Python environment in §6
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 3. System packages (needs sudo)

```bash
sudo apt update
sudo apt install -y \
    build-essential libhwloc-dev libudev-dev pkg-config libclang-dev \
    protobuf-compiler python3-dev cmake \
    libzmq3-dev \
    libssl-dev
```

Three groups, for three different reasons:
- `build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler
  python3-dev cmake` — Dynamo's own documented build dependencies (its
  `docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md`, in the
  clone you'll make in §6).
- `libzmq3-dev` — roundhouse's own `cargo build` needs this transitively, via
  `dynamo-kv-router`'s `standalone-selection` feature (see this repo's root `README.md`,
  "Build and test").
- `libssl-dev` — needed to run roundhouse's full test suite (not just `cargo build`/`cargo
  check`): the `codex-api` dev-dependency (the wire-conformance oracle for `/v1/responses`)
  pulls in `openssl-sys`, deliberately kept off the shipped binary's own dependency graph but
  still needed to compile the test binaries that exercise it. Skippable if you only intend to
  `cargo check`/`cargo build`, not `cargo test`.

## 4. Build and test roundhouse itself

```bash
cargo build -p roundhouse-server --bin roundhouse
timeout 900 cargo test -p roundhouse-server
```

The first build clones `ai-dynamo/dynamo` automatically (a pinned git dependency in
`Cargo.toml`) to resolve `dynamo-kv-router`/`dynamo-tokens` — expect the first build to take a
few minutes; this is separate from, and does not satisfy, the Python-side Dynamo install in §6,
which is for actually *serving* a model, not for building roundhouse against Dynamo's Rust
crates. `timeout 900` is load-bearing, not decorative — see the root `CLAUDE.md`'s "Every test
run is bounded."

Both commands should complete with zero failures before you go further. If they don't, stop
here — nothing past this point will work either.

## 5. Confirm the pinned Dynamo revision

```bash
grep -A3 'ai-dynamo/dynamo' Cargo.toml
```

Everything below is validated against a specific commit of `ai-dynamo/dynamo`. If this repo's
pin has moved since this document was written, re-verify §6–§9 against that new revision's own
`docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md` before
trusting the commands verbatim — this repo's own `CLAUDE.md` calls this out explicitly
("synergy dependencies are watched, not just pinned"), and this document has already had to be
corrected once for exactly this reason (§8's `docker-compose.yml` path moved between revisions).

## 6. Set up Dynamo's Python environment (for serving a model)

```bash
git clone https://github.com/ai-dynamo/dynamo.git ~/dynamo
cd ~/dynamo
# Same revision §5 printed (Cargo.toml pins it 3x, once per crate -- sort -u/head -1
# collapses that to the one value git checkout actually needs):
git checkout "$(grep -A3 'ai-dynamo/dynamo' <roundhouse-repo-root>/Cargo.toml \
  | grep -oP 'rev = "\K[a-f0-9]+' | sort -u | head -1)"
# or just paste the revision from §5's output directly:
#   git checkout <revision from §5>

uv venv .venv
source .venv/bin/activate
uv pip install pip 'maturin[patchelf]'
cd lib/bindings/python && maturin develop --uv && cd ~/dynamo   # ~3.5 min, compiles dynamo-kv-router etc.
uv pip install -e lib/gpu_memory_service
uv pip install -e '.[vllm]'    # pulls torch, vllm, tilelang — several GB, several minutes

python3 -c "import dynamo.vllm"   # should print nothing / exit 0
```

## 7. CUDA-13 gotcha (skip if your CUDA toolkit is 12.x)

```bash
nvcc --version   # check which major version you have
```

If it's CUDA 13.x: vLLM's FlashInfer sampler JIT-compiles against version-skewed headers
(`torch` pins runtime headers to 13.0, vLLM's `tilelang` dependency pulls `nvidia-cuda-nvcc`
13.2) and the worker aborts at startup with a `cuda_toolkit.h` incompatibility error — a known
upstream issue ([flashinfer#3493](https://github.com/flashinfer-ai/flashinfer/issues/3493)).
Work around it:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

`use-cases/cache-aware-routing/serve_model.sh` (used in §10) sets this by default, so you only
need to export it by hand if you're driving `dynamo.vllm` directly rather than through that
script.

## 8. Bring up etcd + nats

```bash
cd ~/dynamo/dev && docker compose up -d
```

If this fails because `dev/docker-compose.yml` doesn't exist at the revision you checked out,
the file has moved again — check `git log --oneline -- '**/docker-compose.yml'` in the Dynamo
clone to find its current location, and consider this a signal to re-run §5's revision check.

## 9. Pull the model weights

```bash
cd <roundhouse repo root>
MODEL=Qwen/Qwen2.5-Coder-32B-Instruct ./use-cases/cache-aware-routing/pull_model.sh pull
```

~62 GB. **If your GPU has less than ~80 GB**, or you hit an out-of-memory error in §10, fall
back to the smaller model instead of troubleshooting the 32B further:

```bash
MODEL=Qwen/Qwen2.5-Coder-14B-Instruct ./use-cases/cache-aware-routing/pull_model.sh pull
```

(~28 GB weights, leaves substantially more KV-cache headroom — see
`cache-aware-routing/GAPS.md`'s 2026-09-23 addendum for the exact numbers this repo measured
on a single H100 80GB: 32B fit with only 1.25× concurrency headroom at `max_model_len=32768`.)
Use the same `MODEL=` override in §10 if you fall back here.

## 10. Serve the model

```bash
GPUS=0 TP=1 MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
    ./use-cases/cache-aware-routing/serve_model.sh serve
```

`GPUS=0 TP=1` is this document's single-GPU default; override `GPUS=0,1,2,3 TP=4` (etc.) on a
real multi-GPU node — `TP` must divide the GPU count you expose. This starts `python -m
dynamo.frontend` (OpenAI-compatible HTTP on `:8000`) and `python -m dynamo.vllm` (publishing KV
events on ZMQ `:20080`, the stream roundhouse's `EmbeddedFleet` subscribes to) as two
background jobs of the current shell — leave this terminal open, or see §11 before you close it.

**Verify:**

```bash
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d \
  '{"model":"Qwen/Qwen2.5-Coder-32B-Instruct","messages":[{"role":"user","content":"Say OK"}],"max_tokens":10}'
```

Expect a real `chat.completion` object back with `"content":"OK"` (or similar) — not a
connection error, not a 404.

## 11. Shutting down (and restarting only half)

```bash
./use-cases/cache-aware-routing/shutdown_served_model.sh          # stop both frontend and worker
./use-cases/cache-aware-routing/shutdown_served_model.sh --status # report what's running, change nothing
```

**Do not just Ctrl+C the terminal §10 is running in** if you only meant to restart the
frontend, or the worker, on its own — `serve_model.sh` backgrounds both as jobs of one shell
with `trap 'kill 0' EXIT`, so exiting that shell for any reason takes both down together, and
restarting the worker alone means several minutes reloading 32B (or 14B) of weights.
`shutdown_served_model.sh --frontend` or `--worker` kills by process pattern from an
independent shell instead, so each half can be brought down and restarted without disturbing
the other. `--all` also brings down etcd/nats.

## 12. Validate end to end

Two independent use cases, both against the worker you just brought up — if both of these
produce real, non-error output, the whole pipeline (Dynamo → roundhouse → both use cases) is
confirmed working on this box.

```bash
# cache-aware-routing: roundhouse routing a turn to the local worker, real KV-cache-hit
# accounting climbing across turns
python3 use-cases/cache-aware-routing/mint_keys.py
TOKENIZER=$(find ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-32B-Instruct \
  -name tokenizer.json | head -1)
INFERENCE_API_KEY=unused-for-local-only-demo \
ROUNDHOUSE_CATALOG=use-cases/cache-aware-routing/catalog.json \
ROUNDHOUSE_CONTROL_PLANE=use-cases/cache-aware-routing/control-plane.json \
ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000 \
ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-Coder-32B-Instruct \
ROUNDHOUSE_LOCAL_TOKENIZER="$TOKENIZER" \
ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080 \
./target/debug/roundhouse &      # or cargo run --release -p roundhouse-server --bin roundhouse
sleep 2
python3 use-cases/cache-aware-routing/run.py

# openjev-demo: real, calibrated log-probability scoring against the same worker
# (needs its own roundhouse process with the score flag on -- stop the one above first,
# `ROUNDHOUSE_LOCAL_ENABLE_SCORE` is off by default and this use case is the one that sets it)
```

See each use case's own `README.md` for the exact launch commands and what "it worked" looks
like (`cache-aware-routing/README.md`'s "Local tier" section; `openjev-demo/README.md`'s "Run
it" and "Which script do I run?" sections) — this section is a pointer to run them, not a
duplicate of their own instructions, which are kept there so a fix to one doesn't require
remembering to also fix a copy here.

---

## 13. Optional: the real `open-jev` reference implementation

`openjev-demo` (this repo's own use case) is a **from-scratch reimplementation** of the
Jev/System-One *shape* against roundhouse's local tier — it does not depend on, call, or
require the actual `open-jev` project to exist anywhere. Skip this section entirely unless you
specifically want to run the real upstream project too, e.g. to compare its own local Gemma
scoring against roundhouse's `/v1/local/score` on the same tickets, or to validate that
`openjev-demo`'s ported prompt renderers and `system_one()` math still match upstream after it
changes.

```bash
git clone https://github.com/daseinlabs/open-jev.git ~/open-jev
cd ~/open-jev

# Python 3.12+ and uv (already installed in §2) required.
uv sync
uv run hf auth login                                              # accept the Gemma license on HF first
uv run hf download google/gemma-3-4b-it --local-dir models/gemma-3-4b-it
uv run openjev serve --backend torch --device auto --port 8000    # its own server, its own port
```

This is a **completely separate environment** from everything in §1–§12: its own `uv sync`
venv, its own model (Gemma 3 4B, not Qwen), its own port (`:8000` by default — if Dynamo's
frontend from §10 is also on `:8000`, change one of the two with `--port`). Nothing here
touches roundhouse, Dynamo, or this repo's own Python environment, and nothing in §1–§12
depends on this section having been run. See `open-jev`'s own README (in the clone) for its
`/score` and `/v1/systemone` endpoints, and `use-cases/openjev-demo/README.md`'s "Why Qwen, not
Gemma" section for why this repo's own use case deliberately doesn't require standing this up.

---

## Troubleshooting quick reference

| Symptom | Cause | Fix |
|---|---|---|
| `cuda_toolkit.h` incompatibility at worker startup | CUDA 13 + FlashInfer version skew | §7: `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| `docker compose up` fails, no such file | Dynamo's `docker-compose.yml` moved between revisions | §8: re-check the path at your pinned revision |
| `cargo test` fails to link, `openssl-sys` error | Missing `libssl-dev` | §3 |
| `cargo build`/`check` fails on a ZMQ-related crate | Missing `libzmq3-dev` | §3 |
| Restarting the frontend also killed the worker | `serve_model.sh`'s shared shell + exit trap | §11: use `shutdown_served_model.sh --frontend`/`--worker`, never Ctrl+C the `serve` terminal for a partial restart |
| OOM loading 32B | Insufficient GPU memory / another process holding it | §9: fall back to `MODEL=Qwen/Qwen2.5-Coder-14B-Instruct`; check `nvidia-smi` for other processes first |
| `/v1/local/score` returns 404 | `ROUNDHOUSE_LOCAL_ENABLE_SCORE` unset (off by default) | Set it — see `openjev-demo/README.md`'s "The `ROUNDHOUSE_LOCAL_ENABLE_SCORE` flag" |
| roundhouse refuses to boot, "provider nvidia not defined" or similar | Catalog missing the M10.1 `"providers"` block | See `cache-aware-routing/README.md`'s note on this exact failure mode |
