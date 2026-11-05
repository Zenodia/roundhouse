# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Environment for a Roundhouse process that routes between the two local
# workers serve_pair.sh started. `source` this, then run the binary.
#
#   source experiments/cross-model-kv/serving/env.sh
#   ./target/release/roundhouse
#
# Worker 1 (primary) uses the plain ROUNDHOUSE_LOCAL_* names; worker 2 the
# ROUNDHOUSE_LOCAL_WORKER_2_* names. Both point at the same Dynamo frontend
# and differ by model name and KV-events port. The tokenizer is shared: one
# family, one vocabulary, which the mapper requires anyway.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SOURCE="${SOURCE:-Qwen/Qwen3-8B}"
export TARGET="${TARGET:-Qwen/Qwen3-14B}"

# Resolve the tokenizer.json inside the HF cache for the target (either works).
if [ -z "${ROUNDHOUSE_LOCAL_TOKENIZER:-}" ]; then
  ROUNDHOUSE_LOCAL_TOKENIZER="$(python3 - <<EOF
from huggingface_hub import hf_hub_download
print(hf_hub_download("${TARGET}", "tokenizer.json"))
EOF
)"
fi
export ROUNDHOUSE_LOCAL_TOKENIZER

export ROUNDHOUSE_LOCAL_ENDPOINT="${ROUNDHOUSE_LOCAL_ENDPOINT:-http://127.0.0.1:8000}"
export ROUNDHOUSE_LOCAL_MODEL="${SOURCE}"
export ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT="${ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT:-tcp://127.0.0.1:20080}"
export ROUNDHOUSE_LOCAL_BLOCK_SIZE="${ROUNDHOUSE_LOCAL_BLOCK_SIZE:-64}"

export ROUNDHOUSE_LOCAL_WORKER_2_MODEL="${TARGET}"
export ROUNDHOUSE_LOCAL_WORKER_2_ENDPOINT="${ROUNDHOUSE_LOCAL_ENDPOINT}"
export ROUNDHOUSE_LOCAL_WORKER_2_KV_EVENTS_ENDPOINT="${ROUNDHOUSE_LOCAL_WORKER_2_KV_EVENTS_ENDPOINT:-tcp://127.0.0.1:20081}"

export ROUNDHOUSE_CATALOG="${ROUNDHOUSE_CATALOG:-$HERE/catalog.json}"
export ROUNDHOUSE_CONTROL_PLANE="${ROUNDHOUSE_CONTROL_PLANE:-$HERE/control-plane.json}"
export ROUNDHOUSE_ADDR="${ROUNDHOUSE_ADDR:-127.0.0.1:8080}"
export RUST_LOG="${RUST_LOG:-info}"

echo "roundhouse local fleet: 1=${ROUNDHOUSE_LOCAL_MODEL}  2=${ROUNDHOUSE_LOCAL_WORKER_2_MODEL}  tokenizer=${ROUNDHOUSE_LOCAL_TOKENIZER}"
