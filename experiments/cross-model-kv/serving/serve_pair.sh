#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Serve a source/target model pair on one GPU behind one Dynamo frontend, each
# worker publishing its own KV-cache events, so Roundhouse can register both
# as local workers and route between them.
#
# Derived from Dynamo's examples/backends/vllm/launch/agg_router.sh at the pinned
# rev ac7b7513790ef1d619b46f805aea03c9f21200ba. Prerequisites: run it in the
# Dynamo venv (Dynamo + vLLM importable), etcd and nats up (`docker compose -f
# dev/docker-compose.yml up -d` in the Dynamo clone), weights pulled. The README
# (Part B) has the full from-scratch setup.
#
# Memory on a single 80 GB card is split by --gpu-memory-utilization, which
# vLLM reads as a fraction of the *whole* card per worker. The defaults below
# give Qwen3-8B (16 GB weights) 0.30 and Qwen3-14B (28 GB) 0.55, leaving ~12 GB
# of KV cache to each. Raise/lower per pair; the two fractions must sum to
# less than 1.0 with headroom for CUDA context.
#
# Usage:
#   ./serve_pair.sh
#   SOURCE=Qwen/Qwen3-4B TARGET=Qwen/Qwen3-8B SOURCE_MEM=0.2 TARGET_MEM=0.35 ./serve_pair.sh

set -euo pipefail

SOURCE="${SOURCE:-Qwen/Qwen3-8B}"
TARGET="${TARGET:-Qwen/Qwen3-14B}"
SOURCE_MEM="${SOURCE_MEM:-0.30}"
TARGET_MEM="${TARGET_MEM:-0.55}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# MUST match ROUNDHOUSE_LOCAL_BLOCK_SIZE (default 64). The Dynamo indexer drops
# KV events whose block_size differs from its own view.
BLOCK_SIZE="${BLOCK_SIZE:-64}"
GPU="${GPU:-0}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SOURCE_KV_PORT="${SOURCE_KV_PORT:-20080}"
TARGET_KV_PORT="${TARGET_KV_PORT:-20081}"
SOURCE_SYSTEM_PORT="${SOURCE_SYSTEM_PORT:-8081}"
TARGET_SYSTEM_PORT="${TARGET_SYSTEM_PORT:-8082}"

# Deterministic hashing so the block hashes Roundhouse computes match the ones
# the workers publish.
export PYTHONHASHSEED=0
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if ! python -c "import dynamo.vllm" >/dev/null 2>&1; then
  echo "ERROR: 'python -m dynamo.vllm' is not importable here. Activate the Dynamo venv/container first." >&2
  exit 1
fi

trap 'echo "Cleaning up..."; kill 0' EXIT

echo "=================================================================="
echo " Dynamo: source + target behind one frontend"
echo "   source        : ${SOURCE}  (gpu mem ${SOURCE_MEM}, KV events tcp://*:${SOURCE_KV_PORT})"
echo "   target        : ${TARGET}  (gpu mem ${TARGET_MEM}, KV events tcp://*:${TARGET_KV_PORT})"
echo "   block_size    : ${BLOCK_SIZE}"
echo "   HTTP (OpenAI) : http://0.0.0.0:${HTTP_PORT}"
echo "=================================================================="

python -m dynamo.frontend --router-mode kv --http-port "${HTTP_PORT}" &

worker() {
  local role="$1" model="$2" mem="$3" kv_port="$4" sys_port="$5"
  # Dynamo refuses a second model on an endpoint that already serves one, so
  # each worker gets its own component; the shared namespace keeps both
  # visible to the one frontend.
  DYN_SYSTEM_PORT="${sys_port}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python3 -m dynamo.vllm \
    --endpoint "dyn://dynamo.${role}.generate" \
    --model "${model}" \
    --block-size "${BLOCK_SIZE}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${mem}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enable-prefix-caching \
    --enforce-eager \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${kv_port}\",\"enable_kv_cache_events\":true}" &
}

# Start the larger model first so its allocation is not fragmented by the
# smaller one's; wait for it to settle before the second claims its share.
worker target "${TARGET}" "${TARGET_MEM}" "${TARGET_KV_PORT}" "${TARGET_SYSTEM_PORT}"
sleep "${STAGGER_S:-90}"
worker source "${SOURCE}" "${SOURCE_MEM}" "${SOURCE_KV_PORT}" "${SOURCE_SYSTEM_PORT}"

wait -n
