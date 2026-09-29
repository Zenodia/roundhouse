#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shuts down everything the sandbox-breach demo started on this machine, in the
# reverse order it was brought up, so it's safe to run before powering the box
# off. Every step is best-effort (`|| true`) -- a step that was never started
# (e.g. no roundhouse process running) is not an error, and later steps still
# run even if an earlier one fails.
#
# Usage:
#   ./shutdown_sandbox_breach_demo.sh            # tear down everything
#   ./shutdown_sandbox_breach_demo.sh --keep-infra # leave etcd/nats + the
#                                                    OpenShell gateway running
#                                                    (only stop the demo's own
#                                                    sandbox/roundhouse/Dynamo/
#                                                    memory-mcp) -- useful if
#                                                    you're about to run a
#                                                    different use case next,
#                                                    not shutting the box down

set -uo pipefail

KEEP_INFRA=0
[[ "${1:-}" == "--keep-infra" ]] && KEEP_INFRA=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BREAKOUT_LAB="${BREAKOUT_LAB:-/home/ubuntu/breakout-lab}"
SANDBOX_NAME="${SANDBOX_NAME:-sandbox-breach-demo}"

step() { echo; echo "[*] $*"; }

step "OpenShell sandbox (${SANDBOX_NAME})"
openshell sandbox delete "${SANDBOX_NAME}" 2>&1 || true

step "roundhouse server"
pkill -f "target/release/roundhouse\b" 2>&1 || true

step "Dynamo/Qwen local worker (frontend + vLLM worker)"
pkill -f "dynamo\.frontend" 2>&1 || true
pkill -f "dynamo\.vllm" 2>&1 || true

step "memory-mcp server (breakout-lab, outside the sandbox)"
if [[ -f "${BREAKOUT_LAB}/openshell/docker-compose.memory-mcp.yml" ]]; then
  # --env-file is required here, not just for `up`: compose interpolates
  # ${NVIDIA_API_KEY:?...} for every command including `down`, and without it
  # this fails at interpolation before it ever touches the container --
  # verified empirically (2026-09-29): the first version of this script
  # dropped --env-file, `down` errored out, and `|| true` silently swallowed
  # it, leaving memory-mcp running after a "Done" shutdown.
  ( cd "${BREAKOUT_LAB}/openshell" && docker compose --env-file ../.env -f docker-compose.memory-mcp.yml down ) 2>&1 || true
fi
# Unconditional fallback regardless of whether the compose invocation above
# worked -- catches the case where the compose file is missing/renamed, and
# is a no-op if `down` above already removed the container.
docker stop memory-mcp 2>&1 || true
docker rm memory-mcp 2>&1 || true

if [[ "${KEEP_INFRA}" -eq 0 ]]; then
  step "etcd + nats (Dynamo's control plane)"
  docker stop dev-etcd-server-1 dev-nats-server-1 2>&1 || true

  step "OpenShell gateway (native, systemd --user)"
  systemctl --user stop openshell-gateway 2>&1 || true
else
  echo
  echo "[*] --keep-infra: leaving etcd/nats and the OpenShell gateway running"
fi

step "Verifying GPU is free"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv 2>&1 || true

step "Remaining containers (should be empty, or only --keep-infra's etcd/nats/gateway-unrelated ones)"
docker ps --format '{{.Names}}\t{{.Status}}' 2>&1 || true

step "Remaining sandbox-breach processes (should be empty)"
pgrep -fa "target/release/roundhouse\b|dynamo\.frontend|dynamo\.vllm" 2>&1 || echo "  (none)"

echo
echo "Done. Safe to shut the machine down now (or run again with --keep-infra"
echo "first if you want etcd/nats/the OpenShell gateway to survive a reboot)."
