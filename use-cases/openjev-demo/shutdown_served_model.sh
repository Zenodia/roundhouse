#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Spin down the Dynamo processes serve_model.sh started -- independently of
# each other, and independently of the shell serve_model.sh is running in.
#
# Why this script exists (not just Ctrl+C on serve_model.sh's terminal):
# serve_model.sh backgrounds `dynamo.frontend` and `dynamo.vllm` as two jobs
# of *one* shell and sets `trap 'kill 0' EXIT` so that Ctrl+C (or the shell
# exiting for any reason) takes both down together -- correct for the normal
# "I'm done, tear it all down" case, but exactly wrong for "I need to restart
# just the frontend with a different flag": killing the frontend from inside
# that shell fires the trap and kills the worker underneath it too, discarding
# several minutes of weight loading for a change that only touched the
# frontend. (This is not hypothetical -- it is what happened restarting the
# frontend with DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE set while validating
# openjev-demo's real log-probability scoring, 2026-09-28.)
#
# This script kills by process pattern from an independent shell instead, so
# it never touches serve_model.sh's own job table or its trap, and each half
# can be brought down (and restarted) without disturbing the other.
#
# Usage:
#   ./shutdown_served_model.sh                 # stop both frontend and worker
#   ./shutdown_served_model.sh --frontend      # stop only dynamo.frontend
#   ./shutdown_served_model.sh --worker        # stop only dynamo.vllm
#   ./shutdown_served_model.sh --all           # also bring down etcd + nats
#   ./shutdown_served_model.sh --status        # report what's running, change nothing
#
# GPU memory is not actually freed until the worker process (dynamo.vllm) has
# fully exited -- `nvidia-smi` after this script returns is the way to confirm
# it, not the script's own exit code.

set -uo pipefail

FRONTEND_PATTERN="dynamo\.frontend"
WORKER_PATTERN="dynamo\.vllm"
GRACE_SECONDS="${GRACE_SECONDS:-10}"

# --- helpers -----------------------------------------------------------------

pids_for() {
  # `pgrep -f` matches the pattern against the whole command line, which is
  # what distinguishes `python -m dynamo.frontend` from `python -m dynamo.vllm`
  # -- both are `python`/`python3` at argv[0].
  pgrep -f "$1" 2>/dev/null || true
}

report_status() {
  local frontend_pids worker_pids
  frontend_pids=$(pids_for "$FRONTEND_PATTERN")
  worker_pids=$(pids_for "$WORKER_PATTERN")
  echo "dynamo.frontend : ${frontend_pids:-not running}"
  echo "dynamo.vllm     : ${worker_pids:-not running}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU memory      : $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader)"
  fi
}

# Graceful (SIGTERM, wait, SIGKILL only if still alive) rather than an
# immediate `-9` -- vLLM's worker process manages GPU memory and CUDA
# contexts, and killing it mid-cleanup is how you get a wedged GPU that
# needs `nvidia-smi --gpu-reset` or a reboot to recover, not just this
# process restarting cleanly next time.
stop_pattern() {
  local label="$1" pattern="$2"
  local pids
  pids=$(pids_for "$pattern")
  if [[ -z "$pids" ]]; then
    echo "  ${label}: not running, nothing to do"
    return 0
  fi
  echo "  ${label}: stopping PID(s) ${pids} (SIGTERM, up to ${GRACE_SECONDS}s grace)"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  local waited=0
  while [[ $waited -lt $GRACE_SECONDS ]]; do
    pids=$(pids_for "$pattern")
    [[ -z "$pids" ]] && { echo "  ${label}: stopped"; return 0; }
    sleep 1
    waited=$((waited + 1))
  done
  pids=$(pids_for "$pattern")
  if [[ -n "$pids" ]]; then
    echo "  ${label}: still alive after ${GRACE_SECONDS}s, sending SIGKILL to ${pids}"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
  pids=$(pids_for "$pattern")
  if [[ -n "$pids" ]]; then
    echo "  ${label}: WARNING -- still running (${pids}) after SIGKILL; check manually" >&2
    return 1
  fi
  echo "  ${label}: stopped"
}

stop_infra() {
  local compose_dir="${DYNAMO_CLONE:-}"
  if [[ -z "$compose_dir" ]]; then
    for candidate in "$HOME/dynamo" "$HOME/ai-dynamo" "$HOME/repos/dynamo"; do
      [[ -f "$candidate/dev/docker-compose.yml" ]] && { compose_dir="$candidate"; break; }
    done
  fi
  if [[ -z "$compose_dir" || ! -f "$compose_dir/dev/docker-compose.yml" ]]; then
    echo "  etcd/nats: could not find dev/docker-compose.yml (set DYNAMO_CLONE=/path/to/dynamo); skipped" >&2
    return 0
  fi
  echo "  etcd/nats: docker compose down (${compose_dir}/dev)"
  (cd "$compose_dir/dev" && docker compose down)
}

# --- main ----------------------------------------------------------------

case "${1:-both}" in
  --status)
    report_status
    ;;
  --frontend)
    stop_pattern "dynamo.frontend" "$FRONTEND_PATTERN"
    ;;
  --worker)
    stop_pattern "dynamo.vllm" "$WORKER_PATTERN"
    ;;
  --all)
    stop_pattern "dynamo.frontend" "$FRONTEND_PATTERN"
    stop_pattern "dynamo.vllm" "$WORKER_PATTERN"
    stop_infra
    ;;
  both|"")
    stop_pattern "dynamo.frontend" "$FRONTEND_PATTERN"
    stop_pattern "dynamo.vllm" "$WORKER_PATTERN"
    ;;
  -h|--help)
    sed -n '2,29p' "$0"
    exit 0
    ;;
  *)
    echo "usage: $0 [--frontend|--worker|--all|--status|both]" >&2
    exit 2
    ;;
esac

echo
echo "Current state:"
report_status
