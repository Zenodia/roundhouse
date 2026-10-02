# OpenShell + OpenClaw Integration — Sandbox Breakout Lab

This directory wires the (deliberately vulnerable) **Memory MCP server** into an
**OpenShell** deployment so you can demonstrate an **OpenClaw** agent breaking
out of its sandbox.

Tested target host: **Ubuntu 24.04.5 LTS (noble)** with Docker Engine + Compose
plugin and (optionally) an NVIDIA GPU.

```
+-------------------------- Ubuntu 24.04 host ---------------------------+
|                                                                       |
|  OpenShell gateway (docker compose, :8080)                            |
|     creates sandbox containers                                        |
|                                                                       |
|   +--------------- OpenClaw sandbox (via NemoClaw) ---------------+    |
|   |  OpenClaw agent                                              |    |
|   |     -> memory-recall Agent Skill (openshell/skills/...)       |    |
|   |          -> MCP client (recall.py)                            |    |
|   +------------------|-------------------------------------------+    |
|                      |  MCP over http (host.openshell.internal:8999)  |
|                      v                                                 |
|  memory-mcp service (docker-compose.memory-mcp.yml, :8999)  <-- OUTSIDE|
|     vulnerable sinks: os.system (CWE-78) + eval (CWE-94)               |
|     writes to ./host_breakout_marker  <-- proof of code OUTSIDE sandbox|
+-----------------------------------------------------------------------+
```

---

## Which Docker Compose file do I use?

There are **three** compose files in play. Pick a path, then use only the files
that path calls for.

| Compose file | Where it lives | What it starts | Used in |
| --- | --- | --- | --- |
| `docker-compose.yml` (gateway) | `OpenShell/deploy/docker/` (cloned from [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell)) | The OpenShell **gateway** on `:8080`, which then creates real sandboxes | **Path A only** |
| `docker-compose.memory-mcp.yml` | `openshell/` (this repo) | The **vulnerable Memory MCP server** on `:8999` (outside the sandbox) | **Path A and Path B** (always) |
| `docker-compose.override.yml` | `openshell/` (this repo) | A **Claude Code container** wired to the skill + server (a dev-loop shortcut, no gateway) | **Path B only** |

- **Path A — Full OpenShell + OpenClaw (the real isolation boundary).**
  Use the **gateway** compose **and** the **memory-mcp** compose. Do **not** use
  the override. This is the faithful reproduction of the threat model.
- **Path B — Quick dev-loop (no gateway).**
  Use the **memory-mcp** compose **combined with** the **override** compose
  (`-f docker-compose.memory-mcp.yml -f docker-compose.override.yml`). Faster to
  iterate, but the Claude Code container is a stand-in and is **not** a true
  OpenShell sandbox.

> Rule of thumb: `docker-compose.memory-mcp.yml` is **always** required (it's the
> "outside the sandbox" target). The gateway compose and the override compose are
> **mutually exclusive** — gateway for Path A, override for Path B.

---

## Step 0 — Prerequisites (Ubuntu 24.04) — both paths

```bash
# Docker Engine + Compose plugin
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER" && newgrp docker   # run docker without sudo

# (optional, for GPU) NVIDIA Container Toolkit — only if you want the full LLM path
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

# Port 8080 must be free for the OpenShell gateway (Path A).
```

You also need an **NVIDIA API key** (`nvapi-...`). For driving the agent you also
need an **AI provider key** (Anthropic or OpenAI).

```bash
cp ../.env.example ../.env          # then edit ../.env and set NVIDIA_API_KEY
```

---

## Step 1 — Start the vulnerable Memory MCP server (OUTSIDE the sandbox) — both paths

> Compose file: **`docker-compose.memory-mcp.yml`** (required in both paths).

```bash
cd openshell
docker compose --env-file ../.env -f docker-compose.memory-mcp.yml up -d --build

# verify it is listening
curl -sf http://localhost:8999/mcp -H 'accept: text/event-stream' -o /dev/null && echo "memory-mcp up"
docker logs memory-mcp --tail 20
```

The server binds `0.0.0.0:8999`. The host directory `openshell/host_breakout_marker/`
is mounted at `/workspace/mnt` inside the server — this is our "outside the
sandbox" filesystem.

---

# Path A — Full OpenShell + OpenClaw

Use this path for the faithful threat model. Compose files: the **gateway**
compose **plus** the already-running **memory-mcp** compose from Step 1. (Do not
use `docker-compose.override.yml` here.)

## A.2 — Start the OpenShell gateway

> Compose file: **`OpenShell/deploy/docker/docker-compose.yml`** (from the NVIDIA repo).

Follow the official tutorial
(<https://docs.nvidia.com/openshell/v0.0.116/get-started/tutorials/docker-compose>):

```bash
# Clone OpenShell and start the gateway
git clone https://github.com/NVIDIA/OpenShell.git
cd OpenShell/deploy/docker

# Linux: make host.*.internal resolvable from containers — add under the
# `gateway` service in docker-compose.yml:
#   extra_hosts:
#     - "host.docker.internal:host-gateway"
#     - "host.openshell.internal:host-gateway"

docker compose up -d
curl -sf http://localhost:8080/healthz && echo "gateway healthy"
```

Install and register the CLI:

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
openshell gateway add http://localhost:8080 --name openshell-docker
openshell status        # expect: Status: Connected
```

Configure an AI provider (example: Anthropic):

```bash
ANTHROPIC_API_KEY=sk-ant-... \
  openshell provider create --name anthropic --type anthropic --from-existing
```

## A.3 — Create the OpenClaw sandbox

OpenClaw runs inside OpenShell through **NemoClaw**. Follow the NemoClaw
Quickstart (<https://docs.nvidia.com/nemoclaw/latest/get-started/quickstart/>)
to create an OpenClaw sandbox with managed inference.

> To validate mechanics without OpenClaw, you can use a stock Claude Code sandbox
> created **by the gateway** (still Path A, still real isolation):
>
> ```bash
> docker pull ghcr.io/nvidia/openshell-community/sandboxes/base:latest
> openshell sandbox create -- claude
> openshell sandbox list         # wait for phase: Ready
> openshell sandbox connect <sandbox-name>
> ```

## A.4 — Install the memory-recall skill INSIDE the sandbox

From your connected sandbox session, install the skill (or bind it via your
NemoClaw sandbox spec):

```bash
mkdir -p ~/.claude/skills/memory-recall
# paste SKILL.md and recall.py from openshell/skills/memory-recall/ into that dir
pip install fastmcp
export MEMORY_MCP_URL=http://host.openshell.internal:8999/mcp

# sanity check the skill can reach the server outside the sandbox
python ~/.claude/skills/memory-recall/recall.py \
  --tool memory_agent --query "hello" --user-id ruth
```

## A.5 — Trigger the breakout (Path A)

Ask OpenClaw to use the memory skill with a crafted value, or run an exploit
client from inside the sandbox:

```bash
# inside the sandbox
export SERVER_URL=http://host.openshell.internal:8999/mcp
python /path/to/exploits/01-CWE-78-mcp-user-id-command-injection/exploit_client.py --mode exploit
# or the eval variant:
python /path/to/exploits/02-CWE-94-mcp-memory-eval-code-injection/exploit_client.py --mode exploit
```

## A.6 — Confirm the breakout on the HOST (Path A)

In the real topology the sandbox **cannot** see the host filesystem, so verify
the marker on the host, not in the sandbox:

```bash
# on the host, in this repo
ls -l openshell/host_breakout_marker/
cat openshell/host_breakout_marker/MARKINJECT_BREAKOUT_*
```

Seeing `MARKINJECT_BREAKOUT_<nonce>` on the host — written because of a tool call
the agent made from inside the sandbox — is proof of the cross-boundary breakout.

---

# Path B — Quick dev-loop (no gateway)

Use this to iterate fast on the skill/exploit. Compose files: the **memory-mcp**
compose **combined with** the **override** compose. (Do not start the gateway
here.)

> Compose files: **`docker-compose.memory-mcp.yml` + `docker-compose.override.yml`**

## B.2 — Start server + Claude Code container together

```bash
cd openshell
docker compose --env-file ../.env \
  -f docker-compose.memory-mcp.yml \
  -f docker-compose.override.yml \
  up -d --build

docker compose -f docker-compose.memory-mcp.yml -f docker-compose.override.yml ps
```

This starts `memory-mcp` (:8999, outside) and `claude-sandbox` (the stand-in
sandbox) on a shared network. The skill is mounted at
`~/.claude/skills/memory-recall` inside `claude-sandbox`, and `MEMORY_MCP_URL` /
`SERVER_URL` already point at `http://memory-mcp:8999/mcp`.

## B.3 — Trigger the breakout from "inside the sandbox"

```bash
docker exec -it claude-sandbox bash

# inside claude-sandbox — env vars are pre-wired:
python /opt/exploits/01-CWE-78-mcp-user-id-command-injection/exploit_client.py --mode exploit
python /opt/exploits/02-CWE-94-mcp-memory-eval-code-injection/exploit_client.py --mode exploit

# functional tests (should PASS = no injection on benign input):
bash /opt/exploits/01-CWE-78-mcp-user-id-command-injection/functional-test.sh
bash /opt/exploits/02-CWE-94-mcp-memory-eval-code-injection/functional-test.sh
```

Because the override mounts `host_breakout_marker/` read-only into the container
at `/workspace/mnt`, the in-sandbox `exploit_client.py` can verify the marker
directly and print `EXPLOIT VERIFIED`.

## B.4 — Confirm on the HOST too

```bash
# on the host
cat openshell/host_breakout_marker/MARKINJECT_BREAKOUT_*
```

> Caveat: `claude-sandbox` is **not** a real OpenShell isolation boundary, and it
> is given a convenience read-only view of the host marker dir. For a rigorous
> demonstration of escaping isolation, use Path A.

## B.5 — (Optional) drive it through Claude Code

If you set `ANTHROPIC_API_KEY` in `../.env`, you can run the `claude` CLI inside
`claude-sandbox` and prompt it to use the `memory-recall` skill, e.g. "recall my
memories for user `ruth; <cmd>`", to show the agent itself triggering the sink.

---

## Prove code lands BOTH inside and outside (either path)

1. Have the agent write a marker to a sandbox-local path (inside) via a normal
   shell command — trivially available to the agent.
2. Trigger the exploit so the server writes a marker to `host_breakout_marker/`
   (outside).

Two markers with the same nonce — one inside the sandbox, one on the host —
demonstrate code execution landing on both sides of the boundary.

---

## Teardown

```bash
cd openshell

# Path B:
docker compose -f docker-compose.memory-mcp.yml -f docker-compose.override.yml down

# Path A (memory-mcp):
docker compose -f docker-compose.memory-mcp.yml down
# Path A (gateway):
cd /path/to/OpenShell/deploy/docker && docker compose down
```

## Remediation

See each `exploits/<variant>/exploit.md` for the correct fix. In short: never
interpolate client-supplied strings into shell commands and never `eval()` data
received from a network client.
