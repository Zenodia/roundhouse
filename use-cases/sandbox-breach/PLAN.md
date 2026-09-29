<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PLAN — sandbox-breach

Phased implementation record. Phases 0–6 are done and verified with real captured output (see
`README.md`); this file tracks what closed each phase and what's left.

## Phase 0 — Vulnerable target, standalone

Bring up breakout-lab's memory-mcp server alone, outside any sandbox/agent plumbing, and confirm
both CWEs fire against a bare server per its own `EXPLOIT_RUN_INSTRUCTIONS.md`, before touching
anything roundhouse- or OpenShell-specific.

**Closed.** Two real bugs found and fixed in breakout-lab's own fixtures (missing `langchain`
package pin; EOL embedding model) — see README §1 for both, with before/after regression evidence.

## Phase 1 — roundhouse, both legs

New `catalog.json` (local Qwen via Dynamo, discovered at runtime + frontier
`azure/anthropic/claude-opus-5`) and `control-plane.json` (two projects/keys, one per leg). Confirm
both dispatch to real backends, not stubs.

**Closed.** One real gap found: this build has no dispatch client for `openai_chat_completions`
(the dialect the user's sample code used) — worked around by using `anthropic_messages` against
the same endpoint, verified to actually answer under that dialect. See README §2.

## Phase 2 — Real OpenShell gateway + sandbox

Install OpenShell, stand up a real sandbox, install the memory-recall skill inside it, and prove
network-policy isolation is real (not merely configured) by having the vulnerable-server-facing and
roundhouse-facing traffic both explicitly need policy rules — no rule, no reach.

**Closed.** One generic environment bug (bad embedded image policy) and two OpenShell-specific
gotchas (provider credential-binding boundary; binary-allowlist symlink resolution) found and
fixed. See README §3.

## Phase 3 — Both CWEs, from inside the real sandbox

Trigger both exploits via `openshell sandbox exec` running the actual memory-recall skill, with the
impact marker landing on the **host** filesystem as the only acceptable proof.

**Closed.** Both nonces landed; see README §4.

## Phase 4 — roundhouse's transparent hookup, corroborated

Confirm the sandboxed agent's own reasoning-LLM traffic really is being transparently redirected
through roundhouse — not just configured to be — using an independent log (OpenShell's own network
audit trail) that agrees with roundhouse's own session/metrics log.

**Closed.** See README §5.

## Phase 5 — NeMo Relay chained in front of roundhouse

Wrap the sandboxed `claude` invocation with `nemo-relay run`, forwarding upstream to roundhouse via
`--anthropic-base-url`, and get a real completion to cross all three hops.

**Closed** for the chain itself; the persistent ATOF JSONL export needed one more round (schema
migrated from v1 to v4 between what NeMo Relay's own hosted docs show and what CLI 0.9.3 actually
accepts — discovered empirically via the CLI's own validation error text, not documented anywhere
found). See README §6.

## Phase 6 — NemoClaw / real OpenClaw

Attempt the real OpenClaw-managed sandbox path (as opposed to the OpenShell-gateway-created stock
Claude Code sandbox used in Phases 2–5), which the user's original ask specifically named.

**Blocked, not closed.** NemoClaw's installer pins OpenShell `0.0.116`; this demo's already-running,
already-verified gateway is `0.1.2`. The installer detected the mismatch itself and refused to
proceed without either downgrading the running gateway or removing the upstream OpenShell package —
both of which risk breaking every already-verified phase above. Declined to force it without
explicit sign-off; see `GAPS.md` for the exact remedy and its cost. One real, if minor, incident
along the way: the installer's OpenShell 0.0.116 binaries landed in `~/.local/bin` ahead of
`/usr/bin` on `$PATH`, and the version-mismatched CLI briefly broke `openshell sandbox list`
against the still-running 0.1.2 gateway (protobuf decode error) until the shadowing binaries were
removed. Fixed immediately; every subsequent phase re-verified clean.

## Phase 7 — Orchestration and docs

`run.sh` (idempotent `up`/`sandbox`/`exploit-78`/`exploit-94`/`relay`/`status`/`down`) and
`terminals.md` (the six-vantage-point watch layout), both smoke-tested end to end for real.

**Closed.**

## What's left

- Real NemoClaw/OpenClaw, if the version-pin conflict in Phase 6 gets resolved deliberately (see
  `GAPS.md`).
- roundhouse's own `roundhouse-mcp` control surface was never wired into this demo — it is a
  separate, unrelated MCP server (the agent's *own* routing-introspection tools) from
  breakout-lab's memory-recall MCP server (the exploit target); nothing in this use case needs it,
  but a future addition could let the agent narrow its own routing mid-session as part of the demo
  narrative.
