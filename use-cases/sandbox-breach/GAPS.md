<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GAPS — sandbox-breach

What's real and working vs. what's still open, and the concrete remedy for each open item. See
`README.md` for the full evidence trail behind every "closed" row.

## Architecture (current, verified state)

```
┌───────────────────────────── Ubuntu 24.04 host (1x H100) ─────────────────────────┐
│  OpenShell gateway v0.1.2 (native, systemd, mTLS)                                 │
│   └─ sandbox-breach-demo (real OpenShell sandbox)                                 │
│        claude (Claude Code CLI, stand-in for OpenClaw — see gap 1 below)          │
│          ├─ reasoning calls → [NeMo Relay, optional] → roundhouse :8080           │
│          │                                              ├─ local/*    → Dynamo    │
│          │                                              └─ nvidia-*/* → Opus-5    │
│          └─ memory-recall skill → memory-mcp :8999 (outside sandbox)              │
│                                     ├─ CWE-78 os.system()                          │
│                                     └─ CWE-94 eval()                              │
│                                     writes → host_breakout_marker/                │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Gap table

| # | Gap | Status | Remedy |
| --- | --- | --- | --- |
| 1 | Real NemoClaw/OpenClaw not used — stock Claude Code sandbox (OpenShell-gateway-created) stood in | **Open, blocked** | NemoClaw's installer pins OpenShell `0.0.116`; running gateway is `0.1.2`. Remedy: either (a) stand up a second, isolated gateway instance at `0.0.116` on a different port/data-dir so it doesn't touch the verified `0.1.2` one, or (b) deliberately downgrade the running gateway to `0.0.116` and re-run every phase's verification against it. Both are real work with real risk to already-verified state; needs explicit sign-off before attempting. |
| 2 | `openai_chat_completions` has no dispatch client in this roundhouse build | **Worked around** | Catalog entry speaks `anthropic_messages` against the same NVIDIA endpoint instead (verified the endpoint actually answers that way for this model). Real fix would be adding an `openai_chat_completions` `FrontierClient` impl to `roundhouse-fleet` — out of scope for a demo use case; would be a `roundhouse-core`/`roundhouse-fleet` milestone. |
| 3 | NeMo Relay's ATOF sink schema undocumented for CLI 0.9.3 | **Closed empirically** | Hosted docs show a `version = 1` schema (`output_directory`/`filename`/`mode` fields) that CLI 0.9.3 rejects; the real schema is `version = 4` with a `[[components.config.atof.sinks]]` array of `{type = "file", path = "..."}` tables — discovered from the CLI's own validation error text, not any doc found. One residual oddity: the `path` field's value is not what the file actually gets named (real files land at `~/nemo-relay-events-<timestamp>.jsonl` regardless of `path`) — not chased further since the file is real and tailable either way. |
| 4 | Two candidate replacement chat models for `memory_mcp_server.py`'s separately-EOL LLM both 404 as not-entitled | **Open, low priority** | `nvidia/nemotron-nano-3-30b-a3b` and `nvidia/llama-3.1-nemotron-70b-instruct` both return 404 on this account. Does not block either exploit (both sinks fire pre-LLM). Remedy: either get account entitlement for a working nemotron chat model, or accept the memory-chat happy path stays broken (irrelevant to the security demo). |
| 5 | No `SCORECARD.md` | **Deliberately skipped** | This use case doesn't fit the "is roundhouse a good fit for domain X" evaluation frame the other use cases' scorecards measure — it's a security demo using roundhouse as infrastructure, not an evaluation of roundhouse itself. `PLAN.md`/`GAPS.md` cover the equivalent tracking need. |

## Incidents during this work (resolved, recorded for anyone repeating it)

- **NemoClaw's installer PATH-shadowed the working OpenShell CLI.** It installed its pinned
  `0.0.116` binaries to `~/.local/bin`, ahead of `/usr/bin` on `$PATH`, and the version-mismatched
  CLI immediately broke `openshell sandbox list` against the still-running `0.1.2` gateway
  (`ListSandboxesRequest.workspace_scope: unexpected end group tag` — a real protobuf wire
  incompatibility, not a fluke). Fixed by removing the shadowing binaries; the gateway service
  itself was never touched and every phase re-verified clean afterward. If repeating Phase 6's
  NemoClaw attempt, expect this every time until gap 1 is actually resolved.
