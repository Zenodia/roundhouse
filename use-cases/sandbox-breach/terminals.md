<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Watching the breakout happen — terminal layout

`run.sh` starts and drives the demo; this is what to have open **while** it runs so the escape is
something you watch happen live, not something you take on faith from a script's exit code. Six
terminals, six different vantage points on the same three seconds.

| # | Command | What you're watching |
| --- | --- | --- |
| 1 | `docker logs -f memory-mcp` | The vulnerable server's own stdout — you'll see the `os.system`/`eval` sink fire the instant a poisoned tool call lands, before any LLM call even starts. |
| 2 | `watch -n1 'ls -la ~/roundhouse/labs/breakout-lab/openshell/host_breakout_marker/'` | The **host** filesystem, outside the sandbox entirely. A `MARKINJECT_BREAKOUT_*` file appearing here, live, is the escape — not a log line claiming one happened. |
| 3 | `docker logs -f $(docker ps --format '{{.Names}}' \| grep supervisor \| grep sandbox-breach-demo)` | OpenShell's own network-policy audit trail (the sandbox's *supervisor* container, not the gateway's systemd journal — verified empirically that `journalctl --user -fu openshell-gateway` does not carry these OCSF lines). `ALLOWED .../8080 [policy:_provider_roundhouse]` next to `DENIED .../api.anthropic.com [reason:transparent_tcp_policy_denied]` proves the sandbox's egress really is locked down to what the demo declared, not merely configured that way. |
| 4 | `curl -s http://127.0.0.1:8080/v1/metrics -H "x-roundhouse-key: $(python3 -c "import json;print(json.load(open('keys.local.json'))['__admin__'])")" \| python3 -m json.tool` (re-run after each step) | roundhouse's own durable session/metrics log — which model actually answered, how many tokens, what it cost. The independent second vantage point on the same reasoning-LLM traffic OpenShell's log (row 3) already shows crossing the boundary. |
| 5 | `openshell sandbox exec -n sandbox-breach-demo -- sh -c 'ls -la /sandbox/*.jsonl 2>/dev/null && tail -f /sandbox/nemo-relay-events-*.jsonl'` (after a `./run.sh relay` call) | NeMo Relay's ATOF JSONL stream — the agent's own lifecycle events (`session.start`, tool calls, hook events) as Relay's gateway saw them, independent of both roundhouse's and OpenShell's logs. |
| 6 | The `./run.sh` invocations themselves | What you're actually driving: `up`, `sandbox`, `exploit-78`, `exploit-94`, `relay "<prompt>"`. |

## Suggested run order, once all six are open

```bash
./run.sh up                              # rows 1 stabilizes; roundhouse/Dynamo come up quiet
./run.sh sandbox                         # sandbox created; row 3 starts showing ALLOWED/DENIED pairs
./run.sh exploit-78                      # watch rows 1 and 2 fire together
./run.sh exploit-94                      # same, other sink
./run.sh relay "reply with exactly: OK"  # watch rows 3, 4, and 5 agree on one turn from three angles
```

Row 2 is the one that matters most: everything else is corroborating evidence for *why* that file
appeared, but the file appearing on the host is the claim itself.
