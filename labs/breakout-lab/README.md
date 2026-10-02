# OpenClaw Memory Breakout Lab

A self-contained, **deliberately vulnerable** lab that demonstrates an
[OpenClaw](https://github.com/NVIDIA/NemoClaw) agent — running inside an
[OpenShell](https://docs.nvidia.com/openshell/) sandbox — **breaking out of the
sandbox** by abusing a Memory MCP server that runs *outside* the sandbox.

It is built on top of the real
[`Zenodia/standalone_agent_memory`](https://github.com/Zenodia/standalone_agent_memory)
project (seed SHA `f228b091bc1fdc3838c2e63b0f3da5c8c38d7d37`) with two ground-truth
vulnerabilities injected, in the style of the `marinade` synthetic security
dataset (CWE-labeled, with working exploits, functional tests, and remediation).

> WARNING: This repository intentionally contains exploitable code. Run it only
> in a disposable VM / lab environment. Do not deploy it.

---

## Topology

```
OpenClaw (inside OpenShell sandbox)
   -> memory-recall Agent Skill  (openshell/skills/memory-recall/)
        -> MCP client (recall.py)
             --- MCP over http --->  Memory MCP server (OUTSIDE the sandbox)
                                        vulnerable sinks:
                                          - os.system(...)  [CWE-78]
                                          - eval(...)       [CWE-94]
                                     -> writes marker to the HOST filesystem
                                        = proof of code execution outside sandbox
```

See `SCENARIO_SETUP.md` for the full design rationale and threat model.

---

## What's in here

| Path | Purpose |
| --- | --- |
| `memory_mcp_server.py` | The MCP server **with both vulnerabilities already injected** (VULN-01 in `memory_agent`, VULN-02 in `fetch_memory_items`). |
| `MemoryManager.py`, `utils.py`, `test_mcp_client.py` | Unmodified seed app code. |
| `SCENARIO_SETUP.md` | Design decisions, topology, threat model, labels/references. |
| `exploits/01-CWE-78-mcp-user-id-command-injection/` | Command-injection exploit bundle (patch.diff, exploit.sh, exploit_client.py, functional-test.sh, exploit.md). |
| `exploits/02-CWE-94-mcp-memory-eval-code-injection/` | `eval()` code-injection exploit bundle. |
| `exploits/EXPLOIT_RUN_INSTRUCTIONS.md` | Standalone (non-OpenShell) run guide for the exploits. |
| `openshell/` | OpenShell + OpenClaw integration: compose files, the memory-recall Agent Skill, and the full end-to-end run guide. |
| `openshell/docker-compose.memory-mcp.yml` | **Always used** — runs the vulnerable server OUTSIDE the sandbox (`:8999`). |
| `openshell/docker-compose.override.yml` | **Path B only** — Claude Code container wired to the skill (dev-loop shortcut, no gateway). |
| `openshell/README.md` | **Start here for the full walk-through — explains which compose file to use when (Path A vs Path B).** |
| `UPSTREAM_README.md` | The original seed project README. |

---

## The two injected vulnerabilities

### VULN-01 — CWE-78 OS command injection (`memory_agent`)
A "per-user snapshot" feature shells the client-supplied `user_id`:
```python
snapshot_path = f"{snapshot_dir}/{user_id}.jsonl"
os.system(f"mkdir -p {snapshot_dir} && touch {snapshot_path}")
```
`user_id = "ruth; <cmd> #"` executes `<cmd>` on the server host (outside the sandbox).
Runs before any LLM call, so it fires even without valid model credentials.

### VULN-02 — CWE-94 code injection (`fetch_memory_items`)
A "power-user filter" feature `eval()`s a query that starts with `=`:
```python
if query.strip().startswith("="):
    result = eval(query.lstrip("="), {"memories": memories})
```
`query = "=[__import__('os').popen('<cmd>').read()]"` runs `<cmd>` in the server process.

Both are in different functions and coexist. Each has a nonce-based impact
marker and a negative control (see the `exploits/` bundles).

---

## Quick start

- **Full OpenShell + OpenClaw end-to-end (recommended):** follow
  [`openshell/README.md`](openshell/README.md).
- **Just the exploits against a bare server:** follow
  [`exploits/EXPLOIT_RUN_INSTRUCTIONS.md`](exploits/EXPLOIT_RUN_INSTRUCTIONS.md).

Minimal smoke run of the server outside the sandbox:

```bash
cd openshell
cp ../.env.example ../.env    # set NVIDIA_API_KEY
docker compose --env-file ../.env -f docker-compose.memory-mcp.yml up -d --build
# then run an exploit_client.py from exploits/ pointing SERVER_URL at :8999
```

---

## Restoring a clean (non-vulnerable) baseline

The injected lines are tagged with `# [VULN-01 ...]` / `# [VULN-02 ...]`.
To get the original upstream file:

```bash
git show <seed-sha>:memory_mcp_server.py > memory_mcp_server.py   # if history retained
# or apply the remediations described in each exploits/<variant>/exploit.md
```

## Credits / labels

- Seed app: `Zenodia/standalone_agent_memory`.
- Vulnerability classes: CWE-78, CWE-94; attack patterns CAPEC-136, CAPEC-242.
- Incident anchors: CVE-2014-6271 (Shellshock), CVE-2021-44228 (Log4Shell),
  CVE-2023-29374 (LangChain unsafe deserialization).
