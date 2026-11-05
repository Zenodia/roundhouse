<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Decision memory: keeping agents aligned as business decisions change

**The problem.** Decisions are made in rooms — finance reviews, risk
committees, legal sign-offs — and only their final outcome reaches the
digital world, usually as a closed ticket. The reasoning, the constraints
and the fact that the decision *replaced an earlier one* stay in the
meeting. An agent working in the codebase then does ordinary work against
rules it was never told, and does it confidently.

**The experiment.** A small payments service (Ledgerline) whose business
rules live outside its code; ten ordinary engineering tasks that silently
depend on those rules; three conditions that differ only in which decision
context the agent sees; objective checks that separate "did the work" from
"got the rule right". Plus the capture step that turns a meeting note into
a decision record a human approves, and the operate loop that retests the
affected tasks when a decision changes.

This follows the NeMo agentic-memory team's methodology (no-memory /
oracle / memory-enabled, staleness and compositional stressors, failure
signatures), with a decision record as the memory write and a file-system
memory as the simplest possible store.

## Layout

```
ledgerline/                 the fixture repo (Python, no deps) — the "customer's" code
  ledgerline/               package: models, payments, storage, reports, retention, money
  ledgerline/decisions.py   reads/validates decisions/, `param()` for policy tests, `accept`
  decisions/                DR-0001..DR-0010: the digital copies of the decisions, with history
  meeting-notes/            the physical-world artifacts the records were captured from
  docs/SERVICE.md           the engineering reference (deliberately behind in places)
  tests/                    baseline suite (passes at the fixture commit)
  .github/CODEOWNERS        decisions/ is approved by decision owners, not engineering
tasks/T01..T10/             task.yaml (the ticket the agent sees) + test_task.py (hidden)
reference/ledgerline/       a human solution to all ten tasks; proves the checks are satisfiable
runner/                     conditions, patch protocol, clients, run_eval, report, affected
capture/draft_record.py     meeting note -> proposed record (model or stub)
roundhouse/                 catalog, control plane, env for running through Roundhouse
```

## The decisions, and why the code does not reveal them

| Record | Rule | State of the code | Ladder rung |
|---|---|---|---|
| DR-0004 (supersedes DR-0002) | Revenue is recognized at **capture**, not settlement | `reports.BILLABLE_STATE` still says `SETTLED` — by decision, for a Q4 comparison | half reflected, revised |
| DR-0005 | Refunds > 30 days after capture need manager approval; cap = captured − prior refunds | no refund code exists | not reflected |
| DR-0003 | Last four digits only; the PAN never persisted or shown, debug views included | ingest masks the audit copy, but `Payment.gateway_payload` still holds the PAN | half reflected |
| DR-0007 (supersedes DR-0001) | Raw events kept **90** days; aggregates kept forever | `retention.RAW_EVENT_RETENTION_DAYS = 30` | revised |
| DR-0010 (supersedes DR-0006) | EU → `eu`, **UK → `uk`**, others → `default` | `payments.EU_REGIONS = {"EU", "UK"}` | revised |
| DR-0008 | *Rejected*: store the PAN for chargebacks | — | distractor |
| DR-0009 | *Proposed*: go back to settlement recognition | — | distractor (must not be acted on) |

None of the rules can be derived from the code; two of them contradict
it. Every task is ordinary work (add a report, add an export, add a purge
job); none says "apply decision X". That is the memory–action coupling
the memory team's benchmark principles ask for: the decision changes what
the agent *writes*, not what it recalls.

## Conditions and what each isolates

| Condition | Decision context in the prompt | Isolates |
|---|---|---|
| `no_memory` | none | the floor: a capable agent that was not in the room |
| `oracle` | every accepted record, in full | the ceiling: do the records suffice when retrieval is perfect |
| `memory` | records whose `scope` globs cover a file the task touches | retrieval: T10 reads retention from `reports.py`, which no retention record scopes — a miss by construction |
| `--records all` | adds superseded, proposed and rejected records | the confusable stressor: the agent must read `status` |

Three checks per attempt, all objective:

1. **baseline** — the repo's own tests still pass (nothing broken);
2. **functional** — the task's hidden tests not marked `policy` (the feature works);
3. **policy** — the hidden tests marked `@pytest.mark.policy`, which read their expected values from `decisions/` via `ledgerline.decisions.param(...)`.

A **stale action** is functional-pass + policy-fail: the agent did the
work and got the business rule wrong. That row is the demonstration.
Because policy tests read the current records, a changed decision changes
the tests and nothing else.

## Part A: run it

### A0. Environment (CPU is enough for everything but the local model tier)

```bash
cd roundhouse/experiments/decision-memory
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd ledgerline && python -m pytest -q)                       # 9 baseline tests
(cd ledgerline && python -m ledgerline.decisions validate)   # 10 records, 5 accepted, 0 problems
```

### A1. Dry run with no model — the checks are satisfiable and the floor is real

```bash
python -m runner.run_eval --client stub:reference --conditions oracle --out runs/dry-ref
python -m runner.run_eval --client stub:none      --conditions no_memory --out runs/dry-none
python -m runner.report runs/dry-ref/results.jsonl runs/dry-none/results.jsonl
```

Expected: `stub:reference` is 10/10 success under every condition (the
reference solution passes all 47 tests), `stub:none` is 0/10. Artifacts
per attempt (`instructions.md`, `input.md`, `reply.md`, `score.json`) land
under `runs/<name>/attempts/`; read one `instructions.md` to see exactly
what the agent sees.

### A2. Frontier tier (the headline)

The frontier model is NVIDIA's inference API, which is OpenAI
chat-completions compatible. Pick a model that is strong at code but not
the premium tier: **Claude Sonnet 5** is the default here; Opus 5.5
(`azure/anthropic/claude-opus-5-5`) for one final headline run at most.
List what your key reaches and fix the id in one place:

```bash
export INFERENCE_API_KEY=...
curl -s -H "Authorization: Bearer $INFERENCE_API_KEY" https://inference-api.nvidia.com/v1/models | python3 -m json.tool | grep '"id"'
export FRONTIER_MODEL=aws/anthropic/bedrock-claude-sonnet-5     # adjust to the id the list shows
```

**Direct (simplest; correctness and stale-action rates):**

```bash
python -m runner.run_eval --client chat:https://inference-api.nvidia.com --model $FRONTIER_MODEL \
    --tier frontier --repeats 3 --out runs/frontier
python -m runner.report runs/frontier/results.jsonl
```

**Through Roundhouse (adds the turn log, pricing and the cache view):**
Roundhouse has no chat-completions client, so the catalog reaches the
Anthropic models on this endpoint through the Messages dialect, the way
the cache-aware-routing branch did. Put `$FRONTIER_MODEL` into
`roundhouse/catalog.json` (`models[0].model`), then:

```bash
cargo build -p roundhouse-server --bin roundhouse --release   # once, at the repo root
python roundhouse/mint_keys.py                                 # writes roundhouse/keys.local.json
source roundhouse/env.sh && ../../target/release/roundhouse &
export ROUNDHOUSE_API_KEY=$(python -c "import json;print(json.load(open('roundhouse/keys.local.json'))['frontier/runner'])")
python -m runner.run_eval --client responses:http://127.0.0.1:8080 --tier frontier --repeats 3 --out runs/frontier-rh
```

Repeats matter: three per cell gives 30 attempts per condition, enough to
see a stale-action rate rather than a coin flip. Budget: ~90 calls of
~5K input / ~1.5K output tokens per full run.

### A3. Local tier (cost column)

Serve Qwen3-8B + Qwen3-14B with `../cross-model-kv/serving/serve_pair.sh`
(the two-worker Dynamo recipe from the KV experiment), then either go
straight to Dynamo:

```bash
python -m runner.run_eval --client chat:http://127.0.0.1:8000 --model Qwen/Qwen3-14B --tier local --out runs/local-direct
```

or through Roundhouse so the cached-token column is Roundhouse's own KV
view (start Roundhouse with `ROUNDHOUSE_LOCAL_ENDPOINT=http://127.0.0.1:8000`
set before `source roundhouse/env.sh`, and use the `local/runner` key):

```bash
export ROUNDHOUSE_API_KEY=$(python -c "import json;print(json.load(open('roundhouse/keys.local.json'))['local/runner'])")
python -m runner.run_eval --client responses:http://127.0.0.1:8080 --tier local --out runs/local
```

The prompt's stable part (service doc, core modules, decision records) is
sent as `instructions`, identical across tasks within a condition, so
from the second task on it is a cache hit; `cached_tokens` in the report
shows it. Adding the oracle's five records costs ~1,500 tokens of prefix
once per condition.

### A4. The confusable stressor

```bash
python -m runner.run_eval --client responses:http://127.0.0.1:8080 --tier frontier --records all --out runs/frontier-all
```

Now DR-0009 (proposed: back to settlement) and DR-0008 (rejected: store
the PAN) are in the context beside the accepted records. An agent that
acts on a proposed record fails T01/T06/T10's policy tests.

### Reading the report

```
tier     model   condition  n  success  func ok  stale  in_tok  cached%
frontier ...     no_memory 30    x%      ~high    HIGH
frontier ...     oracle    30    high    ~high    low
frontier ...     memory    30    mid     ~high    T10 misses
```

The memory team's failure signatures apply directly: `oracle ≈ no_memory`
means the tasks do not depend on the records (fix the tasks);
`oracle ≫ memory` means retrieval is the bottleneck (scope globs are too
narrow — T10 is the planted case); `oracle ≫ no_memory` is the result.

## Part B: capture — fixing the pipeline, not just reading from it

A record exists because someone wrote it down after the meeting. The
materials in the memory program cover storing and retrieving; this is the
step before that.

```bash
# 1. the meeting happened; the note is the physical-world artifact
cat ledgerline/meeting-notes/2026-11-04-risk-committee.md       # risk moves the refund threshold 30 -> 14 days

# 2. draft a record from it (a model, or the stub for this note)
python -m capture.draft_record ledgerline/meeting-notes/2026-11-04-risk-committee.md --client stub
#    -> ledgerline/decisions/DR-0011-...md with status: proposed, supersedes: DR-0005

# 3. open a PR; CODEOWNERS routes decisions/ to the decision owners; the owner edits and accepts
(cd ledgerline && python -m ledgerline.decisions accept DR-0011 && python -m ledgerline.decisions validate)
#    DR-0011 -> accepted, DR-0005 -> superseded (superseded_by: DR-0011); param() now returns 14

# 4. the operate loop: which tasks does this change affect, and do they still pass?
python -m runner.affected DR-0011
python -m runner.affected DR-0011 --run --client responses:http://127.0.0.1:8080 --tier frontier --out runs/retest-DR-0011
```

What the demo shows: before the accept, a refund implementation written
for 30 days passes; after it, the same code fails
`test_late_refund_needs_manager_approval` without anyone editing a test.
The decision moved; the checks moved with it; the agent is re-run on the
affected tasks with the new record in its context. In production, step 4
is a CI job on `decisions/**`.

`accept` refuses to leave the folder inconsistent (two accepted records
defining one parameter), and `validate` runs in CI. A record informs; it
grants no permission and does not resolve a conflict between two accepted
records by itself.

## Compute

| Step | Needs | Time |
|---|---|---|
| A0–A1, Part B with the stub | a laptop / WSL, Python 3.10+ | minutes |
| A2 frontier tier, 10 tasks × 3 conditions × 3 repeats | API access; Roundhouse on any machine | ~90 calls of ~5K in / ~1.5K out tokens; 20–40 min wall |
| A3 local tier | **1× H100 80 GB** for Qwen3-8B + 14B on Dynamo (the same serving setup as the KV experiment) | ~1 h for 90 attempts at 14B |
| A4 confusable | same as A2 | same |

Nothing here trains anything. The GPU is needed only for the local tier.

## Design notes and open questions

- **Patch protocol, no tools.** The agent gets the files it needs and returns
  whole files. This works identically on frontier and local models, keeps a
  large shared prefix for the cache measurement, and sidesteps the fact that
  a tool-declaring turn never routes to a local worker in Roundhouse. A
  tool-using mode (read/search decisions on demand, the memory team's
  "memory-enabled, agent reads naturally") is the natural next ablation.
- **Retrieval rule.** Scope globs are the simplest rule that can miss. T10 is
  the planted miss. Widening DR-0007's scope to `ledgerline/reports.py`
  fixes it; whether that belongs to the record author or the retriever is a
  question for the pilot team.
- **Who owns the records.** CODEOWNERS makes the owner's approval structural;
  for this experiment it names `@Zenodia`. In a pilot the first thing to
  confirm is that finance/risk/legal will actually approve PRs; if not, the
  demo degenerates to RAG over a wiki.
- **Weight-space programming** is deliberately not here: decisions change,
  need provenance and rollback, and must never be permission. If the eval
  shows a systematic small-model behaviour (ignoring `status`, say), that
  is a candidate for a weight-level fix — and any edit to Qwen3-8B
  invalidates the KV mapper fitted on it in `../cross-model-kv`.
