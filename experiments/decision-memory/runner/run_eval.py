#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run tasks x conditions, score each attempt, append rows to results.jsonl.

    # dry run on a CPU, no model: the reference solution must pass everything
    python -m runner.run_eval --client stub:reference --conditions oracle --out runs/dry

    # frontier through Roundhouse (ROUNDHOUSE_API_KEY = a turn key)
    python -m runner.run_eval --client responses:http://127.0.0.1:8080 --tier frontier --out runs/frontier

    # local Qwen3-14B straight from Dynamo's frontend
    python -m runner.run_eval --client chat:http://127.0.0.1:8000 --model Qwen/Qwen3-14B --tier local --out runs/local

Each attempt: fresh copy of the fixture repo -> prompt -> apply returned
files -> three pytest runs (baseline suite; the task's functional tests;
the task's policy tests) -> forbidden-pattern scan of the applied files.

A *stale action* is the row that matters: functional tests pass and a
policy test fails. The agent did the work and got the business rule
wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .clients import Reply, make_client
from .conditions import CONDITIONS, records_for, render_context
from .protocol import build_input, build_instructions, parse_files, safe_to_write
from .tasks import FIXTURE_REPO, REFERENCE_DIR, Task, load_tasks

SUMMARY_RE = re.compile(r"(?:(\d+) passed)?(?:, )?(?:(\d+) failed)?(?:, )?(?:(\d+) error)?")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--client", required=True, help="responses:<url> | chat:<url> | stub:<reference|none>")
    p.add_argument("--model", default=None, help="model name for chat clients (or to pin one through Roundhouse)")
    p.add_argument("--tier", default="unspecified", help="label: frontier | local | ...")
    p.add_argument("--conditions", default=",".join(CONDITIONS))
    p.add_argument("--tasks", default=None, help="comma list of task ids or prefixes (default all)")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--records", default="current", choices=["current", "all"], help="all = include superseded/proposed/rejected too")
    p.add_argument("--decisions", default=None, help="decisions folder (default: the fixture repo's)")
    p.add_argument("--out", required=True, help="output dir; results.jsonl and per-attempt artifacts")
    p.add_argument("--keep-repos", action="store_true", help="keep each attempt's repo copy under --out")
    p.add_argument("--pytest", default=sys.executable + " -m pytest", help="pytest command")
    return p.parse_args()


def run_pytest(repo: Path, args: list[str], pytest_cmd: str, timeout: int = 300) -> dict:
    cmd = pytest_cmd.split() + ["-q", "-p", "no:cacheprovider", "--no-header", "-rN"] + args
    try:
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=timeout, env={**os.environ, "PYTHONPATH": str(repo)})
        out = proc.stdout
        code = proc.returncode
    except subprocess.TimeoutExpired:
        return {"passed": 0, "failed": 0, "errors": 1, "ok": False, "tail": "timeout"}
    # pytest's summary is the last stdout line of the form "3 passed, 1 failed in 0.2s".
    summary = [l for l in out.strip().splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    passed = failed = errors = 0
    if summary:
        m = re.search(r"(\d+) passed", summary[-1])
        passed = int(m.group(1)) if m else 0
        m = re.search(r"(\d+) failed", summary[-1])
        failed = int(m.group(1)) if m else 0
        m = re.search(r"(\d+) error", summary[-1])
        errors = int(m.group(1)) if m else 0
    out = out + proc.stderr
    # Exit code 5 = no tests collected (e.g. nothing marked policy): treat as ok with zero.
    ok = code == 0 or (code == 5 and failed == 0)
    return {"passed": passed, "failed": failed, "errors": errors, "ok": ok, "tail": out[-1500:]}


def score(repo: Path, task: Task, applied: dict[str, str], pytest_cmd: str) -> dict:
    # The hidden task tests live beside the repo's own tests for the run.
    tests_dir = repo / "tests" / "tasks"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("")
    test_path = tests_dir / f"test_{task.id.replace('-', '_')}.py"
    shutil.copy(task.test_file, test_path)
    baseline = run_pytest(repo, ["tests", "--ignore", str(tests_dir)], pytest_cmd)
    functional = run_pytest(repo, [str(test_path), "-m", "not policy"], pytest_cmd)
    policy = run_pytest(repo, [str(test_path), "-m", "policy"], pytest_cmd)
    forbidden_hits = []
    for pat in task.forbidden_patterns:
        rx = re.compile(pat)
        for path, content in applied.items():
            if rx.search(content):
                forbidden_hits.append({"pattern": pat, "file": path})
    functional_ok = functional["ok"] and functional["failed"] == 0 and functional["errors"] == 0 and functional["passed"] > 0
    policy_ok = policy["ok"] and policy["failed"] == 0 and policy["errors"] == 0 and not forbidden_hits
    return {
        "baseline": {k: baseline[k] for k in ("passed", "failed", "errors", "ok")},
        "functional": {k: functional[k] for k in ("passed", "failed", "errors", "ok")},
        "policy": {k: policy[k] for k in ("passed", "failed", "errors", "ok")},
        "forbidden_hits": forbidden_hits,
        "baseline_ok": baseline["ok"],
        "functional_ok": functional_ok,
        "policy_ok": policy_ok,
        "success": baseline["ok"] and functional_ok and policy_ok,
        "stale_action": functional_ok and not policy_ok,
        "tails": {"baseline": baseline["tail"], "functional": functional["tail"], "policy": policy["tail"]},
    }


def attempt(task: Task, condition: str, args, out_dir: Path, repeat: int) -> dict:
    decisions_root = Path(args.decisions) if args.decisions else FIXTURE_REPO / "decisions"
    records = records_for(condition, task, decisions_root, args.records)
    instructions = build_instructions(FIXTURE_REPO, render_context(records))
    user = build_input(FIXTURE_REPO, task)
    client = make_client(args.client, args.model, REFERENCE_DIR, task.files)
    cache_key = f"decision-memory/{condition}/{args.records}"
    reply: Reply = client.complete(instructions, user, cache_key)

    work = Path(tempfile.mkdtemp(prefix=f"{task.id}-{condition}-"))
    repo = work / "ledgerline"
    shutil.copytree(FIXTURE_REPO, repo, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
    files = parse_files(reply.text) if not reply.error else {}
    applied, refused = {}, []
    for path, content in files.items():
        if safe_to_write(path):
            (repo / path).parent.mkdir(parents=True, exist_ok=True)
            (repo / path).write_text(content, encoding="utf-8")
            applied[path] = content
        else:
            refused.append(path)
    result = score(repo, task, applied, args.pytest)

    art = out_dir / "attempts" / f"{task.id}__{condition}__{repeat}"
    art.mkdir(parents=True, exist_ok=True)
    (art / "instructions.md").write_text(instructions, encoding="utf-8")
    (art / "input.md").write_text(user, encoding="utf-8")
    (art / "reply.md").write_text(reply.text, encoding="utf-8")
    (art / "score.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.keep_repos:
        shutil.copytree(repo, art / "repo", dirs_exist_ok=True)
    shutil.rmtree(work, ignore_errors=True)

    row = {
        "task": task.id,
        "ladder": task.ladder,
        "decisions": task.decisions,
        "condition": condition,
        "records_mode": args.records,
        "records_shown": [r.id for r in records],
        "tier": args.tier,
        "client": args.client.split(":")[0],
        "model": reply.model or args.model,
        "repeat": repeat,
        "error": reply.error,
        "files_returned": list(files),
        "files_applied": list(applied),
        "files_refused": refused,
        "input_tokens": reply.input_tokens,
        "cached_tokens": reply.cached_tokens,
        "output_tokens": reply.output_tokens,
        "latency_s": round(reply.latency_s, 3),
        "instructions_chars": len(instructions),
        "input_chars": len(user),
        **{k: v for k, v in result.items() if k != "tails"},
        "artifacts": str(art),
    }
    return row


def main():
    args = parse()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c for c in args.conditions.split(",") if c]
    for c in conditions:
        if c not in CONDITIONS:
            raise SystemExit(f"unknown condition {c}")
    tasks = load_tasks(args.tasks.split(",") if args.tasks else None)
    results = out_dir / "results.jsonl"
    print(f"{len(tasks)} tasks x {len(conditions)} conditions x {args.repeats} repeats -> {results}")
    print(f"{'task':<24} {'cond':<10} {'base':>5} {'func':>7} {'policy':>7} {'stale':>5} {'in_tok':>7} {'cached':>7} {'s':>6}  model")
    for task in tasks:
        for condition in conditions:
            for r in range(args.repeats):
                t0 = time.time()
                row = attempt(task, condition, args, out_dir, r)
                with open(results, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                f_ = row["functional"]
                p_ = row["policy"]
                print(
                    f"{task.id:<24} {condition:<10} {'ok' if row['baseline_ok'] else 'FAIL':>5} "
                    f"{f_['passed']:>3}/{f_['passed'] + f_['failed'] + f_['errors']:<3} "
                    f"{p_['passed']:>3}/{p_['passed'] + p_['failed'] + p_['errors']:<3} "
                    f"{'YES' if row['stale_action'] else '-':>5} {row['input_tokens']:>7} {row['cached_tokens']:>7} {row['latency_s']:>6.1f}  {row['model'] or ''}"
                    + (f"  ERROR {row['error'][:80]}" if row["error"] else "")
                )
    print(f"\nwrote {results}; summarize with: python -m runner.report {results}")


if __name__ == "__main__":
    main()
