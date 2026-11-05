#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize results.jsonl by (tier, model, condition): success, stale
actions, tokens, cache share, latency. Then the per-task grid.

    python -m runner.report runs/frontier/results.jsonl [more.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def pct(n, d):
    return f"{100 * n / d:5.1f}%" if d else "    -"


def main(paths):
    rows = load(paths)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["tier"], r.get("model") or "?", r["condition"], r.get("records_mode", "current"))].append(r)
    print(f"{'tier':<10} {'model':<28} {'condition':<10} {'recs':<8} {'n':>3} {'success':>8} {'func ok':>8} {'stale':>8} {'errors':>6} {'in_tok':>8} {'cached%':>8} {'out_tok':>8} {'lat_s':>6}")
    for key in sorted(groups):
        g = groups[key]
        n = len(g)
        succ = sum(r["success"] for r in g)
        func = sum(r["functional_ok"] for r in g)
        stale = sum(r["stale_action"] for r in g)
        errs = sum(1 for r in g if r.get("error"))
        in_tok = sum(r["input_tokens"] for r in g)
        cached = sum(r["cached_tokens"] for r in g)
        out_tok = sum(r["output_tokens"] for r in g)
        lat = sum(r["latency_s"] for r in g)
        tier, model, cond, recs = key
        print(f"{tier:<10} {model[:28]:<28} {cond:<10} {recs:<8} {n:>3} {pct(succ, n):>8} {pct(func, n):>8} {pct(stale, n):>8} {errs:>6} {in_tok // max(n, 1):>8} {pct(cached, in_tok):>8} {out_tok // max(n, 1):>8} {lat / max(n, 1):>6.1f}")

    print("\nPer task (S = success, F = functional only -> stale action, x = failed, E = error):")
    conds = sorted({r["condition"] for r in rows})
    tasks = sorted({r["task"] for r in rows})
    print(f"{'task':<24} {'ladder':<16} " + " ".join(f"{c:>10}" for c in conds))
    for t in tasks:
        cells = []
        ladder = next((r["ladder"] for r in rows if r["task"] == t), "")
        for c in conds:
            rs = [r for r in rows if r["task"] == t and r["condition"] == c]
            marks = "".join("E" if r.get("error") else "S" if r["success"] else "F" if r["stale_action"] else "x" for r in rs)
            cells.append(f"{marks:>10}")
        print(f"{t:<24} {ladder:<16} " + " ".join(cells))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
