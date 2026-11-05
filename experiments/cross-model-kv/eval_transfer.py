#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 3: retention. Score the same benchmark under the target's own
prefill, the mapped source cache, and the source alone, and report
retention = transfer / target (paper section 4.1), plus floor-normalized
retention = (transfer - chance) / (target - chance).

    python eval_transfer.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
        --mapper runs/qwen3-8b-14b/mapper_k8.pt --tasks hellaswag,arc_challenge --limit 500

`--tasks wikitext` runs the prefix-conditioned perplexity protocol of the
paper's appendix D: second 1,024 tokens scored under a 1,024-token prefix
cache. Results append to `<out-dir>/results.jsonl`, one line per (task, k).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from kvmap.mapper import Mapper
from kvmap.scorer import MODES, Pair
from kvmap.tasks import TASKS, wikitext_chunks


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--mapper", required=True)
    p.add_argument("--tasks", default="hellaswag,arc_challenge", help="comma list of: hellaswag, arc_challenge, wikitext")
    p.add_argument("--modes", default="target,transfer,source", help="subset of target,transfer,source")
    p.add_argument("--limit", type=int, default=None, help="examples per task (chunks for wikitext)")
    p.add_argument("--out-dir", default=None, help="default: the mapper's directory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default=None)
    return p.parse_args()


def multiple_choice(pair: Pair, task: str, modes: list[str], limit: int | None):
    loader, chance = TASKS[task]
    correct = {m: 0 for m in modes}
    correct_norm = {m: 0 for m in modes}
    n = 0
    t0 = time.time()
    for ex in loader(limit):
        n += 1
        for m in modes:
            scores = pair.choice_scores(m, ex.context, ex.choices)
            pick = max(range(len(scores)), key=lambda i: scores[i][0])
            pick_norm = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
            correct[m] += pick == ex.gold
            correct_norm[m] += pick_norm == ex.gold
        if n % 50 == 0:
            line = " ".join(f"{m}={correct_norm[m] / n:.3f}" for m in modes)
            print(f"  [{task}] {n} examples, acc_norm {line}, {time.time() - t0:.0f}s")
    acc = {m: correct[m] / n for m in modes}
    acc_norm = {m: correct_norm[m] / n for m in modes}
    return {"n": n, "chance": chance, "acc": acc, "acc_norm": acc_norm}


def wikitext(pair: Pair, modes: list[str], limit: int | None):
    nll = {m: 0.0 for m in modes}
    tokens = 0
    n = 0
    t0 = time.time()
    for prefix, cont in wikitext_chunks(pair.tokenizer, 1024, 1024, limit):
        n += 1
        tokens += len(cont)
        for m in modes:
            lp = pair.continuation_logprobs(m, prefix, cont)
            nll[m] += -float(lp.sum())
        if n % 10 == 0:
            line = " ".join(f"{m}={math.exp(nll[m] / tokens):.2f}" for m in modes)
            print(f"  [wikitext] {n} chunks, ppl {line}, {time.time() - t0:.0f}s")
    return {"n": n, "tokens": tokens, "ppl": {m: math.exp(nll[m] / tokens) for m in modes}}


def retention(transfer: float, target: float, chance: float) -> dict:
    out = {"retention": transfer / target if target else float("nan")}
    denom = target - chance
    out["retention_floor_normalized"] = (transfer - chance) / denom if denom else float("nan")
    return out


def main():
    args = parse()
    modes = [m for m in args.modes.split(",") if m]
    for m in modes:
        if m not in MODES:
            raise SystemExit(f"unknown mode {m}")
    tasks = [t for t in args.tasks.split(",") if t]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.mapper))
    os.makedirs(out_dir, exist_ok=True)

    print(f"loading mapper {args.mapper} ...")
    mapper = Mapper.load(args.mapper)
    if mapper.source_model != args.source or mapper.target_model != args.target:
        print(f"  note: mapper was fit for {mapper.source_model} -> {mapper.target_model}")
    print(f"  k={mapper.k}, lambda={mapper.lam}, {mapper.num_params / 1e6:.0f}M params")
    pair = Pair(args.source, args.target, mapper, args.device, getattr(torch, args.dtype), args.attn)

    results_path = os.path.join(out_dir, "results.jsonl")
    for task in tasks:
        print(f"== {task}")
        t0 = time.time()
        pair.timing.__init__()
        if task == "wikitext":
            res = wikitext(pair, modes, args.limit)
            summary = {"ppl": res["ppl"]}
            if "transfer" in modes and "target" in modes:
                summary["ppl_ratio_transfer_over_target"] = res["ppl"]["transfer"] / res["ppl"]["target"]
        elif task in TASKS:
            res = multiple_choice(pair, task, modes, args.limit)
            summary = {"acc_norm": res["acc_norm"], "acc": res["acc"]}
            if "transfer" in modes and "target" in modes:
                summary.update(retention(res["acc_norm"]["transfer"], res["acc_norm"]["target"], res["chance"]))
        else:
            raise SystemExit(f"unknown task {task}")
        record = {
            "task": task,
            "source": args.source,
            "target": args.target,
            "mapper": os.path.basename(args.mapper),
            "k": mapper.k,
            "lam": mapper.lam,
            "modes": modes,
            "limit": args.limit,
            **res,
            **summary,
            "timing_ms_total": vars(pair.timing),
            "wall_s": time.time() - t0,
        }
        with open(results_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k in ("task", "k", "n", "acc_norm", "ppl", "retention", "retention_floor_normalized", "ppl_ratio_transfer_over_target")}, indent=2))
    print(f"appended to {results_path}")


if __name__ == "__main__":
    main()
