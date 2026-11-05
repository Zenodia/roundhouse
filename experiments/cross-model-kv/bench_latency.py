#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 4: the latency the paper reports (section 4.7, appendix G): the
mapper applied to an existing source cache, against the target re-running
prefill over the same tokens.

    python bench_latency.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
        --mapper runs/qwen3-8b-14b/mapper_k8.pt --seq-lens 64,512,1024,4096,8192

As in the paper: bf16, eager mode, the target's transformer body without
the LM head for re-prefill, the source prefill excluded from the mapper's
time (the source cache is the thing being reused), warmup then timed trials.
Unlike the paper there is no cross-GPU hop here: both models sit on one card.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from kvmap.mapper import Mapper, map_cache
from kvmap.rope import cos_sin
from kvmap.scorer import Pair


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--mapper", required=True)
    p.add_argument("--seq-lens", default="64,256,1024,4096,8192")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn", default=None, help="e.g. flash_attention_2 (the paper's re-prefill setting)")
    p.add_argument("--out", default=None, help="json file; default <mapper dir>/latency.json")
    return p.parse_args()


def timed(fn, warmup, trials, device):
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    return {"median_ms": times[len(times) // 2], "min_ms": times[0], "mean_ms": sum(times) / len(times)}


@torch.no_grad()
def main():
    args = parse()
    mapper = Mapper.load(args.mapper)
    pair = Pair(args.source, args.target, mapper, args.device, torch.bfloat16, args.attn)
    rows = []
    for t in [int(x) for x in args.seq_lens.split(",")]:
        ids = torch.randint(0, pair.tokenizer.vocab_size, (1, t), device=args.device)
        pos = torch.arange(t, device=args.device).unsqueeze(0)
        s_cos, s_sin = cos_sin(pair.source, pos, torch.bfloat16, args.device)
        t_cos, t_sin = cos_sin(pair.target, pos, torch.bfloat16, args.device)
        src_cache = pair.source(input_ids=ids, use_cache=True).past_key_values

        def do_map():
            map_cache(pair.mapper, src_cache, s_cos, s_sin, t_cos, t_sin, torch.bfloat16)

        body = pair.target.model  # transformer body, no LM head

        def do_prefill():
            body(input_ids=ids, use_cache=True)

        m = timed(do_map, args.warmup, args.trials, args.device)
        p = timed(do_prefill, args.warmup, args.trials, args.device)
        row = {"seq_len": t, "mapper_ms": m["median_ms"], "reprefill_ms": p["median_ms"], "speedup": p["median_ms"] / m["median_ms"]}
        rows.append(row)
        print(f"seq {t:6d}: mapper {m['median_ms']:8.1f} ms   re-prefill {p['median_ms']:8.1f} ms   speedup {row['speedup']:.1f}x")
        del src_cache
        torch.cuda.empty_cache()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.mapper)), "latency.json")
    with open(out, "w") as f:
        json.dump({"source": args.source, "target": args.target, "k": mapper.k, "attn": args.attn, "rows": rows}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
