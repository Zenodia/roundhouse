#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 2: select source layers and solve the ridge mapper from saved
statistics. No model is loaded; this is linear algebra over the stats file.

    python fit_mapper.py --stats runs/qwen3-8b-14b/stats.pt --k 8 --out runs/qwen3-8b-14b/mapper_k8.pt

Several `--k` values may be given; each writes `<out>` with `_k<k>` inserted
before the extension. The paper sweeps k in {1,2,4,6,8,10,12,16,20,24,all}
and picks per pair by mean log-likelihood accuracy; `all` is spelled 0 here.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from kvmap.mapper import DEFAULT_LAMBDA, fit, select_sources
from kvmap.stats import CalibrationStats


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", required=True, help="mapper .pt path; with several --k the k is inserted into the name")
    p.add_argument("--k", type=int, nargs="+", default=[8], help="source layers per target layer; 0 means all")
    p.add_argument("--lam", type=float, default=DEFAULT_LAMBDA, help="ridge lambda on the centered Gram (paper: 0.01)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def out_path(base: str, k: int, multi: bool) -> str:
    if not multi:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}_k{k}{ext}"


def main():
    args = parse()
    t0 = time.time()
    print(f"loading {args.stats} onto {args.device} ...")
    stats = CalibrationStats.load(args.stats, device=args.device)
    print(f"  {stats.source_model} -> {stats.target_model}, {stats.n} token observations")
    print(f"  source {stats.src}\n  target {stats.tgt}")

    # One R^2 table serves every k.
    print("computing the single-source-layer R^2 table ...")
    _, tables = select_sources(stats, 1, args.lam)
    score = tables["score"]
    for k in args.k:
        kk = stats.src.num_layers if k == 0 else min(k, stats.src.num_layers)
        selection = [torch.topk(score[:, tl], kk).indices.sort().values.tolist() for tl in range(stats.tgt.num_layers)]
        print(f"fitting k={kk} ...")
        mapper = fit(stats, kk, args.lam, selection=selection, log=lambda s: None)
        mapper.meta.update({"r2_k": tables["r2_k"], "r2_v": tables["r2_v"], "score": score})
        path = out_path(args.out, kk, len(args.k) > 1)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mapper.save(path)
        gib = mapper.num_params * 4 / 2**30
        print(f"  wrote {path}: {mapper.num_params / 1e6:.0f}M params ({gib:.2f} GiB float32)")
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(
                {
                    "source": stats.source_model,
                    "target": stats.target_model,
                    "k": kk,
                    "lam": args.lam,
                    "n_tokens": stats.n,
                    "num_params": mapper.num_params,
                    "selection": selection,
                    "best_single_source_r2_per_target_layer": score.max(0).values.tolist(),
                },
                f,
                indent=2,
            )
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
