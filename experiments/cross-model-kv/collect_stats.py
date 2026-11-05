#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step 1: run source and target over the calibration set and accumulate
the second-moment statistics every later fit reads.

    python collect_stats.py --source Qwen/Qwen3-8B --target Qwen/Qwen3-14B \
        --num-seqs 500 --seq-len 1024 --stride 4 --out runs/qwen3-8b-14b/stats.pt

Both models are loaded on one device in bf16; the statistics are float32 on
`--stats-device` (default: same device). Keys are RoPE-stripped before they
enter the statistics, so the fit is in content space (paper section 3.3).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kvmap.cache import flatten_heads, kv_shape, layer_kv
from kvmap.data import calibration_sequences, local_text_sequences
from kvmap.rope import cos_sin, strip_rope
from kvmap.stats import CalibrationStats


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--out", required=True, help="stats .pt file to write")
    p.add_argument("--num-seqs", type=int, default=500)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--stride", type=int, default=4, help="keep every Nth token position")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--stats-device", default=None, help="where the float32 moments live (default: --device)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn", default=None, help="attn_implementation, e.g. flash_attention_2 or sdpa")
    p.add_argument("--text-file", default=None, help="offline calibration text instead of FineWeb-Edu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse()
    dtype = getattr(torch, args.dtype)
    stats_device = args.stats_device or args.device
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    print(f"loading {args.source} and {args.target} on {args.device} ({args.dtype}) ...")
    kw = {"torch_dtype": dtype, "device_map": {"": args.device}}
    if args.attn:
        kw["attn_implementation"] = args.attn
    tok = AutoTokenizer.from_pretrained(args.target)
    src = AutoModelForCausalLM.from_pretrained(args.source, **kw).eval()
    tgt = AutoModelForCausalLM.from_pretrained(args.target, **kw).eval()
    src_shape, tgt_shape = kv_shape(src), kv_shape(tgt)
    print(f"source KV {src_shape}\ntarget KV {tgt_shape}")
    if not src_shape.matched(tgt_shape):
        raise SystemExit("not a matched-KV pair (KV head count and head_dim must agree); the paper's mapper needs that")
    need = CalibrationStats.bytes_needed(src_shape, tgt_shape)
    print(f"statistics need {need / 2**30:.1f} GiB on {stats_device}")

    stats = CalibrationStats.empty(args.source, args.target, src_shape, tgt_shape, stats_device)
    stats.meta = {
        "num_seqs": args.num_seqs,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "corpus": args.text_file or "HuggingFaceFW/fineweb-edu:sample-10BT",
        "dtype": args.dtype,
        "seed": args.seed,
    }

    if args.text_file:
        seqs = local_text_sequences(args.text_file, tok, args.num_seqs, args.seq_len)
    else:
        seqs = calibration_sequences(tok, args.num_seqs, args.seq_len, seed=args.seed)

    keep = torch.arange(0, args.seq_len, args.stride, device=args.device)
    t0 = time.time()
    batch = []

    def flush(batch):
        ids = torch.stack(batch).to(args.device)
        b, t = ids.shape
        pos = torch.arange(t, device=args.device).unsqueeze(0).expand(b, t)
        s_cos, s_sin = cos_sin(src, pos, dtype, args.device)
        t_cos, t_sin = cos_sin(tgt, pos, dtype, args.device)
        s_cache = src(input_ids=ids, use_cache=True).past_key_values
        t_cache = tgt(input_ids=ids, use_cache=True).past_key_values

        def features(cache, cos, sin, n_layers):
            xk, xv = [], []
            for l in range(n_layers):
                k, v = layer_kv(cache, l)
                k = strip_rope(k, cos, sin)[:, :, keep, :]
                v = v[:, :, keep, :]
                xk.append(flatten_heads(k))
                xv.append(flatten_heads(v))
            return torch.cat(xk, -1), torch.cat(xv, -1)

        xk, xv = features(s_cache, s_cos, s_sin, src_shape.num_layers)
        yk, yv = features(t_cache, t_cos, t_sin, tgt_shape.num_layers)
        stats.update(xk.to(stats_device), xv.to(stats_device), yk.to(stats_device), yv.to(stats_device))
        del s_cache, t_cache

    for i, seq in enumerate(seqs, 1):
        batch.append(seq)
        if len(batch) == args.batch_size:
            flush(batch)
            batch = []
        if i % 25 == 0 or i == args.num_seqs:
            el = time.time() - t0
            print(f"  {i}/{args.num_seqs} sequences, {stats.n} token observations, {el:.0f}s")
    if batch:
        flush(batch)

    print(f"saving {args.out} ({need / 2**30:.1f} GiB) ...")
    stats.save(args.out)
    with open(os.path.splitext(args.out)[0] + ".json", "w") as f:
        json.dump({"source": args.source, "target": args.target, "src": vars(src_shape), "tgt": vars(tgt_shape), "n_tokens": stats.n, **stats.meta}, f, indent=2)
    print(f"done in {time.time() - t0:.0f}s; {stats.n} token observations")


if __name__ == "__main__":
    main()
