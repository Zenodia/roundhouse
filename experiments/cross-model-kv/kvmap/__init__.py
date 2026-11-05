# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-model KV cache transfer: the closed-form per-head ridge mapper of
arXiv 2608.03893, as a small library over Hugging Face transformers.

Pipeline:

    collect_stats.py  -> second-moment statistics over a calibration set
    fit_mapper.py     -> per-target-layer source selection + ridge solve
    eval_transfer.py  -> standalone vs transfer accuracy / perplexity
    bench_latency.py  -> mapper wall time vs target re-prefill

The library is split by concern so each piece can be unit-tested on a CPU:
`rope` (strip / apply), `cache` (reading and building HF caches), `stats`
(the accumulator), `mapper` (selection, solve, apply), `data` and `tasks`
(calibration corpus and benchmarks).
"""

__version__ = "0.1.0"
