# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoring continuations under three prefix-cache providers:

    target    the target's own prefill          (standalone, the ceiling)
    transfer  the source's prefill, mapped       (the paper's method)
    source    the source alone                   (what the small model gets)

All three score the same continuation tokens with the same model (the
target, except for `source`), so the only thing that varies is where the
prefix KV came from.

One detail decides what the cache covers. The logit that predicts the first
continuation token comes from the forward pass over the *last prefix token*,
and a cache alone does not carry logits. So the prefix cache is built over
`prefix[:-1]`, and the scored forward pass takes `prefix[-1] + cont[:-1]`.
The last prefix token therefore gets native target KV in every mode; the
other P-1 prefix tokens are where the modes differ.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import build_cache, kv_shape, layer_kv, num_layers
from .mapper import Mapper, map_cache
from .rope import cos_sin

MODES = ("target", "transfer", "source")


def clone_cache(cache):
    """A forward pass appends to the cache it is given; scoring several
    continuations against one prefix needs a fresh copy each time."""
    keys, values = [], []
    for l in range(num_layers(cache)):
        k, v = layer_kv(cache, l)
        keys.append(k.clone())
        values.append(v.clone())
    return build_cache(keys, values)


def encode_pair(tokenizer, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """Harness-style split: tokenize the whole string, take the context's
    own token count as the boundary, so a merge across the boundary lands
    on the continuation side."""
    ctx = tokenizer(context, add_special_tokens=False)["input_ids"]
    whole = tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
    return whole[: len(ctx)], whole[len(ctx) :]


@dataclass
class Timing:
    source_prefill_ms: float = 0.0
    map_ms: float = 0.0
    target_prefill_ms: float = 0.0


class Pair:
    """Source and target loaded on one device, plus an optional mapper."""

    def __init__(
        self,
        source_model: str,
        target_model: str,
        mapper: Mapper | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = None,
        need_source: bool = True,
        need_target: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        kwargs = {"torch_dtype": dtype, "device_map": {"": device}}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.source = AutoModelForCausalLM.from_pretrained(source_model, **kwargs).eval() if need_source else None
        self.target = AutoModelForCausalLM.from_pretrained(target_model, **kwargs).eval() if need_target else None
        if self.source is not None and self.target is not None:
            src_tok = AutoTokenizer.from_pretrained(source_model)
            if src_tok.get_vocab() != self.tokenizer.get_vocab():
                raise ValueError("source and target tokenizers differ; the paper's transfer assumes one tokenizer per family")
            if not kv_shape(self.source).matched(kv_shape(self.target)):
                raise ValueError(f"not a matched-KV pair: source {kv_shape(self.source)} vs target {kv_shape(self.target)}")
        self.mapper = mapper.to(device, dtype) if mapper is not None else None
        self.timing = Timing()

    # --- prefix caches --------------------------------------------------

    def _sync(self):
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

    @torch.no_grad()
    def prefix_cache(self, mode: str, prefix_ids: list[int]):
        """Returns (cache or None, model that will consume it)."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        model = self.source if mode == "source" else self.target
        if not prefix_ids:
            return None, model
        ids = torch.tensor([prefix_ids], device=self.device)
        if mode in ("target", "source"):
            t0 = time.perf_counter()
            out = model(input_ids=ids, use_cache=True)
            self._sync()
            if mode == "target":
                self.timing.target_prefill_ms += (time.perf_counter() - t0) * 1e3
            return out.past_key_values, model
        if self.mapper is None:
            raise ValueError("transfer mode needs a mapper")
        t0 = time.perf_counter()
        src_out = self.source(input_ids=ids, use_cache=True)
        self._sync()
        t1 = time.perf_counter()
        pos = torch.arange(len(prefix_ids), device=self.device).unsqueeze(0)
        s_cos, s_sin = cos_sin(self.source, pos, self.dtype, self.device)
        t_cos, t_sin = cos_sin(self.target, pos, self.dtype, self.device)
        mapped = map_cache(self.mapper, src_out.past_key_values, s_cos, s_sin, t_cos, t_sin, self.dtype)
        self._sync()
        t2 = time.perf_counter()
        self.timing.source_prefill_ms += (t1 - t0) * 1e3
        self.timing.map_ms += (t2 - t1) * 1e3
        return mapped, self.target

    # --- scoring --------------------------------------------------------

    @torch.no_grad()
    def continuation_logprobs(self, mode: str, prefix_ids: list[int], cont_ids: list[int], cache=None, model=None) -> torch.Tensor:
        """Per-token log p(cont_i | prefix, cont_<i), shape (len(cont),).
        Pass a (cache, model) from `prefix_cache(mode, prefix_ids[:-1])` to
        reuse one prefix across several continuations; it is cloned here."""
        if not prefix_ids:
            raise ValueError("an empty prefix has no last token to score from")
        if cache is None and model is None:
            cache, model = self.prefix_cache(mode, prefix_ids[:-1])
        suffix = [prefix_ids[-1]] + cont_ids[:-1]
        ids = torch.tensor([suffix], device=self.device)
        past = clone_cache(cache) if cache is not None else None
        logits = model(input_ids=ids, past_key_values=past, use_cache=True).logits[0].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(cont_ids, device=self.device)
        return logprobs.gather(1, targets.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def choice_scores(self, mode: str, context: str, choices: list[str]) -> list[tuple[float, int]]:
        """(sum log-likelihood, continuation character length) per choice,
        with the prefix cache built once and cloned per choice. The length is
        what lm-eval's `acc_norm` divides by."""
        encoded = [encode_pair(self.tokenizer, context, c) for c in choices]
        # Every choice shares the context tokens; the harness splits per choice
        # in case a merge moves the boundary, so take the shortest common prefix.
        prefix = encoded[0][0]
        for ctx, _ in encoded[1:]:
            n = 0
            while n < min(len(prefix), len(ctx)) and prefix[n] == ctx[n]:
                n += 1
            prefix = prefix[:n]
        cache, model = self.prefix_cache(mode, prefix[:-1])
        scores = []
        for (ctx, cont), text in zip(encoded, choices):
            full_cont = ctx[len(prefix) :] + cont
            lp = self.continuation_logprobs(mode, prefix, full_cont, cache=cache, model=model)
            scores.append((float(lp.sum()), len(text)))
        return scores
