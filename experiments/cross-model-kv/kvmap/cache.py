# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reading per-layer K/V out of a Hugging Face cache and building one back
from tensors, across the two cache layouts transformers 4.5x has shipped.

- Before 4.56: `DynamicCache.key_cache[i]`, `.value_cache[i]`.
- 4.56 and later: `DynamicCache.layers[i].keys`, `.values`.

Both store keys *after* RoPE, shaped (B, n_kv, T, head_dim). `cache.update`
has been the stable way to populate a fresh cache throughout, so building
goes through it rather than through either attribute.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache


@dataclass(frozen=True)
class KvShape:
    """The axes the paper calls matched-KV when equal between two models."""

    num_layers: int
    num_kv_heads: int
    head_dim: int

    @property
    def layer_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    def matched(self, other: "KvShape") -> bool:
        return self.num_kv_heads == other.num_kv_heads and self.head_dim == other.head_dim


def kv_shape(model) -> KvShape:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return KvShape(
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
    )


def layer_kv(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(keys, values) of one layer, each (B, n_kv, T, head_dim); keys as
    cached, i.e. after RoPE."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return layers[layer].keys, layers[layer].values
    return cache.key_cache[layer], cache.value_cache[layer]


def num_layers(cache) -> int:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return len(layers)
    return len(cache.key_cache)


def build_cache(keys: list[torch.Tensor], values: list[torch.Tensor]) -> DynamicCache:
    """A cache the target model will accept as `past_key_values`, from
    per-layer tensors already in the target's cached space (RoPE applied)."""
    cache = DynamicCache()
    for layer, (k, v) in enumerate(zip(keys, values)):
        cache.update(k, v, layer)
    return cache


def flatten_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, n_kv, T, d) -> (B*T, n_kv*d), head-major within a token, the
    layout the mapper's design matrix uses."""
    b, h, t, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b * t, h * d)


def unflatten_heads(x: torch.Tensor, batch: int, num_kv_heads: int, head_dim: int) -> torch.Tensor:
    """Inverse of `flatten_heads`."""
    t = x.shape[0] // batch
    return x.reshape(batch, t, num_kv_heads, head_dim).permute(0, 2, 1, 3).contiguous()
