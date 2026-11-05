# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RoPE as an invertible rotation, so keys can move between the cached
(position-coupled) space and the content space the mapper is fit in.

HF's Llama-family convention (Qwen3, Llama 3, Mistral all share it):

    rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])
    k_rope = k * cos + rotate_half(k) * sin

Each (x_i, x_{i+d/2}) pair is rotated by the angle for frequency i at the
token's position. A rotation is orthogonal, so the inverse is the rotation by
the negative angle:

    k = k_rope * cos - rotate_half(k_rope) * sin

The paper's content-space mapping (its section 3.3) is exactly: strip the
source model's RoPE, map, re-apply the target model's RoPE at the same
positions. Source and target may use different theta; `cos`/`sin` must come
from the model whose space the keys are in.
"""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _broadcast(cos: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """`cos`/`sin` arrive as (B, T, D) from HF's rotary embedding; keys are
    (B, H, T, D). Insert the head axis."""
    if cos.dim() == 3 and k.dim() == 4:
        return cos.unsqueeze(1)
    return cos


def apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Content-space keys -> cached keys, at the positions `cos`/`sin` encode."""
    cos = _broadcast(cos, k).to(k.dtype)
    sin = _broadcast(sin, k).to(k.dtype)
    return k * cos + rotate_half(k) * sin


def strip_rope(k_rope: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Cached keys -> content-space keys. Exact inverse of `apply_rope` up to
    the dtype's rounding; see the module doc for the algebra."""
    cos = _broadcast(cos, k_rope).to(k_rope.dtype)
    sin = _broadcast(sin, k_rope).to(k_rope.dtype)
    return k_rope * cos - rotate_half(k_rope) * sin


def cos_sin(model, position_ids: torch.Tensor, dtype: torch.dtype, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The (cos, sin) tables a HF decoder would use at `position_ids`,
    shaped (B, T, head_dim), read from the model's own rotary module so theta,
    scaling and head_dim are whatever that checkpoint says.

    `rotary_emb(x, position_ids)` only reads `x.device` and `x.dtype`, so a
    one-element dummy stands in for hidden states.
    """
    rotary = _rotary_module(model)
    dummy = torch.zeros(1, dtype=dtype, device=device)
    cos, sin = rotary(dummy, position_ids.to(device))
    return cos, sin


def _rotary_module(model):
    # Decoder-only causal LMs in transformers 4.5x expose it at model.model.rotary_emb.
    inner = getattr(model, "model", model)
    rotary = getattr(inner, "rotary_emb", None)
    if rotary is None:
        raise AttributeError(
            "could not find `rotary_emb` on the model; this library assumes the HF "
            "Llama-family layout (model.model.rotary_emb)"
        )
    return rotary
