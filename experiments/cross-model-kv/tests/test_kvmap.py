# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the mapper's math, on toy shapes. Run with
`python -m pytest experiments/cross-model-kv/tests -q`; they need torch and
transformers (CPU builds are fine), no GPU and no model weights.

What each one pins down, so a later refactor cannot silently break the
property the paper's result rests on:

- RoPE strip is the exact inverse of RoPE apply (section 3.3 is a no-op
  without it).
- The accumulated moments, centered, reproduce the closed-form ridge of
  section 3.1, and the R^2 table ranks the truly predictive source layer
  first (section 3.2).
- `map_cache` applied to a synthetic source cache reproduces the target KV
  the generating linear map produced -- strip, concatenate, affine map,
  re-rotate, in that order.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kvmap.cache import KvShape, build_cache, flatten_heads, layer_kv  # noqa: E402
from kvmap.mapper import fit, map_cache, r2_table, select_sources  # noqa: E402
from kvmap.rope import apply_rope, strip_rope  # noqa: E402
from kvmap.stats import CalibrationStats  # noqa: E402

torch.manual_seed(0)


def synthetic_cos_sin(positions: torch.Tensor, head_dim: int, theta: float = 10_000.0):
    """HF's rotary tables for given positions, shape (B, T, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = positions.float()[..., None] * inv_freq  # (B, T, d/2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def test_rope_strip_inverts_apply():
    b, h, t, d = 2, 3, 7, 8
    k = torch.randn(b, h, t, d)
    pos = torch.arange(t).unsqueeze(0).expand(b, t) + 5
    cos, sin = synthetic_cos_sin(pos, d)
    back = strip_rope(apply_rope(k, cos, sin), cos, sin)
    assert torch.allclose(back, k, atol=1e-5)
    # And the rotation is a genuine rotation: norms are preserved.
    assert torch.allclose(apply_rope(k, cos, sin).norm(dim=-1), k.norm(dim=-1), atol=1e-5)


class Toy:
    """A source with 3 layers and a target with 2, both 2 heads x 4 dims.
    Target layer 0 is an affine function of source layers {0, 2}; target
    layer 1 of source layers {1, 2}. Values use the same maps as keys with a
    different seed, so K and V are both exercised."""

    def __init__(self, n_tokens=4000, noise=0.01):
        self.src = KvShape(3, 2, 4)
        self.tgt = KvShape(2, 2, 4)
        self.sources = [[0, 2], [1, 2]]
        w = self.src.layer_width
        g = torch.Generator().manual_seed(1)
        self.w_k = [torch.randn(2 * w, self.tgt.layer_width, generator=g) for _ in range(2)]
        self.b_k = [torch.randn(self.tgt.layer_width, generator=g) for _ in range(2)]
        self.w_v = [torch.randn(2 * w, self.tgt.layer_width, generator=g) for _ in range(2)]
        self.b_v = [torch.randn(self.tgt.layer_width, generator=g) for _ in range(2)]
        self.noise = noise
        self.n = n_tokens
        self.g = g

    def sample(self, n: int):
        """Content-space source features (n, D_s) and the target they imply."""
        w = self.src.layer_width
        xk = torch.randn(n, self.src.num_layers * w, generator=self.g) + 0.5  # non-zero mean: centering matters
        xv = torch.randn(n, self.src.num_layers * w, generator=self.g) - 0.3
        yk, yv = [], []
        for tl in range(self.tgt.num_layers):
            cols = torch.cat([torch.arange(l * w, (l + 1) * w) for l in self.sources[tl]])
            yk.append(xk[:, cols] @ self.w_k[tl] + self.b_k[tl] + self.noise * torch.randn(n, self.tgt.layer_width, generator=self.g))
            yv.append(xv[:, cols] @ self.w_v[tl] + self.b_v[tl] + self.noise * torch.randn(n, self.tgt.layer_width, generator=self.g))
        return xk, xv, torch.cat(yk, -1), torch.cat(yv, -1)

    def stats(self) -> CalibrationStats:
        s = CalibrationStats.empty("toy-src", "toy-tgt", self.src, self.tgt, "cpu")
        for _ in range(4):  # several updates, to exercise accumulation
            s.update(*self.sample(self.n // 4))
        return s


def test_r2_table_ranks_true_sources_first():
    toy = Toy()
    stats = toy.stats()
    score = (r2_table(stats.centered("k")) + r2_table(stats.centered("v"))) / 2  # (L_s, L_t)
    # Target layer 0 depends on sources 0 and 2; source 1 should be worst.
    assert score[1, 0] < min(score[0, 0], score[2, 0])
    # Target layer 1 depends on sources 1 and 2; source 0 should be worst.
    assert score[0, 1] < min(score[1, 1], score[2, 1])
    # An unrelated source layer explains ~nothing (its R^2 sits near zero).
    assert score[1, 0] < 0.1 and score[0, 1] < 0.1


def test_select_sources_top_k():
    toy = Toy()
    selection, _ = select_sources(toy.stats(), k=2)
    assert selection == toy.sources
    all_layers, _ = select_sources(toy.stats(), k=99)
    assert all_layers == [[0, 1, 2], [0, 1, 2]]


def test_fit_recovers_the_generating_map():
    toy = Toy()
    mapper = fit(toy.stats(), k=2, lam=1e-6, log=lambda s: None)
    for tl, lm in enumerate(mapper.layers):
        assert lm.sources == toy.sources[tl]
        assert torch.allclose(lm.w_k, toy.w_k[tl], atol=5e-2), (lm.w_k - toy.w_k[tl]).abs().max()
        assert torch.allclose(lm.b_k, toy.b_k[tl], atol=5e-2)
        assert torch.allclose(lm.w_v, toy.w_v[tl], atol=5e-2)
        assert torch.allclose(lm.b_v, toy.b_v[tl], atol=5e-2)


def test_fit_matches_direct_ridge_on_raw_data():
    """The streamed, centered moments must give the same W as a direct solve
    on the stacked design matrix -- the two spellings of equation 4."""
    toy = Toy(n_tokens=2000, noise=0.1)
    xk, xv, yk, yv = toy.sample(2000)
    stats = CalibrationStats.empty("s", "t", toy.src, toy.tgt, "cpu")
    stats.update(xk[:700], xv[:700], yk[:700], yv[:700])
    stats.update(xk[700:], xv[700:], yk[700:], yv[700:])
    lam = 0.5
    mapper = fit(stats, k=2, lam=lam, selection=toy.sources, log=lambda s: None)
    w = toy.src.layer_width
    for tl in range(toy.tgt.num_layers):
        cols = torch.cat([torch.arange(l * w, (l + 1) * w) for l in toy.sources[tl]])
        x = xk[:, cols].double()
        y = yk[:, tl * toy.tgt.layer_width : (tl + 1) * toy.tgt.layer_width].double()
        xc, yc = x - x.mean(0), y - y.mean(0)
        direct = torch.linalg.solve(xc.T @ xc + lam * torch.eye(x.shape[1], dtype=torch.float64), xc.T @ yc)
        bias = y.mean(0) - x.mean(0) @ direct
        assert torch.allclose(mapper.layers[tl].w_k.double(), direct, atol=1e-4)
        assert torch.allclose(mapper.layers[tl].b_k.double(), bias, atol=1e-4)


def test_map_cache_reproduces_target_kv():
    """End to end on a fake cache: source keys are stored rotated (as HF
    does), the mapper strips, maps and re-rotates with the target's tables,
    and the result matches the target KV the generating map implies."""
    toy = Toy()
    mapper = fit(toy.stats(), k=2, lam=1e-6, log=lambda s: None)
    b, t = 1, 6
    xk, xv, yk, yv = toy.sample(b * t)
    d, h = toy.src.head_dim, toy.src.num_kv_heads
    pos = torch.arange(t).unsqueeze(0) + 11
    s_cos, s_sin = synthetic_cos_sin(pos, d, theta=10_000.0)
    t_cos, t_sin = synthetic_cos_sin(pos, d, theta=1_000_000.0)  # a different theta, as across checkpoints

    def to_heads(flat, layers):
        w = h * d
        return [flat[:, l * w : (l + 1) * w].reshape(b, t, h, d).permute(0, 2, 1, 3).contiguous() for l in range(layers)]

    src_keys = [apply_rope(k, s_cos, s_sin) for k in to_heads(xk, toy.src.num_layers)]
    src_vals = to_heads(xv, toy.src.num_layers)
    source_cache = build_cache(src_keys, src_vals)

    mapped = map_cache(mapper, source_cache, s_cos, s_sin, t_cos, t_sin, dtype=torch.float32)

    exp_keys = [apply_rope(k, t_cos, t_sin) for k in to_heads(yk, toy.tgt.num_layers)]
    exp_vals = to_heads(yv, toy.tgt.num_layers)
    for tl in range(toy.tgt.num_layers):
        k, v = layer_kv(mapped, tl)
        assert k.shape == exp_keys[tl].shape
        assert torch.allclose(k, exp_keys[tl], atol=0.15), (k - exp_keys[tl]).abs().max()
        assert torch.allclose(v, exp_vals[tl], atol=0.15), (v - exp_vals[tl]).abs().max()
    # The round trip through flatten_heads is layout-exact.
    assert torch.equal(flatten_heads(src_vals[0]), xv[:, : h * d])


def test_moments_memory_estimate_is_the_float32_footprint():
    src, tgt = KvShape(36, 8, 128), KvShape(40, 8, 128)
    gb = CalibrationStats.bytes_needed(src, tgt) / 1e9
    assert 22 < gb < 24  # the README's "about 23 GB" (decimal) for Qwen3-8B -> 14B


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
