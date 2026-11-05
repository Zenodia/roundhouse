# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Second-moment statistics over the calibration set, accumulated once and
sufficient for every ridge fit the paper describes.

The paper fits, per target layer, a ridge regression from the concatenated
KV of its top-k source layers. Rather than store the calibration KV (tens of
GB per model) or re-run the models per k, this accumulates the full source
Gram matrix `xx = X^T X` over *all* source layers concatenated, and the full
cross-moment `xy = X^T Y` against *all* target layers. Any subset of source
layers is then a sub-block, so source selection, the R^2 table and the fit
at every k come from one pass over the data.

Memory is the price. With `D_s = L_s * n_kv * d` and `D_t = L_t * n_kv * d`:

    xx: D_s^2 * 4 bytes   (per K and V)
    xy: D_s * D_t * 4     (per K and V)

Qwen3-8B -> Qwen3-14B (36 and 40 layers of 8 x 128): 5.4 GB + 6.0 GB,
twice, about 23 GB in float32. That fits beside both models on one 80 GB
card. Qwen3-14B -> 32B needs 35 GB of statistics on top of 92 GB of weights,
which is a two-card job; see the README.

Only the per-head diagonal blocks of `yy = Y^T Y` are kept, because R^2 and
the ridge solve never need cross-head target covariance.

Centering is deferred: sums are kept, and `centered()` subtracts the mean
outer products at fit time, which is the paper's "center X and Y before
solving" done without a second pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cache import KvShape


def _zeros(*shape, device, dtype=torch.float32):
    return torch.zeros(*shape, device=device, dtype=dtype)


@dataclass
class Moments:
    """Raw (uncentered) sums for one of K or V."""

    sum_x: torch.Tensor  # (D_s,)
    sum_y: torch.Tensor  # (D_t,)
    xx: torch.Tensor  # (D_s, D_s)
    xy: torch.Tensor  # (D_s, D_t)
    yy_heads: torch.Tensor  # (L_t, n_kv, d, d)

    def update(self, x: torch.Tensor, y: torch.Tensor, tgt: KvShape) -> None:
        """`x`: (N, D_s) and `y`: (N, D_t), float32, same N."""
        self.sum_x += x.sum(0)
        self.sum_y += y.sum(0)
        self.xx.addmm_(x.T, x)
        self.xy.addmm_(x.T, y)
        y4 = y.reshape(y.shape[0], tgt.num_layers, tgt.num_kv_heads, tgt.head_dim)
        self.yy_heads += torch.einsum("nlhd,nlhe->lhde", y4, y4)


@dataclass
class CalibrationStats:
    source_model: str
    target_model: str
    src: KvShape
    tgt: KvShape
    n: int = 0
    k: Moments | None = None
    v: Moments | None = None
    meta: dict = field(default_factory=dict)

    @classmethod
    def empty(cls, source_model: str, target_model: str, src: KvShape, tgt: KvShape, device) -> "CalibrationStats":
        ds = src.num_layers * src.layer_width
        dt = tgt.num_layers * tgt.layer_width

        def moments():
            return Moments(
                sum_x=_zeros(ds, device=device),
                sum_y=_zeros(dt, device=device),
                xx=_zeros(ds, ds, device=device),
                xy=_zeros(ds, dt, device=device),
                yy_heads=_zeros(tgt.num_layers, tgt.num_kv_heads, tgt.head_dim, tgt.head_dim, device=device),
            )

        return cls(source_model, target_model, src, tgt, 0, moments(), moments())

    @staticmethod
    def bytes_needed(src: KvShape, tgt: KvShape) -> int:
        ds = src.num_layers * src.layer_width
        dt = tgt.num_layers * tgt.layer_width
        per = 4 * (ds * ds + ds * dt + ds + dt + tgt.num_layers * tgt.num_kv_heads * tgt.head_dim**2)
        return 2 * per

    def update(self, xk: torch.Tensor, xv: torch.Tensor, yk: torch.Tensor, yv: torch.Tensor) -> None:
        assert xk.shape[0] == yk.shape[0] == xv.shape[0] == yv.shape[0]
        self.k.update(xk.float(), yk.float(), self.tgt)
        self.v.update(xv.float(), yv.float(), self.tgt)
        self.n += xk.shape[0]

    # --- centered views -------------------------------------------------

    def centered(self, which: str) -> "CenteredMoments":
        m = self.k if which == "k" else self.v
        n = float(self.n)
        mu_x = m.sum_x / n
        mu_y = m.sum_y / n
        cxx = m.xx - n * torch.outer(mu_x, mu_x)
        cxy = m.xy - n * torch.outer(mu_x, mu_y)
        # Per-head centered Y^T Y: subtract n * mu_h mu_h^T per head block.
        mu_y4 = mu_y.reshape(self.tgt.num_layers, self.tgt.num_kv_heads, self.tgt.head_dim)
        cyy = m.yy_heads - n * torch.einsum("lhd,lhe->lhde", mu_y4, mu_y4)
        return CenteredMoments(n=self.n, mu_x=mu_x, mu_y=mu_y, cxx=cxx, cxy=cxy, cyy_heads=cyy, src=self.src, tgt=self.tgt)

    # --- persistence ----------------------------------------------------

    def save(self, path: str) -> None:
        def pack(m: Moments):
            return {k: v.cpu() for k, v in vars(m).items()}

        torch.save(
            {
                "format": "kvmap.stats.v1",
                "source_model": self.source_model,
                "target_model": self.target_model,
                "src": vars(self.src),
                "tgt": vars(self.tgt),
                "n": self.n,
                "k": pack(self.k),
                "v": pack(self.v),
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device="cpu") -> "CalibrationStats":
        blob = torch.load(path, map_location=device, weights_only=False)
        if blob.get("format") != "kvmap.stats.v1":
            raise ValueError(f"{path}: not a kvmap stats file")

        def unpack(d):
            return Moments(**{k: v.to(device) for k, v in d.items()})

        return cls(
            source_model=blob["source_model"],
            target_model=blob["target_model"],
            src=KvShape(**blob["src"]),
            tgt=KvShape(**blob["tgt"]),
            n=blob["n"],
            k=unpack(blob["k"]),
            v=unpack(blob["v"]),
            meta=blob.get("meta", {}),
        )


@dataclass
class CenteredMoments:
    n: int
    mu_x: torch.Tensor
    mu_y: torch.Tensor
    cxx: torch.Tensor
    cxy: torch.Tensor
    cyy_heads: torch.Tensor
    src: KvShape
    tgt: KvShape

    def src_slice(self, layer: int) -> slice:
        w = self.src.layer_width
        return slice(layer * w, (layer + 1) * w)

    def tgt_slice(self, layer: int) -> slice:
        w = self.tgt.layer_width
        return slice(layer * w, (layer + 1) * w)

    def src_index(self, layers: list[int]) -> torch.Tensor:
        """Row/column indices of several source layers, concatenated in the
        given order, matching how the mapper concatenates their features."""
        w = self.src.layer_width
        return torch.cat([torch.arange(l * w, (l + 1) * w, device=self.cxx.device) for l in layers])
