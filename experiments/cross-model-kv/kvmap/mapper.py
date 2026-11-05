# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mapper: per-target-layer source selection by R^2, a closed-form ridge
solve per target layer (equivalently per head, since ridge columns are
independent), and the apply step that turns a source cache into a target
cache.

Notation follows the paper (arXiv 2608.03893, section 3):

    X^l   = [Kbar_s^{l_1} | ... | Kbar_s^{l_k}]      (T, k * n_kv * d)
    W^*   = (X^T X + lam I)^-1 X^T Y                  centered X, Y
    b     = Ybar - Xbar W^*
    Khat_t = (K_s R_s^-1(t) W_K + b_K) R_t(t)        content-space mapping

All heads of a target layer share the selected source layers; the solve for
the whole layer (Y with n_kv * d columns) is the per-head solve done at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .cache import KvShape, build_cache, flatten_heads, layer_kv, unflatten_heads
from .rope import apply_rope, strip_rope
from .stats import CalibrationStats, CenteredMoments

DEFAULT_LAMBDA = 0.01


# --- source selection -----------------------------------------------------


def r2_table(cm: CenteredMoments, lam: float = DEFAULT_LAMBDA) -> torch.Tensor:
    """R^2 of a single-source-layer ridge fit, for every (source layer,
    target layer), averaged over the target layer's heads. Shape (L_s, L_t).

    For source layer l and target layer L with per-head columns h:
        W_l   = (C_xx[l,l] + lam I)^-1 C_xy[l, L]
        SSE_h = tr(C_yy[L,h]) - 2 tr(W_h^T C_xy[l,L,h]) + tr(W_h^T C_xx[l,l] W_h)
        R^2_h = 1 - SSE_h / tr(C_yy[L,h])
    One factorization per source layer serves every target column.
    """
    ls, lt = cm.src.num_layers, cm.tgt.num_layers
    w_s, w_t = cm.src.layer_width, cm.tgt.layer_width
    h, d = cm.tgt.num_kv_heads, cm.tgt.head_dim
    out = torch.zeros(ls, lt, dtype=torch.float64, device=cm.cxx.device)
    sst = torch.einsum("lhdd->lh", cm.cyy_heads.double())  # (L_t, h)
    eye = torch.eye(w_s, dtype=torch.float64, device=cm.cxx.device)
    for l in range(ls):
        sl = cm.src_slice(l)
        a = cm.cxx[sl, sl].double() + lam * eye
        b = cm.cxy[sl, :].double()  # (w_s, D_t)
        w = torch.linalg.solve(a, b)  # (w_s, D_t)
        # tr(W_h^T B_h) and tr(W_h^T A W_h), per target head column block.
        wb = (w * b).sum(0)  # (D_t,) per-column W^T B diagonal
        waw = (w * (a @ w)).sum(0)
        sse_cols = -2 * wb + waw  # plus SST, added per head below
        sse = sse_cols.reshape(lt, h, d).sum(-1) + sst  # (L_t, h)
        out[l] = (1 - sse / sst.clamp_min(1e-12)).mean(1)
    return out


def select_sources(stats: CalibrationStats, k: int, lam: float = DEFAULT_LAMBDA) -> tuple[list[list[int]], dict]:
    """Top-k source layers per target layer by head-averaged R^2 averaged
    over K and V (paper section 3.2). `k` may exceed L_s, meaning all layers.
    Returns the selection and the R^2 tables for the record."""
    r2k = r2_table(stats.centered("k"), lam)
    r2v = r2_table(stats.centered("v"), lam)
    score = (r2k + r2v) / 2  # (L_s, L_t)
    k = min(k, stats.src.num_layers)
    selection = []
    for tl in range(stats.tgt.num_layers):
        top = torch.topk(score[:, tl], k).indices.sort().values.tolist()
        selection.append(top)
    return selection, {"r2_k": r2k.cpu(), "r2_v": r2v.cpu(), "score": score.cpu()}


# --- fit ------------------------------------------------------------------


@dataclass
class LayerMap:
    sources: list[int]
    w_k: torch.Tensor  # (k * w_s, w_t)
    b_k: torch.Tensor  # (w_t,)
    w_v: torch.Tensor
    b_v: torch.Tensor


@dataclass
class Mapper:
    source_model: str
    target_model: str
    src: KvShape
    tgt: KvShape
    k: int
    lam: float
    layers: list[LayerMap]
    meta: dict = field(default_factory=dict)

    @property
    def num_params(self) -> int:
        return sum(l.w_k.numel() + l.w_v.numel() for l in self.layers)

    def to(self, device, dtype) -> "Mapper":
        for l in self.layers:
            l.w_k = l.w_k.to(device=device, dtype=dtype)
            l.b_k = l.b_k.to(device=device, dtype=dtype)
            l.w_v = l.w_v.to(device=device, dtype=dtype)
            l.b_v = l.b_v.to(device=device, dtype=dtype)
        return self

    def save(self, path: str) -> None:
        torch.save(
            {
                "format": "kvmap.mapper.v1",
                "source_model": self.source_model,
                "target_model": self.target_model,
                "src": vars(self.src),
                "tgt": vars(self.tgt),
                "k": self.k,
                "lam": self.lam,
                "layers": [
                    {"sources": l.sources, "w_k": l.w_k.cpu(), "b_k": l.b_k.cpu(), "w_v": l.w_v.cpu(), "b_v": l.b_v.cpu()}
                    for l in self.layers
                ],
                "meta": self.meta,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device="cpu", dtype=torch.float32) -> "Mapper":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if blob.get("format") != "kvmap.mapper.v1":
            raise ValueError(f"{path}: not a kvmap mapper file")
        layers = [
            LayerMap(
                sources=l["sources"],
                w_k=l["w_k"].to(device=device, dtype=dtype),
                b_k=l["b_k"].to(device=device, dtype=dtype),
                w_v=l["w_v"].to(device=device, dtype=dtype),
                b_v=l["b_v"].to(device=device, dtype=dtype),
            )
            for l in blob["layers"]
        ]
        return cls(
            source_model=blob["source_model"],
            target_model=blob["target_model"],
            src=KvShape(**blob["src"]),
            tgt=KvShape(**blob["tgt"]),
            k=blob["k"],
            lam=blob["lam"],
            layers=layers,
            meta=blob.get("meta", {}),
        )


def _solve_layer(cm: CenteredMoments, sources: list[int], target_layer: int, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    idx = cm.src_index(sources)
    ts = cm.tgt_slice(target_layer)
    a = cm.cxx[idx][:, idx].double()
    a.diagonal().add_(lam)
    b = cm.cxy[idx][:, ts].double()
    w = torch.linalg.solve(a, b)
    bias = cm.mu_y[ts].double() - cm.mu_x[idx].double() @ w
    return w.float(), bias.float()


def fit(stats: CalibrationStats, k: int, lam: float = DEFAULT_LAMBDA, selection: list[list[int]] | None = None, log=print) -> Mapper:
    """Fit the mapper at source-layer count `k`. The solve runs in float64
    (the paper's analysis precision); weights are stored in float32."""
    if selection is None:
        log(f"selecting top-{k} source layers per target layer by R^2 ...")
        selection, tables = select_sources(stats, k, lam)
    else:
        tables = {}
    cmk = stats.centered("k")
    cmv = stats.centered("v")
    layers = []
    for tl in range(stats.tgt.num_layers):
        w_k, b_k = _solve_layer(cmk, selection[tl], tl, lam)
        w_v, b_v = _solve_layer(cmv, selection[tl], tl, lam)
        layers.append(LayerMap(selection[tl], w_k, b_k, w_v, b_v))
        log(f"  target layer {tl:3d}: sources {selection[tl]}")
    mapper = Mapper(
        source_model=stats.source_model,
        target_model=stats.target_model,
        src=stats.src,
        tgt=stats.tgt,
        k=min(k, stats.src.num_layers),
        lam=lam,
        layers=layers,
        meta={"n_tokens": stats.n, "stats_meta": stats.meta, **{n: t for n, t in tables.items()}},
    )
    return mapper


# --- apply ----------------------------------------------------------------


@torch.no_grad()
def map_cache(
    mapper: Mapper,
    source_cache,
    src_cos: torch.Tensor,
    src_sin: torch.Tensor,
    tgt_cos: torch.Tensor,
    tgt_sin: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
):
    """Source cache (as HF produced it) -> a target cache the target model can
    decode from, at the same positions. `*_cos/sin` are each model's own
    rotary tables for those positions, shaped (B, T, d).

    Cost per target layer: one (B*T, k*w_s) x (k*w_s, w_t) matmul for K and
    one for V, which is what the paper times against re-prefill.
    """
    src, tgt = mapper.src, mapper.tgt
    first_k, _ = layer_kv(source_cache, 0)
    batch = first_k.shape[0]
    # Content-space source features, per layer, flattened to (B*T, w_s).
    xk, xv = [], []
    for l in range(src.num_layers):
        k_rope, v = layer_kv(source_cache, l)
        k_content = strip_rope(k_rope, src_cos, src_sin)
        xk.append(flatten_heads(k_content).to(dtype))
        xv.append(flatten_heads(v).to(dtype))
    keys, values = [], []
    for lm in mapper.layers:
        x_k = torch.cat([xk[l] for l in lm.sources], dim=-1)
        x_v = torch.cat([xv[l] for l in lm.sources], dim=-1)
        y_k = torch.addmm(lm.b_k.to(dtype), x_k, lm.w_k.to(dtype))
        y_v = torch.addmm(lm.b_v.to(dtype), x_v, lm.w_v.to(dtype))
        k_content = unflatten_heads(y_k, batch, tgt.num_kv_heads, tgt.head_dim)
        keys.append(apply_rope(k_content, tgt_cos, tgt_sin))
        values.append(unflatten_heads(y_v, batch, tgt.num_kv_heads, tgt.head_dim))
    return build_cache(keys, values)
