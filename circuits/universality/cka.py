"""Universality: CKA between models' residual streams, and node overlap between circuits."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..hooks import parse_node, to_batch


@torch.no_grad()
def resid_features(model, pairs, layers=None, pool: str = "mean"):
    """`hook_resid_post` at post-output positions on the corrupt run.

    pool="mean": one row per pair (mean over its post-output positions); aligns
    across models with different tokenizers.
    pool="position": one row per (pair, post-output offset); more rows, but only
    comparable when the models tokenize identically (row counts must match).
    Returns (features [L, N, d_model] float32 on CPU, row_ids).
    """
    layers = list(range(model.cfg.n_layers)) if layers is None else list(layers)
    rows = {layer: [] for layer in layers}
    row_ids: list[str] = []

    def make_hook(layer):
        def fn(act, hook):
            sel = act[0, cur["pos"]].float().cpu()
            rows[layer].append(sel.mean(0, keepdim=True) if pool == "mean" else sel)
            return act
        return fn

    cur: dict = {}
    hooks = [(f"blocks.{layer}.hook_resid_post", make_hook(layer)) for layer in layers]
    for pair in pairs:
        cur["pos"] = pair.post_output
        model.run_with_hooks(to_batch(model, pair.corrupt_ids), fwd_hooks=hooks)
        row_ids += [pair.id] if pool == "mean" else [f"{pair.id}:{o}" for o in range(len(pair.post_output))]
    feats = torch.stack([torch.cat(rows[layer], 0) for layer in layers], 0)
    return feats, row_ids


def _hsic_debiased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """Song et al. (2012) unbiased HSIC estimator; needs n >= 4."""
    n = K.shape[0]
    K = K.clone().fill_diagonal_(0)
    L = L.clone().fill_diagonal_(0)
    tr = (K * L).sum()
    sK, sL = K.sum(), L.sum()
    kl = (K.sum(0) * L.sum(1)).sum()  # 1^T K L 1
    return (tr + sK * sL / ((n - 1) * (n - 2)) - 2 * kl / (n - 2)) / (n * (n - 3))


def _hsic_biased(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    return ((H @ K @ H) * L).sum() / (n - 1) ** 2


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA between X [n, d1] and Y [n, d2]. Debiased when n >= 4.

    Plain linear CKA with n ~ 100 rows and d ~ 3584 reads ~0.97 for *independent*
    features (Gram ≈ d·I), so the debiased estimator is the default.
    """
    X = X.double() - X.double().mean(0, keepdim=True)
    Y = Y.double() - Y.double().mean(0, keepdim=True)
    K, L = X @ X.T, Y @ Y.T
    hsic = _hsic_debiased if X.shape[0] >= 4 else _hsic_biased
    num = hsic(K, L)
    den = torch.sqrt(hsic(K, K) * hsic(L, L))
    return float(num / den) if den > 0 else float("nan")


def cka_matrix(feats_a: torch.Tensor, feats_b: torch.Tensor) -> np.ndarray:
    """[L_a, L_b] linear CKA between layer features [L, N, d] of two models (same N rows)."""
    if feats_a.shape[1] != feats_b.shape[1]:
        raise ValueError(f"row counts differ: {feats_a.shape[1]} vs {feats_b.shape[1]}")
    out = np.zeros((feats_a.shape[0], feats_b.shape[0]))
    for i in range(feats_a.shape[0]):
        for j in range(feats_b.shape[0]):
            out[i, j] = linear_cka(feats_a[i], feats_b[j])
    return out


def node_overlap(circuit_a: dict, circuit_b: dict) -> dict:
    """Jaccard over (layer, type, index). Only meaningful for same-architecture models."""
    def nodes(c):
        return {(n["type"], int(n["layer"]), int(n["index"])) for n in c["nodes"]}
    a, b = nodes(circuit_a), nodes(circuit_b)
    same_arch = (circuit_a.get("n_layers"), circuit_a.get("n_heads")) == (circuit_b.get("n_layers"), circuit_b.get("n_heads"))
    inter, union = a & b, a | b
    return dict(
        model_a=circuit_a.get("model"), model_b=circuit_b.get("model"),
        n_a=len(a), n_b=len(b), n_common=len(inter),
        jaccard=(len(inter) / len(union)) if union else float("nan"),
        same_arch=bool(same_arch),
        common=[f"{t}{l}" + (f".h{i}" if t == "a" else "") for t, l, i in sorted(inter)],
    )
