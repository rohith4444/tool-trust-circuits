"""Activation patching of shortlisted nodes, with random-node noise floor.

recovery = Σ_pairs (Δ_patched − Δ_corrupt) / Σ_pairs (Δ_clean − Δ_corrupt)   (pooled, not mean of ratios)

Conditions per node: `all` (every position), `postout` (post-output positions only),
`reverse` (corrupt activations patched into the clean run at post-output positions,
with the symmetric formula). Loop order is pair-outer so only one pair's clean and
corrupt caches live on the GPU at a time.
"""
from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd

from ..hooks import Node, all_nodes, cache_acts, cache_source, node_name, replace_hooks
from .delta import run_delta

log = logging.getLogger(__name__)
CONDITIONS = ("all", "postout", "reverse")


def patch_delta(model, pair, nodes, positions, source_cache, run: str = "corrupt") -> float:
    ids = pair.corrupt_ids if run == "corrupt" else pair.clean_ids
    return run_delta(model, ids, pair, replace_hooks(nodes, positions, cache_source(source_cache)))


def plumbing_check(model, pair, tol: float) -> dict:
    """Patch every node at post-output positions -> recovery ≈ 1; patch nothing -> 0."""
    _, cc = cache_acts(model, pair.clean_ids)
    d_clean = run_delta(model, pair.clean_ids, pair)
    d_corr = run_delta(model, pair.corrupt_ids, pair)
    d_all = patch_delta(model, pair, all_nodes(model.cfg), pair.post_output, cc)
    d_none = patch_delta(model, pair, [], pair.post_output, cc)
    rec_all = (d_all - d_corr) / (d_clean - d_corr)
    rec_none = (d_none - d_corr) / (d_clean - d_corr)
    out = dict(pair=pair.id, delta_clean=d_clean, delta_corrupt=d_corr, delta_patch_all=d_all,
               recovery_patch_all=rec_all, recovery_patch_none=rec_none, tol=tol)
    if abs(rec_all - 1.0) > tol:
        raise AssertionError(f"patch-all recovery {rec_all:.4f} != 1 (tol {tol}): {out}")
    if abs(rec_none) > 1e-6:
        raise AssertionError(f"patch-none recovery {rec_none} != 0: {out}")
    log.info("plumbing check ok: %s", out)
    return out


def patch_stage(model, pairs, shortlist: list[Node], n_draws: int = 20, seed: int = 0,
                check: bool = True) -> tuple[pd.DataFrame, dict]:
    tol = 2e-2 if model.cfg.dtype != __import__("torch").float32 else 1e-3
    checks = plumbing_check(model, pairs[0], tol) if check else {}

    rng = random.Random(seed)
    universe = all_nodes(model.cfg)
    randoms: list[Node] = []
    while len(randoms) < n_draws:
        n = rng.choice(universe)
        if n not in shortlist and n not in randoms:
            randoms.append(n)
    todo = list(dict.fromkeys(list(shortlist) + randoms))

    num = {(n, c): 0.0 for n in todo for c in CONDITIONS}
    den = {c: 0.0 for c in CONDITIONS}
    for pair in pairs:
        _, cc = cache_acts(model, pair.clean_ids)
        _, cx = cache_acts(model, pair.corrupt_ids)
        d_clean = run_delta(model, pair.clean_ids, pair)
        d_corr = run_delta(model, pair.corrupt_ids, pair)
        den["all"] += d_clean - d_corr
        den["postout"] += d_clean - d_corr
        den["reverse"] += d_corr - d_clean
        allpos = list(range(pair.T))
        for n in todo:
            num[(n, "all")] += patch_delta(model, pair, [n], allpos, cc) - d_corr
            num[(n, "postout")] += patch_delta(model, pair, [n], pair.post_output, cc) - d_corr
            num[(n, "reverse")] += patch_delta(model, pair, [n], pair.post_output, cx, run="clean") - d_clean
        del cc, cx
        log.info("patched pair %s (Δ_clean=%.3f Δ_corrupt=%.3f)", pair.id, d_clean, d_corr)

    rec = {k: (v / den[k[1]] if den[k[1]] != 0 else float("nan")) for k, v in num.items()}
    noise = np.array([rec[(n, "postout")] for n in randoms])
    p95 = float(np.percentile(noise, 95)) if len(noise) else float("nan")
    rows = []
    for n in shortlist:
        rows.append(dict(layer=n[1], type=n[0], index=n[2], name=node_name(n),
                         rec_all=rec[(n, "all")], rec_postout=rec[(n, "postout")], rec_reverse=rec[(n, "reverse")],
                         above_noise=bool(rec[(n, "postout")] > p95)))
    df = pd.DataFrame(rows)
    info = dict(checks=checks, noise_p95=p95, noise_dist=noise.tolist(), n_random=len(randoms), seed=seed,
                denominators=den)
    return df, info
