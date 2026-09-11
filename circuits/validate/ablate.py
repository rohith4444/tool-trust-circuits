"""Circuit validation: necessity, self-repair, faithfulness (specificity is a stub).

Mean ablation replaces a node's activation at post-output positions with its mean over
(corrupt runs × post-output positions) -- the same run type and positions the ablation
is applied to. Means are cached by a hash of (model, corrupt ids, positions).
"""
from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import numpy as np
import torch

from ..hooks import Node, all_nodes, cache_acts, mean_source, node_name, replace_hooks
from ..localize.delta import run_delta
from ..localize.roles import dla_terms

log = logging.getLogger(__name__)


def _means_key(model, pairs) -> str:
    h = hashlib.sha1(model.tt_name.encode())
    for p in pairs:
        h.update(p.corrupt_ids.numpy().tobytes())
        h.update(np.asarray(p.post_output).tobytes())
    return h.hexdigest()


@torch.no_grad()
def node_means(model, pairs, cache_path: str | Path | None = None) -> dict[str, torch.Tensor]:
    """hook_name -> mean activation ([H, d_head] for hook_z, [d_model] for hook_mlp_out), fp32."""
    key = _means_key(model, pairs)
    if cache_path and Path(cache_path).exists():
        saved = torch.load(cache_path)
        if saved.get("key") == key:
            log.info("means loaded from %s", cache_path)
            return saved["means"]
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for pair in pairs:
        _, cache = cache_acts(model, pair.corrupt_ids)
        for name in cache.keys():
            if name.endswith("hook_z") or name.endswith("hook_mlp_out"):
                s = cache[name][0, pair.post_output].float().sum(0)
                sums[name] = sums[name] + s if name in sums else s
        count += len(pair.post_output)
    means = {k: (v / count) for k, v in sums.items()}
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(key=key, means={k: v.cpu() for k, v in means.items()}), cache_path)
    return means


def ablate(model, pair, nodes, means, mode: str = "mean") -> float:
    src = mean_source({k: v.to(model.cfg.device) for k, v in means.items()}, pair.T, mode)
    return run_delta(model, pair.corrupt_ids, pair, replace_hooks(nodes, pair.post_output, src))


def necessity(model, pairs, circuit: list[Node], means, k_random: int = 20, seed: int = 0, mode: str = "mean") -> dict:
    rng = random.Random(seed)
    universe = all_nodes(model.cfg)
    draws = [rng.sample(universe, len(circuit)) for _ in range(k_random)]
    base = np.array([run_delta(model, p.corrupt_ids, p) for p in pairs])
    abl = np.array([ablate(model, p, circuit, means, mode) for p in pairs])
    rand = np.array([[ablate(model, p, d, means, mode) for p in pairs] for d in draws])   # [k, n_pairs]
    drop = base - abl
    rand_drop = (base[None] - rand).mean(1)
    gap = float(drop.mean())
    return dict(
        delta_base=float(base.mean()), delta_ablated=float(abl.mean()),
        necessity_gap=gap, abs_delta_drop=float((np.abs(base) - np.abs(abl)).mean()),
        random_dist=rand_drop.tolist(), random_mean=float(rand_drop.mean()),
        random_p95=float(np.percentile(rand_drop, 95)), above_random=bool(gap > np.percentile(rand_drop, 95)),
        k_random=k_random, circuit_size=len(circuit), mode=mode,
    )


def self_repair(model, pairs, circuit: list[Node], means, max_iters: int = 3, z: float = 2.0, mode: str = "mean") -> dict:
    """Ablate the circuit, look for non-circuit heads whose DLA spikes, add them, repeat.

    DLA on the ablated run is read with the final-norm scale held at the base run's value
    (ablation changes the norm and would shift every head's DLA uniformly). The z-score is
    over non-circuit heads only.
    """
    current = set(circuit)
    base, scales = [], []
    for pair in pairs:
        _, cache = cache_acts(model, pair.corrupt_ids)
        t = dla_terms(model, cache, pair)
        base.append(t["heads"])
        scales.append(t["scale"].to(model.cfg.device))
    means_dev = {k: v.to(model.cfg.device) for k, v in means.items()}
    added: list[Node] = []
    converged = False
    iters = 0
    for iters in range(1, max_iters + 1):
        change = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
        for i, pair in enumerate(pairs):
            hooks = replace_hooks(current, pair.post_output, mean_source(means_dev, pair.T, mode))
            _, cache = cache_acts(model, pair.corrupt_ids, fwd_hooks=hooks)
            change += dla_terms(model, cache, pair, scale=scales[i])["heads"] - base[i]
        change /= len(pairs)
        mask = torch.ones_like(change, dtype=torch.bool)
        for t, layer, idx in current:
            if t == "a":
                mask[layer, idx] = False
        vals = change[mask]
        zs = (change - vals.mean()) / (vals.std() + 1e-8)
        new = [("a", int(l), int(h)) for l, h in zip(*torch.where(mask & (zs.abs() > z)))]
        if not new:
            converged = True
            break
        log.info("self-repair iter %d: %d spiking heads %s", iters, len(new), [node_name(n) for n in new][:10])
        current |= set(new)
        added += new
    return dict(repair_heads=[node_name(n) for n in added], n_iters=iters, converged=converged,
                z_threshold=z, circuit_size_final=len(current), mode=mode)


def faithfulness(model, pairs, circuit: list[Node], means, mode: str = "mean") -> dict:
    """Mean-ablate everything except the circuit; fraction of the full model's Δ recovered (pooled over pairs)."""
    universe = all_nodes(model.cfg)
    circ = set(circuit)
    complement = [n for n in universe if n not in circ]
    base, alls, circs = [], [], []
    for pair in pairs:
        base.append(run_delta(model, pair.corrupt_ids, pair))
        alls.append(ablate(model, pair, universe, means, mode))
        circs.append(ablate(model, pair, complement, means, mode))
    base, alls, circs = map(np.array, (base, alls, circs))
    # IOI convention (Wang et al.): F(circuit) / F(model), pooled over pairs. The floor-adjusted
    # variant subtracts the everything-ablated Δ, which is not ~0 in general (the ablated stream
    # is the normalised token embedding, with its own arbitrary digit preference).
    den = float(base.sum())
    den_floor = float((base - alls).sum())
    return dict(
        faithfulness=float(circs.sum() / den) if den else float("nan"),
        faithfulness_floor_adjusted=float((circs - alls).sum() / den_floor) if den_floor else float("nan"),
        delta_base=float(base.mean()), delta_all_ablated=float(alls.mean()), delta_circuit_only=float(circs.mean()),
        all_ablated_ratio=float(np.abs(alls).mean() / (np.abs(base).mean() + 1e-8)),
        circuit_size=len(circuit), mode=mode,
    )


def specificity(*args, **kwargs):
    raise NotImplementedError(
        "TODO specificity: needs agreement (accept) traces and benchmark accuracy from the held-out split; "
        "intervene on the circuit and confirm agreement-case Δ and math_verify accuracy are unchanged."
    )
