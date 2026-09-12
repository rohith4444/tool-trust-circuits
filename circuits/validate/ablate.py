"""Circuit validation: necessity, self-repair, faithfulness, specificity.

Two halves, deliberately separated:

  collect_deltas()  runs the model. One row per pair, one column per ablation condition.
  necessity() / faithfulness() / ...  are pure functions of that table.

Redefining a metric therefore costs nothing: re-read the per-pair CSV and recompute on a
laptop (scripts/recompute_metrics.py). Before the split, every change of definition meant
another ~18,000 forward passes.

SIGNS

Δ = logit(Y) − logit(X) is not positive by construction. On the corrupt runs it is mixed:
mean(Δ) is −1.54 for ToRL while mean|Δ| is ≥3.47, so more than half the signal cancels
when pairs are pooled as a signed sum. Every statistic here is therefore either computed
per pair and then aggregated, or pooled against |denominator|. The IOI convention
(F(circuit)/F(model), summed over examples) assumes a sign-consistent metric and does not
survive the move to this task unchanged.

Mean ablation replaces a node's activation at the chosen positions with its mean over
(corrupt runs × those positions) -- the same run type and positions the ablation is
applied to. Means are cached by a hash of (model, corrupt ids, positions).
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..hooks import Node, all_nodes, cache_acts, mean_source, node_name, replace_hooks, to_batch
from ..localize.delta import delta, run_delta
from ..localize.roles import dla_terms

log = logging.getLogger(__name__)

POSITIONS = ("postout", "preout", "all")
EPS = 1e-6


def positions_of(pair, which: str = "postout") -> list[int]:
    """Which token positions an ablation is applied at.

    `postout` is everything after the injected value's closing fence -- where the model
    decides what to do with what it just read. `all` is the whole sequence, the IOI
    convention, which also covers the value's own tokens and the prefix that produced
    the model's own answer.

    `preout` is everything before the injected value: the problem statement, the model's
    reasoning, and its code, up to the opening fence. No tool result exists there yet, so
    nothing in that region is tool assimilation -- which is what makes it the control
    region for specificity. A circuit that only decides whether to trust a tool should be
    close to inert there.
    """
    if which == "postout":
        return list(pair.post_output)
    if which == "preout":
        return list(range(pair.output_span[0]))
    if which == "all":
        return list(range(pair.T))
    raise ValueError(f"unknown positions {which!r}, expected one of {POSITIONS}")


def _means_key(model, pairs, positions: str) -> str:
    h = hashlib.sha1(f"{model.tt_name}|{positions}".encode())
    for p in pairs:
        h.update(p.corrupt_ids.numpy().tobytes())
        h.update(np.asarray(positions_of(p, positions)).tobytes())
    return h.hexdigest()


@torch.no_grad()
def node_means(model, pairs, cache_path: str | Path | None = None,
               positions: str = "postout") -> dict[str, torch.Tensor]:
    """hook_name -> mean activation ([H, d_head] for hook_z, [d_model] for hook_mlp_out), fp32."""
    key = _means_key(model, pairs, positions)
    if cache_path and Path(cache_path).exists():
        saved = torch.load(cache_path)
        if saved.get("key") == key:
            log.info("means loaded from %s", cache_path)
            return saved["means"]
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for pair in pairs:
        pos = positions_of(pair, positions)
        _, cache = cache_acts(model, pair.corrupt_ids)
        for name in cache.keys():
            if name.endswith("hook_z") or name.endswith("hook_mlp_out"):
                s = cache[name][0, pos].float().sum(0)
                sums[name] = sums[name] + s if name in sums else s
        count += len(pos)
    means = {k: (v / count) for k, v in sums.items()}
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(key=key, means={k: v.cpu() for k, v in means.items()}), cache_path)
    return means


def ablate(model, pair, nodes, means, mode: str = "mean", positions: str = "postout",
           run: str = "corrupt") -> float:
    src = mean_source({k: v.to(model.cfg.device) for k, v in means.items()}, pair.T, mode)
    ids = pair.corrupt_ids if run == "corrupt" else pair.clean_ids
    return run_delta(model, ids, pair, replace_hooks(nodes, positions_of(pair, positions), src))


# --------------------------------------------------------------------------- collection

def ablatable(cfg, protect: set[int] | None = None) -> list[Node]:
    """Every node that may be ablated: all of them, minus the protected layers.

    Layer 0 is protected by convention. MLP0 turns token ids into the representation
    everything downstream reads; mean-ablating it does not remove a mechanism, it removes
    the input. Faithfulness makes this concrete -- ablate the complement with m0 in it and
    the circuit-only run has no text to work from, so the number measures nothing (ToRL:
    0.412 with m0 kept, 0.069 with it ablated). Wang et al. treat MLP0 as part of the
    extended embedding for the same reason: never ablated, never counted as circuit.
    """
    protect = protect or set()
    return [n for n in all_nodes(cfg) if n[1] not in protect]


@torch.no_grad()
def collect_deltas(model, pairs, circuits: dict[str, list[Node]], means, mode: str = "mean",
                   positions: str = "postout", k_random: int = 20, seed: int = 0,
                   main: str | None = None, protect: set[int] | None = None) -> pd.DataFrame:
    """Every Δ the metrics need, one row per pair. This is the only GPU-bound part.

    `circuits` maps a name to a node list; the k-curve passes {"k10": [...], "k50": [...]}.
    For each name two columns are produced:

        abl_<name>     the circuit ablated          -> necessity
        circ_<name>    the complement ablated       -> faithfulness

    plus `base` (nothing ablated), `all_abl` (everything ablated, the floor) and
    `rand_0..rand_{k-1}`, random node sets the size of the `main` circuit.

    Per pair that is 2 + 2*|circuits| + k_random forwards. The random draws dominate:
    at the default k_random=20 with five circuit sizes they are 63% of the work.
    """
    main = main or next(iter(circuits))
    if main not in circuits:
        raise KeyError(f"main circuit {main!r} not in {sorted(circuits)}")
    universe = ablatable(model.cfg, protect)
    rng = random.Random(seed)
    draws = [rng.sample(universe, len(circuits[main])) for _ in range(k_random)]
    complements = {name: [n for n in universe if n not in set(nodes)] for name, nodes in circuits.items()}

    per_pair = 2 + 2 * len(circuits) + k_random
    log.info("collect_deltas: %d pairs x %d forwards = %d, positions=%s",
             len(pairs), per_pair, len(pairs) * per_pair, positions)
    t0 = time.time()
    rows = []
    for i, pair in enumerate(pairs):
        row = {
            "pair_id": pair.id, "label": pair.meta.get("label"), "T": pair.T,
            "n_positions": len(positions_of(pair, positions)),
            "base": run_delta(model, pair.corrupt_ids, pair),
            "all_abl": ablate(model, pair, universe, means, mode, positions),
        }
        for name, nodes in circuits.items():
            row[f"abl_{name}"] = ablate(model, pair, nodes, means, mode, positions)
            row[f"circ_{name}"] = ablate(model, pair, complements[name], means, mode, positions)
        for j, d in enumerate(draws):
            row[f"rand_{j}"] = ablate(model, pair, d, means, mode, positions)
        rows.append(row)
        done = i + 1
        if done % 5 == 0 or done == len(pairs):
            el = time.time() - t0
            log.info("collect_deltas: %d/%d pairs  %.2fs/pair  %.1f ms/forward  eta %.1f min",
                     done, len(pairs), el / done, 1000 * el / (done * per_pair),
                     (len(pairs) - done) * el / done / 60)
    df = pd.DataFrame(rows)
    df.attrs.update(main=main, k_random=k_random, mode=mode, positions=positions,
                    sizes={k: len(v) for k, v in circuits.items()},
                    protect=sorted(protect or []), n_ablatable=len(universe))
    return df


# --------------------------------------------------------------------------- metrics (pure)

def _collapse(base: np.ndarray, abl: np.ndarray) -> dict:
    """How much of |Δ| the ablation removed.

    Magnitude, not signed difference: Δ points either way across pairs, so `base - abl`
    averages toward zero for reasons that have nothing to do with the circuit. The pooled
    form (Σ|base| − Σ|abl|) / Σ|base| is the headline because the per-pair fraction is
    unstable wherever |base| is near zero; the median of per-pair fractions is reported
    beside it as the robust check that the pooled number is not carried by a few pairs.
    """
    ab, aa = np.abs(base), np.abs(abl)
    denom = float(ab.sum())
    per_pair = (ab - aa) / np.maximum(ab, EPS)
    return dict(
        pooled=float((denom - aa.sum()) / denom) if denom else float("nan"),
        median=float(np.median(per_pair)),
        mean_abs_base=float(ab.mean()), mean_abs_ablated=float(aa.mean()),
    )


def necessity(df: pd.DataFrame, name: str | None = None, k_random: int | None = None) -> dict:
    """Ablate the circuit; how much of |Δ| collapses, against random sets of the same size.

    `above_random` compares the pooled collapse with the 95th percentile of the random
    draws' pooled collapse. The previous version compared a signed drop (base − ablated,
    negative whenever Δ_base is negative) against a positive percentile, so it could not
    return True regardless of how strong the circuit was.
    """
    name = name or df.attrs.get("main")
    k_random = df.attrs.get("k_random", 0) if k_random is None else k_random
    base = df["base"].to_numpy()
    abl = df[f"abl_{name}"].to_numpy()
    c = _collapse(base, abl)
    rand_cols = [f"rand_{j}" for j in range(k_random) if f"rand_{j}" in df.columns]
    rand = np.array([_collapse(base, df[col].to_numpy())["pooled"] for col in rand_cols])
    p95 = float(np.percentile(rand, 95)) if len(rand) else float("nan")
    return dict(
        collapse=c["pooled"], collapse_median=c["median"],
        abs_delta_base=c["mean_abs_base"], abs_delta_ablated=c["mean_abs_ablated"],
        abs_delta_drop=c["mean_abs_base"] - c["mean_abs_ablated"],
        random_collapse_dist=rand.tolist(),
        random_collapse_mean=float(rand.mean()) if len(rand) else float("nan"),
        random_collapse_p95=p95,
        above_random=bool(c["pooled"] > p95) if len(rand) else None,
        # signed means, kept for reference -- not a test, they cancel across pairs
        delta_base=float(base.mean()), delta_ablated=float(abl.mean()),
        necessity_gap_signed=float((base - abl).mean()),
        k_random=len(rand), circuit_size=df.attrs.get("sizes", {}).get(name), n_pairs=len(df),
    )


def faithfulness(df: pd.DataFrame, name: str | None = None, min_den: float = 0.1) -> dict:
    """Ablate everything except the circuit; how much of Δ the circuit alone reproduces.

    Per pair, against that pair's own floor:

        f_i = (Δ_circuit_only,i − Δ_all_ablated,i) / (Δ_base,i − Δ_all_ablated,i)

    The floor matters: with every node ablated the residual stream is the normalised token
    embedding, which has its own digit preference. On ToRL pair 0 that floor is +2.5 against
    a base of −2.625 -- as large as the signal and opposite in sign, so ignoring it is not
    an option.

    Headline is the median of f_i. `pooled_signed` aligns each pair by the sign of its own
    denominator before summing, which is the pooled number that survives mixed signs; the
    unaligned sum is kept as `pooled_raw` only to show what the old figure was measuring.
    Pairs whose denominator is smaller than `min_den` are excluded from the median (their
    ratio is noise over noise) and counted in `n_excluded`.
    """
    name = name or df.attrs.get("main")
    base = df["base"].to_numpy()
    alls = df["all_abl"].to_numpy()
    circs = df[f"circ_{name}"].to_numpy()
    num, den = circs - alls, base - alls
    keep = np.abs(den) > min_den
    per_pair = np.where(keep, num / np.where(keep, den, 1.0), np.nan)
    sden = float(np.abs(den).sum())
    return dict(
        faithfulness=float(np.nanmedian(per_pair)) if keep.any() else float("nan"),
        faithfulness_mean=float(np.nanmean(per_pair)) if keep.any() else float("nan"),
        pooled_signed=float((num * np.sign(den)).sum() / sden) if sden else float("nan"),
        pooled_raw=float(circs.sum() / base.sum()) if base.sum() else float("nan"),
        iqr=[float(np.nanpercentile(per_pair, 25)), float(np.nanpercentile(per_pair, 75))] if keep.any() else None,
        delta_base=float(base.mean()), delta_all_ablated=float(alls.mean()),
        delta_circuit_only=float(circs.mean()),
        n_den_positive=int((den > 0).sum()), n_den_negative=int((den < 0).sum()),
        n_excluded=int((~keep).sum()), min_den=min_den,
        circuit_size=df.attrs.get("sizes", {}).get(name), n_pairs=len(df),
    )


def k_curve(df: pd.DataFrame, min_den: float = 0.1) -> pd.DataFrame:
    """faithfulness and necessity at every circuit size present in `df`."""
    rows = []
    for name, size in sorted(df.attrs.get("sizes", {}).items(), key=lambda kv: kv[1]):
        if f"circ_{name}" not in df.columns:
            continue
        f = faithfulness(df, name, min_den)
        n = necessity(df, name)
        rows.append(dict(name=name, k=size, faithfulness=f["faithfulness"],
                         faithfulness_pooled_signed=f["pooled_signed"],
                         collapse=n["collapse"], collapse_median=n["collapse_median"],
                         delta_circuit_only=f["delta_circuit_only"]))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- specificity

def specificity_contrast(override: dict, accept: dict) -> dict:
    """Variant 1: the same collapse on pairs the model originally agreed with the tool on.

    Necessity alone cannot separate "this circuit decides whether to trust the tool" from
    "these are 50 generally load-bearing heads". `accept` pairs (tool_case=tool_echoed --
    the tool returned what the model had already computed, so the trace contains no
    conflict) are the control: a circuit specific to the trust decision should collapse
    much less there.

    Both arguments are `necessity()` outputs. The caveat, stated because a reviewer will
    find it: pair construction injects a conflicting value into every pair, accept ones
    included, so this contrasts the surrounding context rather than a genuinely
    conflict-free task. `specificity_damage` is the variant that does not have that
    problem.
    """
    o, a = override["collapse"], accept["collapse"]
    return dict(collapse_override=o, collapse_accept=a, ratio=float(a / o) if o else float("nan"),
                difference=float(o - a),
                n_override=override["n_pairs"], n_accept=accept["n_pairs"])


@torch.no_grad()
def specificity_damage(model, pairs, circuits: dict[str, list[Node]],
                       means_by_region: dict[str, dict], mode: str = "mean",
                       regions: tuple[str, ...] = ("preout", "postout"),
                       max_positions: int = 128, seed: int = 0) -> pd.DataFrame:
    """Variant 2: collateral damage from ablating the circuit, by region and against random.

    WHAT THE EARLIER VERSION GOT WRONG

    It ablated at post-output positions of the clean run and read a top-1 agreement of
    0.012 as a specificity failure. But post-output is where the model assimilates the
    tool result and writes its answer -- the circuit's own job. Removing the conflict does
    not make that region unrelated work, so the test asked whether the circuit matters for
    what the circuit does, and the answer was never going to be no.

    It also had no baseline. Necessity means something because random same-size sets
    collapse |Delta| by 0.06 against the circuit's 0.78. Damage had nothing to compare to,
    so 0.012 was uninterpretable: mean-ablating any 50 well-scoring components across
    hundreds of positions may drive top-1 to zero whatever they are.

    Both are fixed here. `regions` contrasts `preout` -- problem statement, reasoning and
    code, before any tool result exists -- with `postout`, and `circuits` carries random
    node sets alongside the real one. The claim this can support is that damage is
    concentrated at post-output and exceeds random there, not that ablating the circuit
    harms nothing.

    Run on clean ids: the control asks what the circuit does when there is nothing to
    distrust. Logits are sliced to the sampled positions immediately, so only one full
    [1, T, d_vocab] tensor is alive at a time.
    """
    rng = random.Random(seed)
    dev = {r: {k: v.to(model.cfg.device) for k, v in m.items()} for r, m in means_by_region.items()}
    rows = []
    for i, pair in enumerate(pairs):
        ids = to_batch(model, pair.clean_ids)
        for region in regions:
            pos = positions_of(pair, region)
            if len(pos) < 2:
                continue                       # nothing to measure (e.g. a truncated prefix)
            sample = sorted(rng.sample(pos, min(max_positions, len(pos))))
            base_lg = model(ids)[0, sample].float()
            lp_base = torch.log_softmax(base_lg, dim=-1)
            top_base = base_lg.argmax(-1)
            p_base = lp_base.exp()
            for name, nodes in circuits.items():
                fwd = replace_hooks(nodes, pos, mean_source(dev[region], pair.T, mode))
                lg = model.run_with_hooks(ids, fwd_hooks=fwd)[0, sample].float()
                kl = (p_base * (lp_base - torch.log_softmax(lg, dim=-1))).sum(-1)
                rows.append(dict(
                    pair_id=pair.id, circuit=name, region=region, n_sampled=len(sample),
                    top1_agreement=float((top_base == lg.argmax(-1)).float().mean()),
                    kl_mean=float(kl.mean()), kl_p95=float(kl.quantile(0.95)),
                ))
                del lg, kl
            del base_lg, lp_base, p_base, top_base
        if (i + 1) % 25 == 0:
            log.info("specificity_damage: %d/%d pairs", i + 1, len(pairs))
    return pd.DataFrame(rows)


def summarise_damage(df: pd.DataFrame) -> dict:
    """{region: {circuit: stats}}, plus the circuit-over-random ratio that is the result."""
    out: dict[str, dict] = {}
    for region, g in df.groupby("region"):
        per = {}
        for name, gg in g.groupby("circuit"):
            per[name] = dict(top1_agreement=float(gg["top1_agreement"].mean()),
                             top1_agreement_median=float(gg["top1_agreement"].median()),
                             kl_mean=float(gg["kl_mean"].mean()), n_pairs=int(gg["pair_id"].nunique()))
        rand = [v["top1_agreement"] for k, v in per.items() if k.startswith("random")]
        if "circuit" in per and rand:
            r = float(np.mean(rand))
            per["random_mean_top1"] = r
            # <1 means the circuit damages this region more than random sets of its size.
            per["circuit_over_random"] = float(per["circuit"]["top1_agreement"] / r) if r else float("nan")
        out[region] = per
    return out


# --------------------------------------------------------------------------- self-repair

def self_repair(model, pairs, circuit: list[Node], means, max_iters: int = 3, z: float = 2.0,
                mode: str = "mean", positions: str = "postout") -> dict:
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
            hooks = replace_hooks(current, positions_of(pair, positions),
                                  mean_source(means_dev, pair.T, mode))
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
                z_threshold=z, circuit_size_final=len(current), mode=mode, positions=positions)
