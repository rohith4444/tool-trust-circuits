"""Difference-of-means steering: a "trust the tool" direction added to the residual stream.

    direction = mean(resid | accept) - mean(resid | override)

read at the post-output positions, then added back there at inference. Training-free and
token-free: no weights change and no control tokens enter the prompt, which is the whole
claim against ATTC.

WHICH RUN

Fitted and applied on the CLEAN run, not the corrupt one. Localization reads Δ on corrupt
because that is the perturbed condition it wants to explain. The intervention targets the
real failure instead: the tool reports the truth Y and the model writes its own wrong
answer X anyway. That is the clean run, and for `tool_right` pairs it is not a
counterfactual -- the tool really did say Y and the model really did override it, so the
validation split is a set of genuine Tool-Ignored cases.

HOW accept / override ARE DEFINED

Not from `tool_case`. That label describes the ORIGINAL trace, but the clean text
replaces the tool's value with Y, so for a `tool_echoed` trace the original tool said X
while the run in front of us says Y -- the label no longer describes what the model
faces. Only `tool_right` survives that substitution unchanged.

So the split is measured on the run itself:

    accept    Δ_clean > 0    the model prefers the tool's value
    override  Δ_clean < 0    the model prefers its own answer

The obvious objection is that a direction fitted to separate high-Δ from low-Δ residuals
must raise Δ by construction. That is why it is fitted on DISCOVERY and reported on
VALIDATION: if it moves Δ on held-out real Tool-Ignored cases it is a direction, not a
fit. `direction_from_labels` builds the same vector from the `tool_case` split as a
robustness check -- if the two agree by cosine the choice does not matter.
"""
from __future__ import annotations

import logging

import torch

from ..hooks import cache_acts, steer_hooks
from ..localize.delta import delta, run_delta
from .common import dedupe, run_ids

log = logging.getLogger(__name__)


def resid_mean_hooks(pair, layers, store: dict):
    """Reducing hooks: mean residual over the post-output positions, per layer.

    Stores [d_model] per layer instead of caching [T, d_model] for every layer -- the
    full stream is ~800 MB per forward on a 7B at 4k tokens, and every position outside
    post_output is discarded anyway. Same shape as roles.attention_mass_hooks.
    """
    pos = torch.as_tensor(pair.post_output)

    def make(layer):
        def fn(act, hook):
            store[layer] = act[0, pos].float().mean(0).cpu()
            return act
        return fn
    return [(f"blocks.{layer}.hook_resid_post", make(layer)) for layer in layers]


@torch.no_grad()
def collect(model, pairs, run: str = "clean") -> dict:
    """Per-run Δ and per-layer mean residual at the post-output positions.

    Deduplicated by the exact token ids. Both Z-variants of a trace share one
    `clean_text` -- Z appears only in the corrupt text -- so a ToRL discovery file of 712
    pairs holds just 356 distinct clean runs. Without this the fit would spend half its
    forwards on byte-identical repeats and report double the population sizes. On corrupt
    runs every pair differs, so the dedup is a no-op there.

    Returns resid [n_runs, n_layers, d_model] fp32 on CPU, deltas [n_runs], and the
    ids/labels so the caller can slice.
    """
    layers = list(range(model.cfg.n_layers))
    resid, deltas, ids, cases = [], [], [], []
    for pair in dedupe(pairs, run):
        ids_t = run_ids(pair, run)
        store: dict[int, torch.Tensor] = {}
        # No `extra=` here on purpose. Forward hooks fire regardless of the cache's
        # names_filter, so the reducing hooks above already have the residual stream;
        # adding resid_post to the filter as well would cache all [T, d_model] of it
        # per layer -- ~800 MB a forward at 4k tokens -- and then throw it away.
        if not len(pair.post_output):
            raise ValueError(f"pair {pair.id} has no post-output positions to average over")
        logits, _ = cache_acts(model, ids_t, fwd_hooks=resid_mean_hooks(pair, layers, store))
        deltas.append(delta(logits, pair.own_tok, pair.tool_tok, pair.measure_pos).item())
        resid.append(torch.stack([store[layer] for layer in layers]))
        ids.append(pair.id)
        cases.append(pair.meta.get("tool_case"))
    out = torch.stack(resid)
    # A NaN here would propagate into the direction and out into every swept Δ, without
    # raising anything -- it would just look like the intervention doing nothing.
    if not torch.isfinite(out).all():
        raise ValueError("non-finite residuals collected; check post_output and dtype")
    return dict(resid=out, deltas=torch.tensor(deltas), ids=ids, cases=cases)


def _difference(resid: torch.Tensor, accept_mask: torch.Tensor, override_mask: torch.Tensor) -> dict:
    """mean(accept) - mean(override) per layer, unit-normalised.

    The direction is normalised so the sweep's alpha is in absolute residual units and
    comparable across layers, whose raw norms differ by an order of magnitude. The
    pre-normalisation norm is reported: a direction whose norm is tiny relative to the
    residual it is added to is unlikely to do anything at a sane alpha.
    """
    n_acc, n_ovr = int(accept_mask.sum()), int(override_mask.sum())
    if n_acc == 0 or n_ovr == 0:
        raise ValueError(f"need both populations, got accept={n_acc} override={n_ovr}")
    d = resid[accept_mask].mean(0) - resid[override_mask].mean(0)         # [n_layers, d_model]
    norm = d.norm(dim=-1)                                                  # [n_layers]
    # Mean of the norms, not the norm of the mean: residuals point in different
    # directions across runs, so averaging first shrinks the vector and would make the
    # direction look larger relative to the stream than it is. This number is only used
    # to pick sane alphas, and a wrong one picks wrong alphas.
    typical = resid.norm(dim=-1).mean(0)                                   # [n_layers]
    return dict(direction=d / norm[:, None].clamp_min(1e-8), raw_norm=norm,
                resid_norm=typical, n_accept=n_acc, n_override=n_ovr)


def direction_from_delta(collected: dict, split: str = "sign") -> dict:
    """Split the runs by Δ and difference their residuals.

    `sign` is the principled cut -- Δ > 0 means the model actually prefers the tool's
    value. But in a clean run the tool reports the truth and these models mostly copy it,
    so the override group can come out small and the direction noisy. `median` splits at
    the median Δ instead, guaranteeing balanced halves at the cost of calling some
    weakly-positive runs "override". Check n_accept / n_override before trusting `sign`.
    """
    d = collected["deltas"]
    if split == "sign":
        acc, ovr = d > 0, d < 0
    elif split == "median":
        med = d.median()
        acc, ovr = d > med, d <= med
    else:
        raise ValueError(f"unknown split {split!r}")
    return _difference(collected["resid"], acc, ovr) | dict(split=f"delta_{split}")


def direction_from_labels(collected: dict) -> dict:
    """Split on `tool_case`: tool_echoed = accept, tool_right/tool_other = override.

    Robustness check only -- see the module docstring on why the label no longer matches
    what the model faces once the tool value is replaced.
    """
    cases = collected["cases"]
    acc = torch.tensor([c == "tool_echoed" for c in cases])
    ovr = torch.tensor([c in ("tool_right", "tool_other") for c in cases])
    return _difference(collected["resid"], acc, ovr) | dict(split="tool_case")


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-layer cosine between two [n_layers, d_model] directions."""
    return torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)


@torch.no_grad()
def steered_delta(model, pair, layer: int, vec: torch.Tensor, alpha: float, run: str = "clean") -> float:
    hooks = steer_hooks(layer, vec.to(model.cfg.device), pair.post_output, alpha)
    return run_delta(model, run_ids(pair, run), pair, hooks)


@torch.no_grad()
def sweep(model, pairs, direction: torch.Tensor, layers, alphas, run: str = "clean") -> list[dict]:
    """Mean Δ over `pairs` for each (layer, alpha). alpha 0 is the untouched baseline.

    Cost is len(layers) * len(alphas) * len(pairs) forwards, so sweep layers at one
    alpha first and only then sweep alpha at the winning layer -- the grid is separable
    enough in practice and the full product is an hour where the two passes are minutes.
    """
    pairs = dedupe(pairs, run)
    base = [run_delta(model, run_ids(p, run), p) for p in pairs]
    base_mean = float(sum(base) / len(base))
    rows = []
    for layer in layers:
        vec = direction[layer]
        for alpha in alphas:
            if alpha == 0:
                ds = base
            else:
                ds = [steered_delta(model, p, layer, vec, alpha, run) for p in pairs]
            mean = float(sum(ds) / len(ds))
            rows.append(dict(layer=layer, alpha=alpha, delta_mean=mean,
                             delta_shift=mean - base_mean,
                             frac_positive=float(sum(d > 0 for d in ds) / len(ds)),
                             n_pairs=len(ds)))
    return rows


@torch.no_grad()
def identity_check(model, pair, layer: int, d_model: int, atol: float = 1e-3) -> None:
    """A zero vector and alpha=0 must both reproduce the untouched Δ exactly.

    If this fails the hook is wrong -- wrong positions, wrong layer, dtype truncation --
    and every number downstream is measuring the bug instead of the intervention.
    """
    base = run_delta(model, pair.clean_ids, pair)
    zero = torch.zeros(d_model, device=model.cfg.device)
    for name, d in (("zero vector", steered_delta(model, pair, layer, zero, 1.0)),
                    ("alpha=0", steered_delta(model, pair, layer, torch.ones(d_model), 0.0))):
        if abs(d - base) > atol:
            raise AssertionError(f"steering hook is not an identity with {name}: {d} vs {base}")
    log.info("steering identity check ok at layer %d (delta %.4f)", layer, base)
