"""Head-scaling: turn the circuit's tool-reading components up and its own-answer ones down.

Reads `circuit.json` from localization and multiplies each node's output at the
post-output positions by a factor that depends on its role:

    dla_sign == "tool"  ->  alpha  (> 1, amplify)
    dla_sign == "own"   ->  beta   (< 1, attenuate)

Training-free and token-free like steering, but surgical rather than broad: it needs the
circuit, and it says something sharper if it works -- not "nudging the stream helps" but
"these specific heads carry the decision, and re-weighting them changes it".

A NOTE ON WHERE THE ROLES CAME FROM

`label_roles()` computes DLA and attention masses on the CORRUPT run, because that is
what localization reads Δ on. We intervene on the CLEAN run, since that is the real
failure. A head's function should not change between the two -- only its activation does
-- but that is an assumption, so `role_transfer()` recomputes DLA signs on clean runs and
reports how often they agree. If agreement is poor the policy should be rebuilt from
clean-run DLA instead of trusted from the file.

WHY MULTIPLY z AND NOT THE HEAD'S CONTRIBUTION

Scaling at `hook_z` (before W_O) scales that head's contribution to the residual stream
by exactly the same factor, since W_O is linear. Same effect, and it keeps heads separable
-- past W_O they are summed and no longer addressable individually.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from ..hooks import Node, cache_acts, node_name, scale_hooks
from ..localize.delta import run_delta
from ..localize.roles import dla_terms
from .common import dedupe, run_ids

log = logging.getLogger(__name__)

NEUTRAL = ("neither", "n/a", None)


def load_circuit(path: str | Path) -> dict:
    with open(path) as f:
        circuit = json.load(f)
    if not circuit.get("nodes"):
        raise ValueError(f"{path} has no nodes")
    return circuit


def select(circuit: dict, confirmed_only: bool = True) -> list[dict]:
    """Node records to intervene on.

    `circuit.json` holds the whole top-N EAP-IG shortlist; `above_noise` marks the ones
    activation patching actually confirmed against the random-node floor. Default to the
    confirmed set -- the screen is an estimate and the unconfirmed remainder is exactly
    the false positives patching was run to remove.
    """
    nodes = circuit["nodes"]
    if confirmed_only:
        nodes = [n for n in nodes if n.get("above_noise")]
    if not nodes:
        raise ValueError("no nodes selected; try confirmed_only=False and check patch_results.csv")
    return nodes


def node_of(rec: dict) -> Node:
    return (rec["type"], int(rec["layer"]), int(rec["index"]))


def policy(nodes: list[dict], alpha: float, beta: float, key: str = "dla_sign") -> dict[Node, float]:
    """Per-node multipliers from a role field.

    `dla_sign` splits every node -- heads and MLPs -- by which answer its direct logit
    contribution favours. `attn_role` splits heads by where they attend (tool output vs
    the model's own answer in the prefix) and leaves MLPs untouched, since attention role
    is undefined for them. The two are different claims about what a component is doing,
    so running both is a real comparison rather than a parameter tweak.

    Nodes whose role is neutral get 1.0 and are left in the returned dict on purpose: a
    factor of 1 is an exact no-op, and keeping them makes the hook set identical across
    policies so any difference is the factors, not the coverage.
    """
    if key not in ("dla_sign", "attn_role"):
        raise ValueError(f"unknown policy key {key!r}")
    up = "tool" if key == "dla_sign" else "tool_reader"
    down = "own" if key == "dla_sign" else "own_reader"
    scales: dict[Node, float] = {}
    for rec in nodes:
        role = rec.get(key)
        scales[node_of(rec)] = alpha if role == up else (beta if role == down else 1.0)
    return scales


def policy_counts(scales: dict[Node, float], alpha: float, beta: float) -> dict:
    return dict(n_up=sum(v == alpha for v in scales.values()),
                n_down=sum(v == beta for v in scales.values()),
                n_neutral=sum(v == 1.0 for v in scales.values()), n_total=len(scales))


@torch.no_grad()
def scaled_delta(model, pair, scales: dict[Node, float], run: str = "clean") -> float:
    return run_delta(model, run_ids(pair, run), pair,
                     scale_hooks(scales, pair.post_output))


@torch.no_grad()
def sweep(model, pairs, nodes: list[dict], alphas, betas, key: str = "dla_sign",
          run: str = "clean") -> list[dict]:
    """Mean Δ over `pairs` for each (alpha, beta). (1, 1) is the untouched baseline.

    Deduplicated: both Z-variants of a trace share a clean run, so without this half the
    forwards are byte-identical repeats.
    """
    pairs = dedupe(pairs, run)
    base = [run_delta(model, run_ids(p, run), p) for p in pairs]
    base_mean = float(sum(base) / len(base))
    rows = []
    for alpha in alphas:
        for beta in betas:
            if alpha == 1.0 and beta == 1.0:
                ds = base
            else:
                scales = policy(nodes, alpha, beta, key)
                ds = [scaled_delta(model, p, scales, run) for p in pairs]
            mean = float(sum(ds) / len(ds))
            rows.append(dict(key=key, alpha=alpha, beta=beta, delta_mean=mean,
                             delta_shift=mean - base_mean,
                             frac_positive=float(sum(d > 0 for d in ds) / len(ds)),
                             n_pairs=len(ds)))
    return rows


@torch.no_grad()
def role_transfer(model, pairs, nodes: list[dict], run: str = "clean") -> dict:
    """Do the file's DLA signs still hold on the run we intervene on?

    circuit.json's `dla_sign` was measured on corrupt runs. If a head favours the truth
    there but the model's own answer here, scaling it up pushes the wrong way, and the
    sweep would report that as "head-scaling does not work" rather than "the policy is
    inverted". Cheap to check: one forward per pair.
    """
    pairs = dedupe(pairs, run)
    heads = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    mlps = torch.zeros(model.cfg.n_layers)
    for pair in pairs:
        _, cache = cache_acts(model, run_ids(pair, run))
        t = dla_terms(model, cache, pair)
        heads += t["heads"]
        mlps += t["mlps"]
    heads /= len(pairs)
    mlps /= len(pairs)

    rows, agree = [], 0
    for rec in nodes:
        t, layer, idx = node_of(rec)
        dla_now = float(heads[layer, idx] if t == "a" else mlps[layer])
        sign_now = "tool" if dla_now > 0 else "own"
        same = sign_now == rec.get("dla_sign")
        agree += same
        rows.append(dict(name=node_name((t, layer, idx)), dla_file=rec.get("dla"),
                         dla_clean=dla_now, sign_file=rec.get("dla_sign"),
                         sign_clean=sign_now, agrees=same))
    out = dict(n_nodes=len(nodes), n_agree=agree, frac_agree=agree / max(len(nodes), 1),
               run=run, rows=rows)
    log.info("DLA sign transfer corrupt->%s: %d/%d agree", run, agree, len(nodes))
    return out


@torch.no_grad()
def identity_check(model, pair, nodes: list[dict], atol: float = 1e-3) -> None:
    """alpha = beta = 1 must reproduce the untouched Δ exactly.

    Otherwise the hook is touching something -- wrong positions, a dtype round trip --
    and every swept number is measuring that instead of the intervention.
    """
    base = run_delta(model, pair.clean_ids, pair)
    got = scaled_delta(model, pair, policy(nodes, 1.0, 1.0))
    if abs(got - base) > atol:
        raise AssertionError(f"scale hook is not an identity at alpha=beta=1: {got} vs {base}")
    log.info("scale identity check ok (delta %.4f, %d nodes)", base, len(nodes))
