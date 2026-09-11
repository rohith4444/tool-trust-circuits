"""The metric: Δ = logit(tool answer) − logit(own answer) at the measure position.

Δ is read on the corrupt run (tool ≠ own). Differentiable, so EAP-IG can backprop it.
The clean reference Δ_clean_ref is the plain clean run: patching every head and MLP
at post-output positions from clean reproduces the clean residual stream there exactly,
so the onboarding plan's "corrupt run with everything patched" is the same number and
is kept as the plumbing check in patch.py instead.
"""
from __future__ import annotations

from typing import Sequence

import torch

from ..hooks import to_batch


def delta(logits: torch.Tensor, own_tok: int, tool_tok: int, measure_pos: int) -> torch.Tensor:
    last = logits[0, measure_pos].float()
    return last[tool_tok] - last[own_tok]


def delta_curve(logits: torch.Tensor, own_tok: int, tool_tok: int, positions: Sequence[int]) -> torch.Tensor:
    """Δ at every position in `positions`. Diagnostic only: before measure_pos the
    logits predict other tokens, so the curve shows when the preference becomes readable."""
    sel = logits[0, list(positions)].float()
    return sel[:, tool_tok] - sel[:, own_tok]


@torch.no_grad()
def run_delta(model, ids: torch.Tensor, pair, fwd_hooks: Sequence = ()) -> float:
    logits = model.run_with_hooks(to_batch(model, ids), fwd_hooks=list(fwd_hooks))
    return delta(logits, pair.own_tok, pair.tool_tok, pair.measure_pos).item()


@torch.no_grad()
def baseline(model, pair) -> dict:
    """Δ, top-1 token and Δ-curve endpoints for both runs of a pair."""
    out = {"id": pair.id, "label": pair.meta.get("label"), "T": pair.T}
    for run, ids in (("clean", pair.clean_ids), ("corrupt", pair.corrupt_ids)):
        logits = model(to_batch(model, ids))
        out[f"delta_{run}"] = delta(logits, pair.own_tok, pair.tool_tok, pair.measure_pos).item()
        top = int(logits[0, pair.measure_pos].argmax())
        out[f"top1_{run}"] = model.tokenizer.decode([top])
        out[f"top1_{run}_is"] = "tool" if top == pair.tool_tok else ("own" if top == pair.own_tok else "other")
        curve = delta_curve(logits, pair.own_tok, pair.tool_tok, pair.post_output)
        out[f"curve_first_{run}"] = curve[0].item()
    return out


def check_sign_flip(model, pair, atol: float = 1e-4) -> None:
    d = run_delta(model, pair.corrupt_ids, pair)
    swapped = type(pair)(**{**pair.__dict__, "own_tok": pair.tool_tok, "tool_tok": pair.own_tok})
    d_sw = run_delta(model, pair.corrupt_ids, swapped)
    if abs(d + d_sw) > atol:
        raise AssertionError(f"delta does not flip sign on token swap: {d} vs {d_sw}")
