"""Role labels for circuit nodes: attention role, direct logit attribution, depth.

DLA with `fold_ln=False` on RMSNorm (Qwen): logits = (resid / scale) * ln_final.w @ W_U,
where `scale = sqrt(mean(resid²) + eps)` is cached at `ln_final.hook_scale`. Every
residual-stream term (head output z·W_O, MLP output, embedding, attention biases) is
projected the same way, so the terms sum to Δ exactly (up to dtype) -- the check
`dla_terms(...)["delta_check"]` relies on that. TL's `apply_ln_to_stack` divides by the
scale but never applies `ln_final.w`, which is why this is done by hand.
"""
from __future__ import annotations

import logging

import torch
from einops import einsum

from ..hooks import Node, cache_acts, depth_label, node_name

log = logging.getLogger(__name__)


def dla_direction(model, pair) -> torch.Tensor:
    """Unembed direction for Δ, with the final-norm gain folded in. [d_model] fp32."""
    W_U = model.W_U
    d = (W_U[:, pair.tool_tok] - W_U[:, pair.own_tok]).float()
    if hasattr(model.ln_final, "w"):
        d = d * model.ln_final.w.float()
    return d


def _project(term: torch.Tensor, direction: torch.Tensor, scale: torch.Tensor, cfg) -> torch.Tensor:
    if cfg.normalization_type in ("LN", "LNPre"):
        term = term - term.mean(-1, keepdim=True)
    return (term @ direction) / scale


@torch.no_grad()
def dla_terms(model, cache, pair, scale: torch.Tensor | None = None) -> dict:
    """Per-head [L, H], per-MLP [L], embedding and bias DLA at measure_pos, plus a Δ check.

    `scale` lets a caller hold the final-norm scale at a base run's value while
    reading DLA from an ablated run (self-repair).
    """
    cfg = model.cfg
    mp = pair.measure_pos
    direction = dla_direction(model, pair)
    if scale is None:
        scale = cache["ln_final.hook_scale"][0, mp].float()
    scale = scale.reshape(())
    heads, mlps = [], []
    bias = torch.zeros((), device=direction.device)
    for layer in range(cfg.n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"][0, mp].float()                      # [H, d_head]
        head_out = einsum(z, model.blocks[layer].attn.W_O.float(), "h d, h d m -> h m")  # [H, d_model]
        heads.append(_project(head_out, direction, scale, cfg))
        mlps.append(_project(cache[f"blocks.{layer}.hook_mlp_out"][0, mp].float(), direction, scale, cfg))
        bias = bias + _project(model.blocks[layer].attn.b_O.float(), direction, scale, cfg)
    heads_t = torch.stack(heads)                                                       # [L, H]
    mlps_t = torch.stack(mlps)                                                         # [L]
    embed = _project(cache["hook_embed"][0, mp].float(), direction, scale, cfg)
    if hasattr(model.unembed, "b_U"):
        bias = bias + (model.unembed.b_U[pair.tool_tok] - model.unembed.b_U[pair.own_tok]).float()
    total = heads_t.sum() + mlps_t.sum() + embed + bias
    return dict(heads=heads_t.cpu(), mlps=mlps_t.cpu(), embed=embed.item(), bias=bias.item(),
                total=total.item(), scale=scale.detach().cpu())


def check_dla_sum(model, pair, tol_rel: float = 0.05, tol_abs: float = 0.1) -> dict:
    from .delta import delta
    logits, cache = cache_acts(model, pair.corrupt_ids)
    d = delta(logits, pair.own_tok, pair.tool_tok, pair.measure_pos).item()
    t = dla_terms(model, cache, pair)
    err = abs(t["total"] - d)
    ok = err <= tol_abs + tol_rel * abs(d)
    out = dict(pair=pair.id, delta=d, dla_total=t["total"], abs_err=err, ok=ok)
    if not ok:
        raise AssertionError(f"DLA terms do not sum to Δ: {out}")
    log.info("DLA sum check ok: %s", out)
    return out


def attention_mass_hooks(pair, layers, store: dict):
    """Reducing hooks on hook_pattern: mean over post-output queries of attention mass
    to the output span and to the own-answer spans. Stores [H] tensors per layer;
    never keeps a full [H, T, T] pattern."""
    q = torch.as_tensor(pair.post_output)
    t0, t1 = pair.output_span

    def make(layer):
        def fn(pattern, hook):
            p = pattern[0, :, q]                                    # [H, Q, T]
            tool = p[:, :, t0:t1].sum(-1).mean(-1)                  # [H]
            own = torch.zeros_like(tool)
            for a, b in pair.own_spans:
                own = own + p[:, :, a:b].sum(-1).mean(-1)
            store.setdefault(layer, []).append(torch.stack([tool, own]).float().cpu())
            return pattern
        return fn
    return [(f"blocks.{layer}.attn.hook_pattern", make(layer)) for layer in layers]


def attn_role(tool_mass: float, own_mass: float, ratio: float = 1.5, min_mass: float = 0.02) -> str:
    if max(tool_mass, own_mass) < min_mass:
        return "neither"
    if tool_mass >= ratio * own_mass:
        return "tool_reader"
    if own_mass >= ratio * tool_mass:
        return "own_reader"
    return "neither"


@torch.no_grad()
def label_roles(model, pairs, nodes: list[Node], ratio: float = 1.5, min_mass: float = 0.02) -> dict[Node, dict]:
    """One forward per pair (corrupt run): attention masses via reducing hooks, DLA from the filtered cache."""
    layers = sorted({n[1] for n in nodes if n[0] == "a"})
    masses: dict[int, list] = {}
    dla_heads = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    dla_mlps = torch.zeros(model.cfg.n_layers)
    for pair in pairs:
        _, cache = cache_acts(model, pair.corrupt_ids, fwd_hooks=attention_mass_hooks(pair, layers, masses))
        t = dla_terms(model, cache, pair)
        dla_heads += t["heads"]
        dla_mlps += t["mlps"]
    n = len(pairs)
    dla_heads /= n
    dla_mlps /= n
    mass = {layer: torch.stack(v).mean(0) for layer, v in masses.items()}   # [2, H]
    out = {}
    for node in nodes:
        t, layer, idx = node
        rec = dict(name=node_name(node), depth=depth_label(layer, model.cfg.n_layers))
        if t == "a":
            tool_m, own_m = mass[layer][0, idx].item(), mass[layer][1, idx].item()
            rec.update(attn_role=attn_role(tool_m, own_m, ratio, min_mass), tool_mass=tool_m, own_mass=own_m,
                       dla=dla_heads[layer, idx].item())
        else:
            rec.update(attn_role="n/a", tool_mass=None, own_mass=None, dla=dla_mlps[layer].item())
        rec["dla_sign"] = "tool" if rec["dla"] > 0 else "own"
        out[node] = rec
    return out
