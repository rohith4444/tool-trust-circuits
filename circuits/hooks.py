"""Shared hook plumbing: node ids, hook names, filtered caching, replace hooks, EAP mode.

Node = (type, layer, index): ("a", L, H) for attention head H of layer L, ("m", L, 0)
for the MLP of layer L. String form matches EAP-IG's naming: "a{L}.h{H}" / "m{L}".

Heads are read / overwritten at `blocks.{L}.attn.hook_z` (per-head, d_head wide) and
MLPs at `blocks.{L}.hook_mlp_out`. Working at hook_z rather than attn.hook_result
avoids `use_attn_result`, which materialises a [pos, n_heads, d_model] tensor per layer.
"""
from __future__ import annotations

import contextlib
import random
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

Node = tuple[str, int, int]


def node_name(node: Node) -> str:
    t, layer, idx = node
    return f"a{layer}.h{idx}" if t == "a" else f"m{layer}"


def parse_node(name: str) -> Node:
    if name.startswith("a"):
        layer, head = name[1:].split(".h")
        return ("a", int(layer), int(head))
    if name.startswith("m"):
        return ("m", int(name[1:]), 0)
    raise ValueError(f"not a node name: {name!r}")


def all_nodes(cfg) -> list[Node]:
    out: list[Node] = []
    for layer in range(cfg.n_layers):
        out.extend(("a", layer, h) for h in range(cfg.n_heads))
        out.append(("m", layer, 0))
    return out


def hook_name(node: Node) -> str:
    t, layer, _ = node
    return f"blocks.{layer}.attn.hook_z" if t == "a" else f"blocks.{layer}.hook_mlp_out"


def is_head_hook(name: str) -> bool:
    return name.endswith("attn.hook_z")


def depth_label(layer: int, n_layers: int) -> str:
    third = n_layers / 3
    return "early" if layer < third else ("mid" if layer < 2 * third else "late")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- caching

CACHE_ALWAYS = ("ln_final.hook_scale", "hook_embed")


def cache_filter(name: str) -> bool:
    return name.endswith("attn.hook_z") or name.endswith("hook_mlp_out") or name in CACHE_ALWAYS


def to_batch(model, ids: torch.Tensor) -> torch.Tensor:
    ids = ids.to(model.cfg.device)
    return ids[None] if ids.dim() == 1 else ids


@torch.no_grad()
def cache_acts(model, ids: torch.Tensor, fwd_hooks: Sequence = (), extra: Callable[[str], bool] | None = None):
    """run_with_cache restricted to hook_z / hook_mlp_out / ln_final scale / embed.

    Default caching would keep hook_pattern and hook_attn_scores too (about 50 GB on a
    7B at 4k tokens). `fwd_hooks` run during the same forward, so a caller can add
    reducing hooks (attention masses) or ablation hooks and cache the result.
    """
    flt = cache_filter if extra is None else (lambda n: cache_filter(n) or extra(n))
    with model.hooks(fwd_hooks=list(fwd_hooks)):
        logits, cache = model.run_with_cache(to_batch(model, ids), names_filter=flt)
    return logits, cache


# --------------------------------------------------------------------------- overwrite hooks

def replace_hooks(nodes: Iterable[Node], positions: Sequence[int], source: Callable[[str], torch.Tensor | None]):
    """Forward hooks overwriting `nodes` at `positions` with `source(hook_name)`.

    `source` returns a tensor with the activation's shape ([1, T, H, d_head] for
    hook_z, [1, T, d_model] for hook_mlp_out), or None for zeros. Nodes on the same
    hook share one hook function so an ablation of 800 nodes costs 56 hooks, not 800.
    """
    by_hook: dict[str, list[int]] = {}
    for n in nodes:
        by_hook.setdefault(hook_name(n), []).append(n[2])
    pos_list = list(positions)
    hooks = []
    for hname, idx in by_hook.items():
        def fn(act, hook, hname=hname, idx=sorted(set(idx))):
            if not pos_list:
                return act
            pos = torch.as_tensor(pos_list, device=act.device)
            src = source(hname)
            if is_head_hook(hname):
                heads = torch.as_tensor(idx, device=act.device)
                sel = (slice(None), pos[:, None], heads[None, :], slice(None))
            else:
                sel = (slice(None), pos, slice(None))
            act[sel] = 0 if src is None else src[sel].to(act.dtype)
            return act
        hooks.append((hname, fn))
    return hooks


def cache_source(cache) -> Callable[[str], torch.Tensor]:
    """Source for patching: the activation of the same hook from another run."""
    return lambda hname: cache[hname]


def mean_source(means: dict[str, torch.Tensor], T: int, mode: str = "mean"):
    """Source for ablation: per-node means ([H, d_head] / [d_model]) broadcast over positions, or zeros."""
    if mode == "zero":
        return lambda hname: None
    if mode != "mean":
        raise ValueError(f"unknown ablation mode {mode!r}")
    return lambda hname: means[hname][None, None].expand(1, T, *means[hname].shape)


# --------------------------------------------------------------------------- EAP-IG mode

EAP_FLAGS = ("use_split_qkv_input", "use_attn_result", "use_hook_mlp_in", "ungroup_grouped_query_attention")


@contextlib.contextmanager
def eap_mode(model):
    """Model configuration `eap.attribute_node` asserts, restored on exit.

    - The four cfg flags (attn_result / split_qkv / mlp_in / ungrouped GQA) cost memory,
      so they are only on for the EAP stage.
    - EAP-IG's interpolation hook keeps the autograd graph alive through the embedding
      output (`corrupt + activations * 0`), so `embed.W_E` must require grad. Every other
      parameter stays frozen: no 15 GB of weight grads, no saved q/k/v inputs.
    - EAP-IG tokenizes with `model.to_tokens(..., prepend_bos=True)`, which for Qwen
      manually prepends `<|endoftext|>`. The traces never had one, so `to_tokens` is
      shadowed to force `prepend_bos=False`; tokenization then equals `Pair.*_ids`.
    """
    old_flags = {f: getattr(model.cfg, f) for f in EAP_FLAGS}
    for f in EAP_FLAGS:
        setattr(model.cfg, f, True)
    w_e = model.embed.W_E
    old_req = w_e.requires_grad
    w_e.requires_grad_(True)
    orig_to_tokens = model.to_tokens

    def to_tokens_nobos(input, prepend_bos=None, **kw):
        return orig_to_tokens(input, prepend_bos=False, **kw)

    model.to_tokens = to_tokens_nobos
    try:
        yield model
    finally:
        del model.to_tokens
        w_e.requires_grad_(old_req)
        w_e.grad = None
        for f, v in old_flags.items():
            setattr(model.cfg, f, v)
