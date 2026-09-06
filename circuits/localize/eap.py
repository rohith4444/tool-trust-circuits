"""EAP-IG screen: node-level attribution via `hannamw/EAP-IG` (`eap.attribute_node`).

Library facts the wrapper respects (v2.0.0, commit pinned in envs/analysis.txt):
- asserts use_attn_result / use_split_qkv_input / use_hook_mlp_in / ungrouped GQA -> `eap_mode`
- dataloader yields (clean: list[str], corrupt: list[str], label); it tokenizes with
  `model.to_tokens(prepend_bos=True)` -> `eap_mode` shadows to_tokens so no BOS is added
  and tokenization equals `Pair.*_ids` (asserted by `check_tokenization`)
- metric(logits, clean_logits, input_lengths, label); we index `input_lengths - 1`,
  i.e. the last real token, which is `Pair.measure_pos`
- `method` is required, `ig_steps` must be given, 'EAP-IG-inputs' needs intervention='patching'
- returns None; scores live in `graph.nodes_scores` (needs `Graph.from_model(node_scores=True)`)
- `node.score` is a 0-dim tensor; scores accumulate in the model dtype (bf16 on the pod;
  pass --dtype float32 if memory allows and ranks look noisy)
- every buffer is allocated with device='cuda' hardcoded: this stage runs on the pod only
"""
from __future__ import annotations

import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..hooks import Node, all_nodes, eap_mode, node_name, parse_node

log = logging.getLogger(__name__)


class PairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        return p.clean_text, p.corrupt_text, [p.tool_tok, p.own_tok]


def collate(xs):
    clean, corrupt, labels = zip(*xs)
    return list(clean), list(corrupt), torch.tensor(labels)


def delta_metric(logits, clean_logits, input_lengths, labels, mean: bool = True, loss: bool = False):
    """Δ at the last real token of each item; labels [B, 2] = (tool_tok, own_tok)."""
    idx = torch.arange(logits.size(0), device=logits.device)
    last = logits[idx, input_lengths - 1].float()
    picked = last.gather(-1, labels.to(logits.device))
    res = picked[:, 0] - picked[:, 1]
    if loss:
        res = -res
    if mean:
        res = res.mean()
    return res


def check_tokenization(model, pairs) -> None:
    """EAP-IG re-tokenizes the strings; the result must equal the Pair's ids exactly."""
    from eap.utils import tokenize_plus
    for p in pairs:
        for text, ids in ((p.clean_text, p.clean_ids), (p.corrupt_text, p.corrupt_ids)):
            toks, _, lengths, _ = tokenize_plus(model, [text])
            got = toks[0, : int(lengths[0])].cpu()
            if not torch.equal(got, ids):
                raise RuntimeError(f"EAP-IG tokenization differs from Pair ids for {p.id}: "
                                   f"{got.shape[0]} vs {ids.shape[0]} tokens")


def _finish(df: pd.DataFrame, cfg, topn: int, method: str) -> pd.DataFrame:
    df["abs_score"] = df["score"].abs()
    df = df.sort_values("abs_score", ascending=False, kind="stable").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["in_topn"] = df["rank"] <= topn
    df["method"] = method
    check_scores(df, cfg)
    return df


def check_scores(df: pd.DataFrame, cfg) -> None:
    expected = {node_name(n) for n in all_nodes(cfg)}
    got = list(df["name"])
    if len(got) != len(set(got)) or set(got) != expected:
        raise AssertionError(f"node_scores incomplete/duplicated: {len(got)} rows vs {len(expected)} nodes")
    if not np.isfinite(df["score"].to_numpy()).all():
        raise AssertionError("non-finite EAP scores")
    if (df["score"] == 0).all():
        raise AssertionError("all EAP scores are zero: the backward hooks did not fire")


def eap_ig_scores(model, pairs, ig_steps: int = 5, topn: int = 50, out_dir: str | Path | None = None,
                  batch_size: int = 1) -> pd.DataFrame:
    """node_scores table: layer, type, index, name, score (signed), abs_score, rank, in_topn."""
    from eap.attribute_node import attribute_node
    from eap.graph import Graph

    loader = DataLoader(PairDataset(pairs), batch_size=batch_size, collate_fn=collate, shuffle=False)
    with eap_mode(model):
        check_tokenization(model, pairs)
        graph = Graph.from_model(model, node_scores=True)
        attribute_node(model, graph, loader, partial(delta_metric, mean=True, loss=False),
                       method="EAP-IG-inputs", intervention="patching", ig_steps=ig_steps, quiet=False)
        rows = []
        for name, node in graph.nodes.items():
            if name in ("input", "logits"):
                continue
            t, layer, idx = parse_node(name)
            rows.append(dict(layer=layer, type=t, index=idx, name=name, score=float(node.score)))
        if out_dir:
            graph.to_pt(str(Path(out_dir) / "graph.pt"))
    return _finish(pd.DataFrame(rows), model.cfg, topn, "eap-ig")


def shortlist(df: pd.DataFrame, topn: int) -> list[Node]:
    return [parse_node(n) for n in df["name"].iloc[:topn]]
