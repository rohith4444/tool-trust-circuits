#!/usr/bin/env python3
"""Stage 3 -- Universality: CKA between models' residual streams, node overlap between circuits.

    python scripts/run_universality.py --models torl simpletir --pairs pairs.jsonl \
        --circuits outputs/torl/circuit.json outputs/simpletir/circuit.json --out outputs/universality

Writes feats_<model>.pt, cka_<A>_<B>.npy for every model pair, overlap.json, universality_checks.json.
Models are loaded one at a time; rows are aligned across models by pair id.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from circuits.cli import add_common, get_pairs, setup_logging, write_json
from circuits.hooks import set_seed
from circuits.models.load import load_model
from circuits.universality.cka import cka_matrix, node_overlap, resid_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p, model_arg=False)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--circuits", nargs="*", default=[], help="circuit.json files, same order as --models")
    p.add_argument("--pool", choices=["mean", "position"], default="mean")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if len(set(args.models)) != len(args.models):
        raise SystemExit("--models must be distinct")
    feats: dict[str, tuple[torch.Tensor, list[str]]] = {}
    for key in args.models:
        model = load_model(key, device=args.device, dtype=args.dtype)
        _, pairs = get_pairs(model, args)
        f, ids = resid_features(model, pairs, pool=args.pool)
        torch.save(dict(features=f, row_ids=ids, model=key), out / f"feats_{key}.pt")
        feats[key] = (f, ids)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # align rows across models by id
    common = set.intersection(*(set(ids) for _, ids in feats.values()))
    aligned = {}
    for key, (f, ids) in feats.items():
        keep = [i for i, r in enumerate(ids) if r in common]
        aligned[key] = f[:, keep]

    first = next(iter(aligned))
    self_diag = np.diag(cka_matrix(aligned[first], aligned[first]))
    if not np.allclose(self_diag, 1.0, atol=1e-6):
        raise AssertionError(f"self-CKA diagonal != 1: {self_diag}")
    checks = dict(n_rows=len(common), pool=args.pool, self_cka_diag_min=float(self_diag.min()))

    written = []
    for a, b in itertools.combinations(aligned, 2):
        m = cka_matrix(aligned[a], aligned[b])
        path = out / f"cka_{a}_{b}.npy"
        np.save(path, m)
        written.append(str(path))
        checks[f"cka_{a}_{b}_max"] = float(m.max())

    overlaps = []
    circuits = [json.load(open(c)) for c in args.circuits]
    for (i, ca), (j, cb) in itertools.combinations(enumerate(circuits), 2):
        overlaps.append(node_overlap(ca, cb))
    if circuits:
        self_ov = node_overlap(circuits[0], circuits[0])["jaccard"]
        if abs(self_ov - 1.0) > 1e-9:
            raise AssertionError(f"self-overlap {self_ov} != 1")
        checks["self_overlap"] = self_ov
    write_json(out / "overlap.json", overlaps)
    write_json(out / "universality_checks.json", checks)
    print(f"wrote {len(written)} CKA matrices and overlap.json ({len(overlaps)} circuit pairs) to {out}")


if __name__ == "__main__":
    main()
