#!/usr/bin/env python3
"""Stage 1 -- Circuit localization: EAP-IG screen -> activation patching -> role labels.

    python scripts/run_localize.py --model torl --pairs pairs.jsonl --out outputs/torl

Writes to --out:
  pairs_summary.csv   Δ_clean / Δ_corrupt / top-1 token per pair
  node_scores.csv     every head and MLP, ranked by |EAP-IG score|
  patch_results.csv   recovery per shortlisted node (all / post-output / reverse) + noise floor
  circuit.json        role-labelled circuit; hand-off file for Interventions
  localize_checks.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from circuits.cli import add_common, get_pairs, push_outputs, setup_logging, write_json
from circuits.hooks import node_name, set_seed
from circuits.localize.delta import baseline, check_sign_flip
from circuits.localize.eap import eap_ig_scores, shortlist
from circuits.localize.patch import patch_stage
from circuits.localize.roles import check_dla_sum, label_roles
from circuits.models.load import load_model

CIRCUIT_FIELDS = ("layer", "type", "index", "name", "eap_score", "eap_rank", "rec_all", "rec_postout", "rec_reverse",
                  "above_noise", "attn_role", "tool_mass", "own_mass", "dla", "dla_sign", "depth")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    p.add_argument("--ig-steps", type=int, default=5)
    p.add_argument("--topn", type=int, default=50, help="shortlist size from the screen")
    p.add_argument("--n-draws", type=int, default=20, help="random single-node draws for the noise floor")
    p.add_argument("--eap-max-tokens", type=int, default=2000,
                   help="left-truncate pairs for the EAP stage only (its backward does not fit at 4k on 80 GB)")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    examples, pairs = get_pairs(model, args)
    checks: dict = {"n_pairs": len(pairs)}

    # 1a metric
    check_sign_flip(model, pairs[0])
    checks["sign_flip"] = "ok"
    summary = pd.DataFrame([baseline(model, pr) for pr in pairs])
    summary.to_csv(out / "pairs_summary.csv", index=False)

    # 1b EAP-IG screen
    _, eap_pairs = get_pairs(model, args, examples, max_tokens=args.eap_max_tokens)
    checks["eap_pairs"] = len(eap_pairs)
    scores = eap_ig_scores(model, eap_pairs, ig_steps=args.ig_steps, topn=args.topn, out_dir=out)
    scores.to_csv(out / "node_scores.csv", index=False)
    nodes = shortlist(scores, args.topn)
    checks["screen"] = dict(method=str(scores["method"].iloc[0]), n_nodes=len(scores), topn=len(nodes))

    # 1c patching + controls
    patch_df, patch_info = patch_stage(model, pairs, nodes, n_draws=args.n_draws, seed=args.seed)
    patch_df.to_csv(out / "patch_results.csv", index=False)
    checks["patching"] = patch_info["checks"]
    checks["noise_p95"] = patch_info["noise_p95"]

    # 1d roles
    checks["dla_sum"] = check_dla_sum(model, pairs[0])
    roles = label_roles(model, pairs, nodes)
    score_by = {r["name"]: r for r in scores.to_dict("records")}
    patch_by = {r["name"]: r for r in patch_df.to_dict("records")}
    circuit_nodes = []
    for n in nodes:
        name = node_name(n)
        s, pr, ro = score_by[name], patch_by[name], roles[n]
        circuit_nodes.append(dict(
            layer=n[1], type=n[0], index=n[2], name=name,
            eap_score=s["score"], eap_rank=int(s["rank"]),
            rec_all=pr["rec_all"], rec_postout=pr["rec_postout"], rec_reverse=pr["rec_reverse"],
            above_noise=bool(pr["above_noise"]),
            attn_role=ro["attn_role"], tool_mass=ro["tool_mass"], own_mass=ro["own_mass"],
            dla=ro["dla"], dla_sign=ro["dla_sign"], depth=ro["depth"],
        ))
    for rec in circuit_nodes:
        missing = [f for f in CIRCUIT_FIELDS if f not in rec]
        if missing:
            raise AssertionError(f"circuit node {rec['name']} missing fields {missing}")
    circuit = dict(
        model=args.model, repo=model.tt_repo, n_layers=model.cfg.n_layers, n_heads=model.cfg.n_heads,
        n_pairs=len(pairs), topn=args.topn, seed=args.seed, ig_steps=args.ig_steps,
        screen=checks["screen"]["method"], noise_p95=patch_info["noise_p95"], nodes=circuit_nodes,
    )
    write_json(out / "circuit.json", circuit)
    write_json(out / "localize_checks.json", checks)
    print(f"wrote {out}/pairs_summary.csv, node_scores.csv, patch_results.csv, circuit.json "
          f"({len(circuit_nodes)} nodes, {sum(r['above_noise'] for r in circuit_nodes)} above noise)")
    if not args.no_push:
        push_outputs(out, args.hf_repo, "localize")


if __name__ == "__main__":
    main()
