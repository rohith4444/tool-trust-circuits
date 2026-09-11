#!/usr/bin/env python3
"""Stage 3b -- Intervention: head-scaling from the localized circuit.

    python scripts/run_scale.py --model torl \\
        --circuit outputs/circuits/torl/circuit.json \\
        --pairs pairs/torl_discovery.jsonl \\
        --eval-pairs pairs/torl_validation.jsonl \\
        --out outputs/intervene/torl_scale -v

Turns the circuit's tool-favouring components up and its own-answer ones down at the
post-output positions. Chooses (alpha, beta) on DISCOVERY and reports on VALIDATION,
the held-out 17 traces where the tool really was right and the model really overrode it.

Writes to --out:
  role_transfer.csv    do circuit.json's DLA signs still hold on the clean run?
  scale_sweep.csv      mean Δ per (alpha, beta), per policy
  eval_validation.csv  the chosen policy applied to held-out pairs
  scale_results.json   the headline numbers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from circuits.cli import add_common, get_pairs, push_outputs, setup_logging, write_json
from circuits.hooks import set_seed
from circuits.intervene import scale
from circuits.intervene.common import dedupe, run_ids, subsample
from circuits.localize.delta import run_delta
from circuits.models.load import load_model


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    p.add_argument("--circuit", required=True, help="circuit.json from run_localize.py")
    p.add_argument("--eval-pairs", required=True, help="held-out jsonl (the validation split)")
    p.add_argument("--alphas", default="1,1.5,2,3,5", help="multipliers for tool-favouring nodes")
    p.add_argument("--betas", default="1,0.75,0.5,0.25,0", help="multipliers for own-answer nodes")
    p.add_argument("--keys", default="dla_sign,attn_role", help="policies to sweep, comma-separated")
    p.add_argument("--all-nodes", action="store_true",
                   help="use the whole EAP-IG shortlist, not just patching-confirmed nodes")
    p.add_argument("--sweep-n", type=int, default=0,
                   help="subsample discovery when CHOOSING (alpha, beta); 0 = all "
                        "(default). The "
                        "held-out report always uses every pair")
    p.add_argument("--run", default="clean", choices=("clean", "corrupt"),
                   help="which run to intervene on; clean is the real failure condition")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    circuit = scale.load_circuit(args.circuit)
    nodes = scale.select(circuit, confirmed_only=not args.all_nodes)
    checks: dict = {"circuit": {k: circuit.get(k) for k in ("model", "n_pairs", "topn", "noise_p95")},
                    "n_nodes_in_file": len(circuit["nodes"]), "n_nodes_used": len(nodes)}
    print(f"circuit: {len(nodes)} of {len(circuit['nodes'])} nodes "
          f"({'all' if args.all_nodes else 'patching-confirmed'})")
    if circuit.get("model") != args.model:
        print(f"WARNING: circuit.json says model={circuit.get('model')!r} but --model is {args.model!r}")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    _, fit_pairs = get_pairs(model, args)
    eval_args = argparse.Namespace(**{**vars(args), "pairs": args.eval_pairs})
    _, eval_pairs = get_pairs(model, eval_args)
    n_fit, n_eval = len(dedupe(fit_pairs, args.run)), len(dedupe(eval_pairs, args.run))
    checks["n_fit_runs"], checks["n_eval_runs"] = n_fit, n_eval
    print(f"fit on {n_fit} distinct runs, evaluate on {n_eval}")

    scale.identity_check(model, fit_pairs[0], nodes)
    checks["identity"] = "ok"

    # ---- do the file's roles still describe the run we intervene on? ----
    transfer = scale.role_transfer(model, fit_pairs, nodes, run=args.run)
    pd.DataFrame(transfer["rows"]).to_csv(out / "role_transfer.csv", index=False)
    checks["role_transfer"] = {k: transfer[k] for k in ("n_nodes", "n_agree", "frac_agree", "run")}
    print(f"DLA sign transfer corrupt->{args.run}: "
          f"{transfer['n_agree']}/{transfer['n_nodes']} agree ({transfer['frac_agree']:.0%})")
    if transfer["frac_agree"] < 0.7:
        print("WARNING: the circuit's roles were measured on corrupt runs and mostly do NOT "
              "hold here. A policy keyed on dla_sign may be pushing the wrong way.")

    # ---- sweep ----
    alphas = [float(a) for a in args.alphas.split(",")]
    betas = [float(b) for b in args.betas.split(",")]
    sweep_pairs = subsample(dedupe(fit_pairs, args.run), args.sweep_n, args.seed)
    n_cfg = len(alphas) * len(betas) * len(args.keys.split(","))
    checks["sweep_n"] = len(sweep_pairs)
    print(f"sweeping {n_cfg} configs over {len(sweep_pairs)} runs "
          f"= {n_cfg * len(sweep_pairs)} forwards")
    rows = []
    for key in (k.strip() for k in args.keys.split(",")):
        counts = scale.policy_counts(scale.policy(nodes, 2.0, 0.5, key), 2.0, 0.5)
        checks.setdefault("policies", {})[key] = counts
        print(f"policy {key}: {counts['n_up']} up, {counts['n_down']} down, "
              f"{counts['n_neutral']} untouched")
        if counts["n_up"] == 0 and counts["n_down"] == 0:
            print(f"  skipping {key}: no node carries a role under it")
            continue
        rows += scale.sweep(model, sweep_pairs, nodes, alphas, betas, key, args.run)
    if not rows:
        raise SystemExit("no policy assigned a role to any node; check circuit.json fields")
    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(out / "scale_sweep.csv", index=False)

    best = max(rows, key=lambda r: r["delta_shift"])
    print(f"chosen on discovery: {best['key']} alpha={best['alpha']} beta={best['beta']} "
          f"(Δ shift {best['delta_shift']:+.3f})")
    # max() returns something even when nothing helped, and a confident "chosen" line over
    # a losing sweep reads exactly like a working intervention.
    if best["delta_shift"] <= 0:
        print("WARNING: no (alpha, beta) improved Δ on discovery. The held-out table below "
              "reports the least-bad option, not a working intervention.")
    checks["improves_on_discovery"] = bool(best["delta_shift"] > 0)

    # ---- report on held-out ----
    scales = scale.policy(nodes, best["alpha"], best["beta"], best["key"])
    ev_rows = []
    for pair in dedupe(eval_pairs, args.run):
        before = run_delta(model, run_ids(pair, args.run), pair)
        after = scale.scaled_delta(model, pair, scales, args.run)
        ev_rows.append(dict(id=pair.id, tool_case=pair.meta.get("tool_case"),
                            delta_before=before, delta_after=after, shift=after - before,
                            flipped_sign=bool(before < 0 <= after)))
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(out / "eval_validation.csv", index=False)

    results = dict(
        model=args.model, run=args.run, policy=best["key"],
        alpha=float(best["alpha"]), beta=float(best["beta"]),
        n_nodes=len(nodes), n_fit_runs=n_fit, n_eval_runs=n_eval,
        discovery_delta_shift=float(best["delta_shift"]),
        eval_delta_before=float(ev["delta_before"].mean()),
        eval_delta_after=float(ev["delta_after"].mean()),
        eval_delta_shift=float(ev["shift"].mean()),
        eval_frac_negative_before=float((ev["delta_before"] < 0).mean()),
        eval_frac_positive_after=float((ev["delta_after"] > 0).mean()),
        eval_frac_sign_flipped=float(ev["flipped_sign"].mean()),
        checks=checks,
    )
    write_json(out / "scale_results.json", results)
    print(f"held-out Δ {results['eval_delta_before']:+.3f} -> {results['eval_delta_after']:+.3f} "
          f"({results['eval_frac_sign_flipped']:.0%} of pairs flipped sign)")
    if not args.no_push:
        push_outputs(out, args.hf_repo, "scale")


if __name__ == "__main__":
    main()
