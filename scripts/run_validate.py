#!/usr/bin/env python3
"""Stage 2 -- Circuit validation: necessity, self-repair, faithfulness, specificity.

    python scripts/run_validate.py --model torl --pairs pairs/torl_discovery.jsonl \\
        --circuit outputs/torl/circuit.json --out outputs/torl

    # the full pass: re-rank by where the effect actually lives, sweep circuit size,
    # and measure both specificity controls
    python scripts/run_validate.py --model torl --pairs pairs/torl_discovery.jsonl \\
        --circuit outputs/torl/circuit.json --rank-by rec_all \\
        --k-curve 10,25,50,100,200 --specificity --out outputs/torl_recall

Writes to --out:
  validation_pairs.csv   every Δ, one row per pair -- the evidence behind every metric
  validation.json        the metrics computed from it
  faithfulness_k.csv     faithfulness / collapse vs circuit size (with --k-curve)
  damage.csv             per-pair clean-run collateral damage (with --specificity)
  means.pt               cached ablation means

Every number in validation.json is a pure function of validation_pairs.csv, so a metric
can be redefined without a GPU: see scripts/recompute_metrics.py.

CIRCUIT SELECTION

--rank-by chooses which column of circuit.json orders the nodes. The default `eap` keeps
the localization shortlist as-is. `rec_all` re-selects by measured recovery at all
positions, which is where most of each node's effect sits (rec_all is ~2.4x rec_postout on
ToRL) -- no re-localization needed, circuit.json already stores it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.cli import add_common, get_pairs, push_outputs, setup_logging, write_json
from circuits.hooks import all_nodes, set_seed
from circuits.localize.delta import run_delta
from circuits.models.load import load_model
from circuits.validate.ablate import (POSITIONS, ablate, collect_deltas, faithfulness, k_curve,
                                      necessity, node_means, self_repair, specificity_contrast,
                                      specificity_damage, summarise_damage)

RANK_COLUMNS = ("eap", "rec_all", "rec_postout")


def rank_nodes(circuit: dict, rank_by: str, only_above_noise: bool) -> list[dict]:
    """circuit.json nodes, ordered. `eap` keeps the file's order (already the EAP rank)."""
    rows = [n for n in circuit["nodes"] if not only_above_noise or n["above_noise"]]
    if rank_by == "eap":
        return sorted(rows, key=lambda n: n["eap_rank"])
    return sorted(rows, key=lambda n: -abs(float(n[rank_by])))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    p.add_argument("--circuit", required=True, help="circuit.json from run_localize")
    p.add_argument("--zero", action="store_true", help="zero-ablation instead of mean-ablation")
    p.add_argument("--positions", default="postout", choices=POSITIONS,
                   help="postout = after the injected value (default); all = whole sequence (IOI convention)")
    p.add_argument("--rank-by", default="eap", choices=RANK_COLUMNS,
                   help="which circuit.json column orders the nodes before --topk")
    p.add_argument("--topk", type=int, default=None, help="keep the top k nodes (default: all in the file)")
    p.add_argument("--k-curve", default=None,
                   help="comma-separated circuit sizes to sweep, e.g. 10,25,50,100,200")
    p.add_argument("--k-random", type=int, default=20)
    p.add_argument("--limit", type=int, default=None, help="use only the first N pairs (for timing runs)")
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--z", type=float, default=2.0, help="self-repair spike threshold (z-score of DLA change)")
    p.add_argument("--min-den", type=float, default=0.1,
                   help="drop pairs whose faithfulness denominator |Δ_base − Δ_all_ablated| is below this")
    p.add_argument("--only-above-noise", action="store_true", help="validate only nodes with above_noise=true")
    p.add_argument("--specificity", action="store_true",
                   help="run both specificity controls: accept-pair contrast and clean-run damage")
    p.add_argument("--accept-pairs", default=None,
                   help="pair file for the accept control (default: --pairs, filtered to label=accept)")
    p.add_argument("--max-accept", type=int, default=150, help="cap on accept-control pairs")
    p.add_argument("--no-self-repair", action="store_true")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "zero" if args.zero else "mean"
    pos = args.positions

    circuit = json.load(open(args.circuit))
    ranked = rank_nodes(circuit, args.rank_by, args.only_above_noise)
    if not ranked:
        raise SystemExit("empty circuit")
    as_node = lambda n: (n["type"], int(n["layer"]), int(n["index"]))          # noqa: E731
    topk = args.topk or len(ranked)
    circuits = {f"k{topk}": [as_node(n) for n in ranked[:topk]]}
    main_name = f"k{topk}"
    for k in ([int(x) for x in args.k_curve.split(",")] if args.k_curve else []):
        if k <= len(ranked):
            circuits.setdefault(f"k{k}", [as_node(n) for n in ranked[:k]])
        else:
            print(f"skipping k={k}: circuit.json has only {len(ranked)} nodes")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    _, pairs = get_pairs(model, args)
    if args.limit:
        pairs = pairs[:args.limit]
    means = node_means(model, pairs, out / "means.pt", positions=pos)

    # checks: ablating nothing = baseline; ablating everything is the floor, which is NOT
    # ~0 -- the fully ablated stream is the normalised token embedding and has its own
    # digit preference, which is exactly why faithfulness is floor-adjusted per pair.
    p0 = pairs[0]
    d_base = run_delta(model, p0.corrupt_ids, p0)
    d_empty = ablate(model, p0, [], means, mode, pos)
    if abs(d_empty - d_base) > 1e-6:
        raise AssertionError(f"ablate(empty) {d_empty} != baseline {d_base}")
    d_all = ablate(model, p0, all_nodes(model.cfg), means, mode, pos)
    checks = dict(ablate_empty_equals_baseline=True, delta_base=d_base, delta_all_ablated=d_all,
                  floor_over_signal=abs(d_all) / (abs(d_base) + 1e-8))

    df = collect_deltas(model, pairs, circuits, means, mode=mode, positions=pos,
                        k_random=args.k_random, seed=args.seed, main=main_name)
    df.to_csv(out / "validation_pairs.csv", index=False)

    # faithfulness(all nodes) == 1 and faithfulness(empty) == 0 are identities of the
    # formula, checked on the collected table rather than with two more forward passes.
    ident = df.copy()
    ident.attrs.update(df.attrs)
    ident["circ__id"] = ident["base"]
    ident.attrs["sizes"] = {**ident.attrs["sizes"], "_id": len(all_nodes(model.cfg))}
    f_all = faithfulness(ident, "_id", args.min_den)["faithfulness"]
    if abs(f_all - 1.0) > 1e-6:
        raise AssertionError(f"faithfulness(all nodes) = {f_all} != 1")
    checks["faithfulness_all_nodes"] = f_all

    result = dict(
        model=args.model, circuit=str(args.circuit), rank_by=args.rank_by, positions=pos,
        n_nodes=len(circuits[main_name]), n_pairs=len(pairs), mode=mode, seed=args.seed,
        circuit_sizes={k: len(v) for k, v in circuits.items()},
        necessity=necessity(df, main_name),
        faithfulness=faithfulness(df, main_name, args.min_den),
        checks=checks,
    )
    if not args.no_self_repair:
        result["self_repair"] = self_repair(model, pairs, circuits[main_name], means,
                                            max_iters=args.max_iters, z=args.z, mode=mode, positions=pos)
    if args.k_curve:
        curve = k_curve(df, args.min_den)
        curve.to_csv(out / "faithfulness_k.csv", index=False)
        result["k_curve"] = curve.to_dict("records")

    if args.specificity:
        # variant 2 first: it reuses the pairs and means already in memory
        damage = specificity_damage(model, pairs, circuits[main_name], means, mode, pos, seed=args.seed)
        damage.to_csv(out / "damage.csv", index=False)
        spec = dict(clean_run_damage=summarise_damage(damage))

        # variant 1: the same collapse on pairs the model originally agreed with the tool on
        acc_args = argparse.Namespace(**{**vars(args), "label": "accept",
                                         "pairs": args.accept_pairs or args.pairs})
        _, acc_pairs = get_pairs(model, acc_args)
        acc_pairs = acc_pairs[:args.max_accept]
        acc_means = node_means(model, acc_pairs, out / "means_accept.pt", positions=pos)
        acc_df = collect_deltas(model, acc_pairs, {main_name: circuits[main_name]}, acc_means,
                                mode=mode, positions=pos, k_random=0, seed=args.seed, main=main_name)
        acc_df.to_csv(out / "validation_pairs_accept.csv", index=False)
        spec["accept_contrast"] = specificity_contrast(result["necessity"], necessity(acc_df, main_name))
        result["specificity"] = spec

    write_json(out / "validation.json", result)

    n, f = result["necessity"], result["faithfulness"]
    print(f"\nwrote {out}/validation.json and validation_pairs.csv")
    print(f"  circuit          {result['n_nodes']} nodes, ranked by {args.rank_by}, ablated at {pos}")
    print(f"  necessity        collapse {n['collapse']:.3f} (median {n['collapse_median']:.3f}) "
          f"vs random p95 {n['random_collapse_p95']:.3f}  -> above_random={n['above_random']}")
    print(f"  faithfulness     {f['faithfulness']:.3f} (median of per-pair, IQR {f['iqr']})  "
          f"pooled_signed {f['pooled_signed']:.3f}")
    print(f"  denominator sign +{f['n_den_positive']} / -{f['n_den_negative']}, {f['n_excluded']} excluded")
    if "specificity" in result:
        d, c = result["specificity"]["clean_run_damage"], result["specificity"]["accept_contrast"]
        print(f"  specificity      clean-run top-1 agreement {d['top1_agreement']:.3f}, KL {d['kl_mean']:.4f}")
        print(f"                   accept collapse {c['collapse_accept']:.3f} vs override "
              f"{c['collapse_override']:.3f} (ratio {c['ratio']:.2f})")
    if not args.no_push:
        push_outputs(out, args.hf_repo, "validate")


if __name__ == "__main__":
    main()
