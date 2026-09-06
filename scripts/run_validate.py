#!/usr/bin/env python3
"""Stage 2 -- Circuit validation: necessity, self-repair, faithfulness (specificity: TODO).

    python scripts/run_validate.py --model torl --pairs pairs.jsonl --circuit outputs/torl/circuit.json --out outputs/torl

Writes validation.json (and means.pt, the cached ablation means) to --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from circuits.cli import add_common, get_pairs, setup_logging, write_json
from circuits.hooks import all_nodes, set_seed
from circuits.localize.delta import run_delta
from circuits.models.load import load_model
from circuits.validate.ablate import ablate, faithfulness, necessity, node_means, self_repair


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    p.add_argument("--circuit", required=True, help="circuit.json from run_localize")
    p.add_argument("--zero", action="store_true", help="zero-ablation instead of mean-ablation")
    p.add_argument("--k-random", type=int, default=20)
    p.add_argument("--max-iters", type=int, default=3)
    p.add_argument("--z", type=float, default=2.0, help="self-repair spike threshold (z-score of DLA change)")
    p.add_argument("--only-above-noise", action="store_true", help="validate only nodes with above_noise=true")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mode = "zero" if args.zero else "mean"

    circuit = json.load(open(args.circuit))
    nodes = [(n["type"], int(n["layer"]), int(n["index"])) for n in circuit["nodes"]
             if not args.only_above_noise or n["above_noise"]]
    if not nodes:
        raise SystemExit("empty circuit")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    _, pairs = get_pairs(model, args)
    means = node_means(model, pairs, out / "means.pt")

    # checks: ablating nothing = baseline; ablating everything collapses Δ; faithfulness(all) ≈ 1
    p0 = pairs[0]
    d_base = run_delta(model, p0.corrupt_ids, p0)
    d_empty = ablate(model, p0, [], means, mode)
    if abs(d_empty - d_base) > 1e-6:
        raise AssertionError(f"ablate(empty) {d_empty} != baseline {d_base}")
    d_all = ablate(model, p0, all_nodes(model.cfg), means, mode)
    f_all = faithfulness(model, pairs[:1], all_nodes(model.cfg), means, mode)["faithfulness"]
    if abs(f_all - 1.0) > 1e-3:
        raise AssertionError(f"faithfulness(all nodes) = {f_all} != 1")
    checks = dict(ablate_empty_equals_baseline=True, delta_base=d_base, delta_all_ablated=d_all,
                  all_ablated_ratio=abs(d_all) / (abs(d_base) + 1e-8), faithfulness_all_nodes=f_all)

    result = dict(
        model=args.model, circuit=str(args.circuit), n_nodes=len(nodes), n_pairs=len(pairs), mode=mode, seed=args.seed,
        necessity=necessity(model, pairs, nodes, means, k_random=args.k_random, seed=args.seed, mode=mode),
        self_repair=self_repair(model, pairs, nodes, means, max_iters=args.max_iters, z=args.z, mode=mode),
        faithfulness=faithfulness(model, pairs, nodes, means, mode),
        specificity="NotImplemented: needs agreement traces + benchmark accuracy (held-out split, math_verify)",
        checks=checks,
    )
    write_json(out / "validation.json", result)
    f = result["faithfulness"]["faithfulness"]
    print(f"wrote {out}/validation.json: necessity_gap={result['necessity']['necessity_gap']:.3f} "
          f"(random p95 {result['necessity']['random_p95']:.3f}), faithfulness={f:.3f}, "
          f"self_repair converged={result['self_repair']['converged']} in {result['self_repair']['n_iters']} iters")
    if f < 0.5:
        print("WARNING: faithfulness < 50% -- escalate (see onboarding plan)")


if __name__ == "__main__":
    main()
