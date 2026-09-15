#!/usr/bin/env python3
"""Stage 4 -- Evaluation: Tool-Ignored rate, accuracy and specificity, swept over alpha.

    # steering
    python scripts/run_tune_alpha.py --model torl --method steer \\
        --direction outputs/intervene/torl_steer_bigalpha/directions.pt \\
        --layer 18 --alphas 36,72,144 \\
        --pairs pairs/torl_validation.jsonl --out outputs/eval/torl_steer -v

    # head-scaling
    python scripts/run_tune_alpha.py --model torl --method scale \\
        --circuit results/torl/circuit.json --policy attn_role --alphas 2,3,5 \\
        --pairs pairs/torl_validation.jsonl --out outputs/eval/torl_scale -v

Both methods go through one evaluator so the numbers are comparable: same pairs, same
decoding, same parsing, same metrics.

WHY SWEEP HERE RATHER THAN ON Δ

run_steer.py and run_scale.py choose their config by Δ, and both picked the top of their
range -- steering at 144 (79% of the residual norm), scaling at 5x. Δ moving says the
logit ordering changed; it does not say the model still writes a sensible answer, and at
those perturbation sizes that is a real question. Sweeping alpha on the metric that
matters answers it directly.

The goal is the SMALLEST alpha that works, not the largest. "Displacing 40% of the
residual stream fixes 6 of 7" is a stronger claim than the same result at 79%, and a huge
perturbation invites "you broke the model until it agreed with you".

Writes to --out:
  answers_a{alpha}.csv  per pair: what the model wrote before and after
  eval_sweep.csv        one row per alpha: accuracy, Tool-Ignored, specificity
  eval_results.json     the chosen alpha and the full table
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from circuits.cli import add_common, get_pairs, push_outputs, setup_logging, write_json
from circuits.hooks import scale_hooks, set_seed, steer_hooks
from circuits.intervene import scale, tune_alpha as ta
from circuits.intervene.common import dedupe


def hooks_for(args, model, state: dict, alpha: float):
    """(pair, positions) -> hooks, for one alpha. Method-agnostic by design.

    Positions are passed in rather than read off the pair because generation extends them;
    see ta.positions_from.
    """
    if args.method == "steer":
        vec = state["vec"]
        return lambda pair, positions: steer_hooks(args.layer, vec, positions, alpha)
    scales = scale.policy(state["nodes"], alpha, args.beta, args.policy)
    return lambda pair, positions: scale_hooks(scales, positions)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    p.add_argument("--method", required=True, choices=("steer", "scale"))
    p.add_argument("--alphas", required=True,
                   help="comma-separated. Steering: a length, compare against the residual "
                        "norm. Scaling: a plain multiplier")
    p.add_argument("--direction", help="directions.pt from run_steer.py (method=steer)")
    p.add_argument("--layer", type=int, help="layer to steer at (method=steer)")
    p.add_argument("--circuit", help="circuit.json (method=scale)")
    p.add_argument("--policy", default="attn_role", choices=("dla_sign", "attn_role"),
                   help="attn_role by default: dla_sign's labels were measured on corrupt "
                        "runs and transfer to clean ones at chance (18/37)")
    p.add_argument("--beta", type=float, default=1.0,
                   help="own-answer multiplier (method=scale). Inert under attn_role, "
                        "which finds no own_readers among confirmed nodes")
    p.add_argument("--all-nodes", action="store_true")
    p.add_argument("--max-new", type=int, default=ta.MAX_NEW,
                   help="tokens to generate; the probe leaves the model inside \\boxed{ "
                        "so the answer ends within a few")
    p.add_argument("--tol", type=float, default=0.02,
                   help="accuracy tolerance when picking the smallest sufficient alpha")
    p.add_argument("--skip-agreement", action="store_true",
                   help="skip the agreement control (tool reports X, nothing to arbitrate)")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.method == "steer" and not (args.direction and args.layer is not None):
        raise SystemExit("--method steer needs --direction and --layer")
    if args.method == "scale" and not args.circuit:
        raise SystemExit("--method scale needs --circuit")
    alphas = [float(a) for a in args.alphas.split(",")]

    from circuits.models.load import load_model
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    _, pairs = get_pairs(model, args)
    runs = dedupe(pairs, "clean")

    state: dict = {}
    if args.method == "steer":
        state["vec"] = torch.load(args.direction)["delta_split"][args.layer].to(model.cfg.device)
        print(f"steering at layer {args.layer}, alphas {alphas}")
    else:
        circuit = scale.load_circuit(args.circuit)
        state["nodes"] = scale.select(circuit, confirmed_only=not args.all_nodes)
        counts = scale.policy_counts(scale.policy(state["nodes"], 2.0, 0.5, args.policy), 2.0, 0.5)
        print(f"scaling {len(state['nodes'])} confirmed nodes, policy {args.policy}: "
              f"{counts['n_up']} up, {counts['n_down']} down, {counts['n_neutral']} untouched")
        if counts["n_up"] == 0 and counts["n_down"] == 0:
            raise SystemExit(f"policy {args.policy} assigns no role to any confirmed node")
    print(f"evaluating {len(runs)} distinct held-out runs\n")

    # Generated once: independent of alpha, so every row below is compared against the
    # same baseline rather than a separately generated one.
    before = ta.baseline_answers(model, pairs, max_new=args.max_new)
    base_correct = sum(ta.matches(before[p.id], ta.targets(p)[0])
                       for p in runs) / max(len(runs), 1)
    print(f"baseline accuracy {base_correct:.0%} on {len(runs)} runs "
          f"(should be ~0: these are labelled failures)")
    if base_correct > 0.05:
        print("WARNING: these pairs are not the failures they are labelled as, so the "
              "relative reduction below is measured against the wrong floor.")

    rows = []
    for alpha in alphas:
        hooks_fn = hooks_for(args, model, state, alpha)
        flip = ta.answer_flip(model, pairs, hooks_fn, max_new=args.max_new,
                                    before_by_id=before)
        pd.DataFrame(flip).to_csv(out / f"answers_a{alpha:g}.csv", index=False)
        s = ta.summarise(flip)

        spec = None
        if not args.skip_agreement:
            conflict = sum(ta.delta_shift_for(model, pr, hooks_fn)
                           for pr in runs) / max(len(runs), 1)
            spec = ta.specificity(
                ta.agreement_delta(model, pairs, hooks_fn), conflict)

        rows.append(dict(alpha=alpha, **s,
                         agreement_shift=spec["agreement_delta_shift"] if spec else None,
                         conflict_shift=spec["conflict_delta_shift"] if spec else None,
                         agreement_ratio=spec["ratio"] if spec else None,
                         specific=spec["specific"] if spec else None))
        print(f"alpha {alpha:>7g}: accuracy {s['accuracy_after']:>5.0%}   "
              f"Tool-Ignored {s['tool_ignored_before']:.0%} -> {s['tool_ignored_after']:.0%}   "
              f"(-{s['tool_ignored_rel_drop']:.0%} rel)"
              + (f"   specificity {spec['ratio']:.2f} "
                 f"{'ok' if spec['specific'] else 'FAIL'}" if spec else ""))

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / "eval_sweep.csv", index=False)

    # Smallest alpha within tol of the best accuracy, not the best accuracy outright: a
    # tie between 72 and 144 should report 72, because the smaller perturbation is the
    # stronger claim about mechanism.
    best_acc = sweep["accuracy_after"].max()
    ok = sweep[sweep["accuracy_after"] >= best_acc - args.tol]
    chosen = ok.loc[ok["alpha"].idxmin()]
    print(f"\nchosen: alpha {chosen['alpha']:g} -- smallest within {args.tol:.0%} of the "
          f"best accuracy ({best_acc:.0%})")
    if chosen["specific"] is False:
        print("WARNING: at this alpha the intervention moves agreement cases nearly as "
              "much as conflicts. It is pushing toward Y unconditionally, not arbitrating "
              "trust.")

    results = dict(
        model=args.model, method=args.method, layer=args.layer,
        policy=args.policy if args.method == "scale" else None,
        beta=args.beta if args.method == "scale" else None,
        n_runs=len(runs), alphas=alphas, chosen_alpha=float(chosen["alpha"]),
        token_cost=0, training=False,
        baseline_accuracy=base_correct,
        **{k: (None if pd.isna(chosen[k]) else chosen[k])
           for k in sweep.columns if k != "alpha"},
        sweep=rows,
    )
    write_json(out / "eval_results.json", results)
    if not args.no_push:
        push_outputs(out, args.hf_repo, "evaluate")


if __name__ == "__main__":
    main()
