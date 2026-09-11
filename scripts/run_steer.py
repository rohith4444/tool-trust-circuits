#!/usr/bin/env python3
"""Stage 3a -- Intervention: difference-of-means steering vector.

    python scripts/run_steer.py --model torl \\
        --pairs pairs_updated/torl_discovery.jsonl \\
        --eval-pairs pairs_updated/torl_validation.jsonl \\
        --out outputs/intervene/torl_steer -v

Fits a "trust the tool" direction on the DISCOVERY split and reports it on VALIDATION,
which is held out and is the only split whose clean run reproduces a real Tool-Ignored
case. Needs no circuit, so it does not wait on localization.

Writes to --out:
  hook_checks.json     tensor-level hook tests + the identity check on a real pair
  directions.pt        [n_layers, d_model] for both splits, plus norms and cosines
  layer_sweep.csv      mean Δ per layer at one alpha, on discovery
  alpha_sweep.csv      mean Δ per alpha at the winning layer, on discovery
  eval_validation.csv  the chosen (layer, alpha) applied to held-out pairs
  steer_results.json   the headline numbers
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from circuits.cli import add_common, get_pairs, push_outputs, setup_logging, write_json
from circuits.hooks import scale_hooks, set_seed, steer_hooks
from circuits.intervene import steer
from circuits.intervene.common import subsample
from circuits.localize.delta import run_delta
from circuits.models.load import load_model


def hook_unit_tests() -> dict:
    """Pure-tensor checks, before any model loads. Seconds, and they fail loudly.

    A scaling or steering hook that touches the wrong positions produces numbers that
    look like results, so these run first rather than never.
    """
    out = {}

    act = torch.ones(1, 6, 4, 2)                                   # [batch, T, H, d_head]
    fn = scale_hooks({("a", 0, 1): 2.0, ("a", 0, 3): 0.5}, positions=[4, 5])[0][1]
    got = fn(act.clone(), None)
    out["heads_scaled"] = got[0, 4, :, 0].tolist()                 # [1, 2, 1, 0.5]
    out["heads_untouched"] = got[0, 3, :, 0].tolist()              # [1, 1, 1, 1]
    assert out["heads_scaled"] == [1.0, 2.0, 1.0, 0.5], out
    assert out["heads_untouched"] == [1.0] * 4, out

    act = torch.ones(1, 6, 3)                                      # [batch, T, d_model]
    fn = scale_hooks({("m", 5, 0): 3.0}, positions=[5])[0][1]
    got = fn(act.clone(), None)
    assert got[0, 4].tolist() == [1.0] * 3 and got[0, 5].tolist() == [3.0] * 3, got

    act = torch.zeros(1, 4, 3)
    fn = steer_hooks(2, torch.tensor([1.0, 2.0, 3.0]), positions=[3], alpha=0.5)[0][1]
    got = fn(act.clone(), None)
    assert got[0, 2].tolist() == [0.0] * 3 and got[0, 3].tolist() == [0.5, 1.0, 1.5], got

    a = torch.randn(1, 3, 2, 2)                                    # alpha 1.0 is a no-op
    fn = scale_hooks({("a", 0, 1): 1.0}, positions=[0, 1])[0][1]
    assert torch.allclose(fn(a.clone(), None), a)

    out["ok"] = True
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p)
    p.add_argument("--eval-pairs", required=True, help="held-out jsonl (the validation split)")
    p.add_argument("--probe-alpha", type=float, default=2.0, help="alpha used while sweeping layers")
    p.add_argument("--alphas", default="0,0.5,1,2,4,8", help="comma-separated, swept at the winning layer")
    p.add_argument("--run", default="clean", choices=("clean", "corrupt"),
                   help="which run to fit and steer on; clean is the real failure condition")
    p.add_argument("--split", default="sign", choices=("sign", "median"),
                   help="how to cut accept from override: Δ>0 (principled) or median Δ "
                        "(balanced; use if the sign split leaves one side tiny)")
    p.add_argument("--sweep-n", type=int, default=0,
                   help="subsample discovery when CHOOSING (layer, alpha); 0 = all "
                        "(default). The "
                        "direction is always fitted on every run, and the held-out "
                        "report always uses every pair -- only the search is subsampled")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    checks = {"hook_unit_tests": hook_unit_tests()}
    print("hook unit tests ok")

    model = load_model(args.model, device=args.device, dtype=args.dtype)
    _, fit_pairs = get_pairs(model, args)
    eval_args = argparse.Namespace(**{**vars(args), "pairs": args.eval_pairs})
    _, eval_pairs = get_pairs(model, eval_args)
    checks["n_fit"], checks["n_eval"] = len(fit_pairs), len(eval_pairs)
    print(f"fit on {len(fit_pairs)} pairs, evaluate on {len(eval_pairs)}")

    # the hook must be an identity before it is trusted to be an intervention
    steer.identity_check(model, fit_pairs[0], model.cfg.n_layers // 2, model.cfg.d_model)
    checks["identity"] = "ok"

    # ---- fit ----
    collected = steer.collect(model, fit_pairs, run=args.run)
    by_delta = steer.direction_from_delta(collected, split=args.split)
    checks["fit_split"] = dict(n_accept=by_delta["n_accept"], n_override=by_delta["n_override"],
                               how=by_delta["split"])
    print(f"fit split ({by_delta['split']}): {by_delta['n_accept']} accept / "
          f"{by_delta['n_override']} override")
    # In a clean run the tool reports the truth and these models mostly copy it, so the
    # sign split can leave one side too small to average meaningfully.
    if min(by_delta["n_accept"], by_delta["n_override"]) < 20:
        print(f"WARNING: smaller population is {min(by_delta['n_accept'], by_delta['n_override'])} runs. "
              f"Consider --split median.")
    d = collected["deltas"]
    checks["delta_distribution"] = dict(mean=float(d.mean()), median=float(d.median()),
                                        frac_positive=float((d > 0).float().mean()))
    print(f"fit Δ: mean {d.mean():+.3f}  median {d.median():+.3f}  "
          f"{(d > 0).float().mean():.0%} positive")

    saved = dict(delta_split=by_delta["direction"], raw_norm=by_delta["raw_norm"],
                 resid_norm=by_delta["resid_norm"])
    try:
        by_label = steer.direction_from_labels(collected)
        cos = steer.cosine(by_delta["direction"], by_label["direction"])
        saved["label_split"] = by_label["direction"]
        saved["cosine_delta_vs_label"] = cos
        checks["cosine_delta_vs_label"] = dict(mean=float(cos.mean()), min=float(cos.min()),
                                               max=float(cos.max()))
        print(f"cosine(delta-split, tool_case-split): mean {cos.mean():.3f}")
    except ValueError as e:
        checks["label_split"] = f"skipped: {e}"       # one population empty in this file

    torch.save(saved, out / "directions.pt")
    direction = by_delta["direction"]

    # ---- sweep layers, then alpha at the winner ----
    # The direction above was fitted on every run; only the search for (layer, alpha) is
    # subsampled. 28 layers over 356 runs is ~10k forwards and the ranking of layers
    # settles long before the mean does.
    sweep_pairs = subsample(steer.dedupe(fit_pairs, args.run), args.sweep_n, args.seed)
    checks["sweep_n"] = len(sweep_pairs)
    print(f"sweeping {model.cfg.n_layers} layers over {len(sweep_pairs)} runs "
          f"= {model.cfg.n_layers * len(sweep_pairs)} forwards")
    layer_rows = steer.sweep(model, sweep_pairs, direction, range(model.cfg.n_layers),
                             [args.probe_alpha], run=args.run)
    pd.DataFrame(layer_rows).to_csv(out / "layer_sweep.csv", index=False)
    best_layer = max(layer_rows, key=lambda r: r["delta_shift"])["layer"]

    # The direction is a unit vector, so alpha is an absolute length added to a residual
    # whose own norm is whatever it is. Without this, an alpha range that is 1% of the
    # stream looks like "steering does nothing" rather than "we did not push hard enough".
    typical = float(by_delta["resid_norm"][best_layer])
    checks["resid_norm_at_best_layer"] = typical
    alphas = [float(a) for a in args.alphas.split(",")]
    print(f"layer {best_layer}: typical residual norm {typical:.1f}; "
          f"alphas {alphas} are {100*max(alphas)/typical:.1f}% of it at the top end")
    if max(alphas) < 0.02 * typical:
        print(f"WARNING: the whole alpha range is under 2% of the residual norm. "
              f"Try --alphas 0,{typical*0.05:.0f},{typical*0.1:.0f},{typical*0.2:.0f}")
    alpha_rows = steer.sweep(model, sweep_pairs, direction, [best_layer], alphas, run=args.run)
    pd.DataFrame(alpha_rows).to_csv(out / "alpha_sweep.csv", index=False)
    best = max(alpha_rows, key=lambda r: r["delta_shift"])
    print(f"chosen on discovery: layer {best_layer}, alpha {best['alpha']} "
          f"(Δ shift {best['delta_shift']:+.3f})")
    # max() always returns something, so without this a direction that helps nowhere
    # still produces a confident-looking "chosen" line and a held-out table.
    if best["delta_shift"] <= 0:
        print("WARNING: no (layer, alpha) improved Δ on discovery. The held-out numbers "
              "below are reporting the least-bad option, not a working intervention.")
    checks["improves_on_discovery"] = bool(best["delta_shift"] > 0)

    # ---- report on held-out ----
    vec = direction[best_layer]
    rows = []
    # One row per distinct run, not per pair: on clean runs the two Z-variants of a
    # trace are byte-identical, so keeping both would double-count every trace in the
    # held-out average.
    for pair in steer.dedupe(eval_pairs, args.run):
        before = run_delta(model, steer.run_ids(pair, args.run), pair)
        after = steer.steered_delta(model, pair, best_layer, vec, best["alpha"], args.run)
        rows.append(dict(id=pair.id, tool_case=pair.meta.get("tool_case"),
                         delta_before=before, delta_after=after, shift=after - before,
                         flipped_sign=bool(before < 0 <= after)))
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "eval_validation.csv", index=False)

    results = dict(
        model=args.model, run=args.run, layer=int(best_layer), alpha=float(best["alpha"]),
        n_fit=len(fit_pairs), n_eval=len(eval_pairs),
        discovery_delta_shift=float(best["delta_shift"]),
        eval_delta_before=float(ev["delta_before"].mean()),
        eval_delta_after=float(ev["delta_after"].mean()),
        eval_delta_shift=float(ev["shift"].mean()),
        # the headline: a validation pair starts with the model preferring its own wrong
        # answer, so a sign flip is the model changing its mind about the tool
        eval_frac_negative_before=float((ev["delta_before"] < 0).mean()),
        eval_frac_positive_after=float((ev["delta_after"] > 0).mean()),
        eval_frac_sign_flipped=float(ev["flipped_sign"].mean()),
        checks=checks,
    )
    write_json(out / "steer_results.json", results)
    write_json(out / "hook_checks.json", checks)
    print(f"held-out Δ {results['eval_delta_before']:+.3f} -> {results['eval_delta_after']:+.3f} "
          f"({results['eval_frac_sign_flipped']:.0%} of pairs flipped sign)")
    if not args.no_push:
        push_outputs(out, args.hf_repo, "steer")


if __name__ == "__main__":
    main()
