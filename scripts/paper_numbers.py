#!/usr/bin/env python3
"""Every number section 4 of the paper needs, read off the artifacts on disk.

    python scripts/paper_numbers.py --results /workspace/results \\
        --runs /workspace/repo/outputs/circuits --intervene /workspace/repo/outputs/intervene \\
        --out /workspace/outputs/paper_numbers.txt

Reads only; runs nothing on a GPU. Missing files are reported and skipped, so it can be
run before the last jobs finish and again afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ("torl", "simpletir", "demyagent")


def _p(s=""):
    print(s)


def delta_validity(results: Path) -> None:
    """Does Delta's sign track the behaviour it is supposed to describe?

    Three readings, because they are not the same question and the paper should use the
    one it can defend:

      top1_*_is       what the model actually predicts at the readout ("tool"/"own"/other).
                      The most direct statement: on the clean run the tool's value is the
                      ground truth, so "tool" means the model would write the correct answer.
      sign vs label   sign(Delta) against the ORIGINAL trace's accept/override label. Weaker
                      than it looks for tool_echoed/tool_other, where clean_text replaces the
                      tool's value with Y and the label no longer describes what the model
                      faces; only tool_right survives the substitution unchanged.
      clean vs corrupt  whether injecting Z instead of Y moves Delta at all. This is the
                      manipulation check for the whole dataset.
    """
    _p("=" * 72)
    _p("1. DELTA AS A BEHAVIOURAL DESCRIPTOR")
    _p("=" * 72)
    for m in MODELS:
        f = results / m / "pairs_summary.csv"
        if not f.exists():
            _p(f"  {m}: MISSING {f}")
            continue
        df = pd.read_csv(f)
        _p(f"\n  {m}  n={len(df)}")
        for run in ("clean", "corrupt"):
            col = f"top1_{run}_is"
            if col in df:
                vc = df[col].value_counts(normalize=True)
                _p(f"    top-1 at readout, {run:<8} " +
                   "  ".join(f"{k} {v:.1%}" for k, v in vc.items()))
        if "delta_clean" in df and "delta_corrupt" in df:
            dc, dx = df["delta_clean"].to_numpy(), df["delta_corrupt"].to_numpy()
            _p(f"    Delta clean   mean {dc.mean():+.3f}  median {np.median(dc):+.3f}  "
               f"{(dc > 0).mean():.1%} positive")
            _p(f"    Delta corrupt mean {dx.mean():+.3f}  median {np.median(dx):+.3f}  "
               f"{(dx > 0).mean():.1%} positive")
            _p(f"    injecting Z lowers Delta on {(dx < dc).mean():.1%} of pairs "
               f"(manipulation check)")
        if "label" in df and df["label"].notna().any():
            sub = df[df["label"].notna()]
            pred = np.where(sub["delta_clean"] > 0, "accept", "override")
            _p(f"    sign(Delta_clean) == label: {(pred == sub['label']).mean():.1%} "
               f"of {len(sub)} labelled pairs   (label counts {dict(sub['label'].value_counts())})")


def circuit_facts(results: Path) -> None:
    _p("\n" + "=" * 72)
    _p("2. THE CIRCUIT")
    _p("=" * 72)
    for m in MODELS:
        cj, ns = results / m / "circuit.json", results / m / "node_scores.csv"
        if not cj.exists():
            _p(f"  {m}: MISSING {cj}")
            continue
        c = json.load(open(cj))
        n_comp = c["n_layers"] * c["n_heads"] + c["n_layers"]
        conf = [n for n in c["nodes"] if n["above_noise"]]
        heads = [n for n in conf if n["type"] == "a" and n["layer"] != 0]
        layers = sorted({n["layer"] for n in heads})
        _p(f"\n  {m}  {c['n_layers']} layers x {c['n_heads']} heads = {n_comp} components")
        _p(f"    shortlist {len(c['nodes'])}  ->  patching-confirmed {len(conf)}  "
           f"->  confirmed non-layer-0 heads {len(heads)} ({len(heads)/n_comp:.1%})")
        _p(f"    layers {min(layers)}-{max(layers)} of {c['n_layers']} "
           f"(top {(c['n_layers']-min(layers))/c['n_layers']:.0%} of depth)")
        by = lambda k: {r: sum(1 for n in heads if n.get(k) == r)                    # noqa: E731
                        for r in sorted({n.get(k) for n in heads}, key=str)}
        _p(f"    attn_role  {by('attn_role')}")
        _p(f"    dla_sign   {by('dla_sign')}")
        top = sorted(heads, key=lambda n: -abs(float(n["rec_postout"])))[:8]
        _p(f"    top by rec_postout: " +
           ", ".join(f"{n['name']} {float(n['rec_postout']):.3f}" for n in top))
        if ns.exists():
            s = pd.read_csv(ns).sort_values("rank")
            tot = s["abs_score"].sum()
            for k in (10, 20, 50):
                _p(f"    top-{k:<3} carry {s['abs_score'].head(k).sum()/tot:.1%} of |attribution|")
            first = s.iloc[0]
            pr = [n for n in c["nodes"] if n["name"] == first["name"]]
            rec = float(pr[0]["rec_postout"]) if pr else float("nan")
            _p(f"    attribution rank 1 is {first['name']}, rec_postout {rec:.3f}  "
               f"(confirmed={bool(pr and pr[0]['above_noise'])})")
    # cross-model head agreement
    _p("")
    sets = {}
    for m in MODELS:
        cj = results / m / "circuit.json"
        if cj.exists():
            c = json.load(open(cj))
            h = [n for n in c["nodes"] if n["above_noise"] and n["type"] == "a" and n["layer"] != 0]
            sets[m] = (c, sorted(h, key=lambda n: -abs(float(n["rec_postout"]))))
    for a, b in (("torl", "simpletir"), ("torl", "demyagent"), ("simpletir", "demyagent")):
        if a not in sets or b not in sets:
            continue
        ca, ha = sets[a]
        cb, hb = sets[b]
        same = (ca["n_layers"], ca["n_heads"]) == (cb["n_layers"], cb["n_heads"])
        na, nb = {n["name"] for n in ha}, {n["name"] for n in hb}
        t12a, t12b = {n["name"] for n in ha[:12]}, {n["name"] for n in hb[:12]}
        _p(f"  {a} vs {b}  same_arch={same}  "
           f"full-set overlap {len(na & nb)}/{len(na | nb)} (jaccard {len(na & nb)/len(na | nb):.2f})  "
           f"top-12 shared {len(t12a & t12b)}")
        if same:
            _p(f"      shared top-12: {sorted(t12a & t12b)}")


def validation(runs: Path) -> None:
    _p("\n" + "=" * 72)
    _p("3. VALIDATION  (circuit vs matched control)")
    _p("=" * 72)
    for m in MODELS:
        f = runs / f"final_{m}" / "validation.json"
        if not f.exists():
            _p(f"  {m}: MISSING {f}")
            continue
        d = json.load(open(f))
        n, fa = d["necessity"], d["faithfulness"]
        fr = d.get("faithfulness_random", {})
        s = d["specificity"]["clean_run_damage"]
        _p(f"\n  {m}  {d['n_nodes']} heads  n_pairs={d['n_pairs']}  "
           f"ablatable={d.get('n_ablatable')}  protect={d.get('protect_layers')}/{d.get('protect_types')!r}")
        _p(f"    |Delta| base {n['abs_delta_base']:.3f} -> ablated {n['abs_delta_ablated']:.3f}")
        _p(f"    necessity     {n['collapse']:.3f}  vs matched random p95 {n['random_collapse_p95']:.3f}"
           f"   ({n['collapse']/max(n['random_collapse_p95'],1e-9):.1f}x)")
        _p(f"    faithfulness  {fa['faithfulness']:.3f}  IQR {[round(x,3) for x in fa['iqr']]}"
           + (f"  vs null mean {fr['mean']:.3f} max {fr['max']:.3f} ({fr['n']} draws)" if fr.get("n") else ""))
        for reg in ("preout", "postout"):
            if reg in s:
                c, r = s[reg]["circuit"]["top1_agreement"], s[reg]["random_mean_top1"]
                _p(f"    specificity {reg:<8} circuit {c:.3f}  random {r:.3f}  "
                   f"gap {(r-c)*100:.1f}pp")
        if "preout" in s and "postout" in s:
            gp = s["preout"]["random_mean_top1"] - s["preout"]["circuit"]["top1_agreement"]
            gq = s["postout"]["random_mean_top1"] - s["postout"]["circuit"]["top1_agreement"]
            _p(f"    damage concentration: {gq/max(gp,1e-9):.1f}x more at post-output")
        if "self_repair" in d:
            sr = d["self_repair"]
            _p(f"    self-repair   {len(sr['repair_heads'])} backup heads, "
               f"converged={sr['converged']} in {sr['n_iters']} iters")
        if "k_curve" in d:
            _p(f"    k-curve       " +
               "  ".join(f"k={r['k']}:{r['faithfulness']:.3f}" for r in d["k_curve"]))


def universality(runs: Path) -> None:
    _p("\n" + "=" * 72)
    _p("4. UNIVERSALITY")
    _p("=" * 72)
    d = runs / "final_universality"
    if not d.exists():
        _p(f"  MISSING {d}")
        return
    ck = d / "universality_checks.json"
    if ck.exists():
        _p(f"  checks: {json.load(open(ck))}")
    for f in sorted(d.glob("cka_*.npy")):
        m = np.load(f)
        i, j = np.unravel_index(m.argmax(), m.shape)
        _p(f"  {f.stem}: shape {m.shape}  max {m.max():.3f} at layers ({i},{j})  "
           f"mean {m.mean():.3f}  diag-ish mean {np.mean([m[k, k] for k in range(min(m.shape))]):.3f}")
        # late-layer block, where the circuit lives
        la, lb = int(m.shape[0] * 0.75), int(m.shape[1] * 0.75)
        _p(f"      late-layer block (>=75% depth) mean {m[la:, lb:].mean():.3f}  "
           f"vs early block {m[:la//2, :lb//2].mean():.3f}")
    ov = d / "overlap.json"
    if ov.exists():
        for r in json.load(open(ov)):
            _p(f"  overlap {r['model_a']} vs {r['model_b']}: jaccard {r['jaccard']:.3f} "
               f"({r['n_common']} common) same_arch={r['same_arch']}")
            _p(f"      NOTE: computed over the full {r['n_a']}-node shortlist, not the confirmed heads")


def _delta_col(df: pd.DataFrame) -> str | None:
    """Whichever column holds the Δ shift; the two sweeps name it differently."""
    for c in ("delta_shift", "shift", "delta_after", "delta_mean", "delta"):
        if c in df.columns:
            return c
    cands = [c for c in df.columns if "delta" in c.lower()]
    return cands[-1] if cands else None


def intervention(inter: Path) -> None:
    """Both handles, on the held-out split. The numbers the paper quotes are the held-out
    ones; the discovery sweep only picks the operating point."""
    _p("\n" + "=" * 72)
    _p("5. INTERVENTION  (chosen on discovery, reported on held-out)")
    _p("=" * 72)
    for name in ("final_torl_steer", "final_torl_scale"):
        d = inter / name
        if not d.exists():
            _p(f"\n  MISSING {d}")
            continue
        _p(f"\n  --- {name} ---")
        for f in sorted(d.glob("*.json")):
            try:
                obj = json.load(open(f))
            except Exception as e:                                    # noqa: BLE001
                _p(f"    {f.name}: unreadable ({e})")
                continue
            _p(f"    {f.name}")
            for k, v in (obj.items() if isinstance(obj, dict) else []):
                if isinstance(v, (list, dict)) and len(str(v)) > 300:
                    _p(f"      {k}: <{type(v).__name__} of {len(v)}>")
                else:
                    _p(f"      {k}: {v}")
        for f in sorted(d.glob("*.csv")):
            df = pd.read_csv(f)
            _p(f"\n    {f.name}  ({len(df)} rows, columns {list(df.columns)})")
            if f.name == "role_transfer.csv" and "agree" in df.columns:
                _p(f"      role agreement corrupt->clean: {df['agree'].mean():.1%} "
                   f"of {len(df)} nodes")
                continue
            if f.name == "eval_validation.csv" and "delta_before" in df.columns:
                b, a = df["delta_before"], df["delta_after"]
                neg = df[b < 0]
                _p(f"      held-out n={len(df)}   Delta {b.mean():+.3f} -> {a.mean():+.3f} "
                   f"(shift {(a - b).mean():+.3f})")
                _p(f"      positive after: {(a > 0).sum()}/{len(df)}")
                # The denominator that matters. A pair already at Delta>0 cannot flip, so
                # "fraction of all pairs flipped" understates the effect and describes the
                # wrong quantity; report flips among the pairs that could flip.
                _p(f"      started negative: {len(neg)}/{len(df)}   of those, flipped positive: "
                   f"{int((neg['delta_after'] > 0).sum())}/{len(neg)}")
                if "tool_case" in df.columns:
                    _p(f"      tool_case: {dict(df['tool_case'].value_counts())}")
                continue
            dc = _delta_col(df)
            if dc and len(df) > 12:
                # the operating points that matter: strongest effect, plus the identity row
                show = df.reindex(df[dc].abs().sort_values(ascending=False).index).head(10)
                _p("      strongest by |%s|:" % dc)
                _p("      " + show.to_string(index=False).replace("\n", "\n      "))
                ident = df[(df.get("alpha", pd.Series(dtype=float)) == 1.0)
                           & (df.get("beta", pd.Series(dtype=float)) == 1.0)]
                if len(ident):
                    _p("      identity (alpha=beta=1, must be a no-op):")
                    _p("      " + ident.to_string(index=False).replace("\n", "\n      "))
            else:
                _p("      " + df.to_string(index=False, max_rows=40).replace("\n", "\n      "))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="/workspace/results", help="localization outputs per model")
    p.add_argument("--runs", default="/workspace/repo/outputs/circuits", help="validation run dirs")
    p.add_argument("--intervene", default="/workspace/repo/outputs/intervene")
    p.add_argument("--out", default=None, help="also write everything to this file")
    args = p.parse_args()

    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        delta_validity(Path(args.results))
        circuit_facts(Path(args.results))
        validation(Path(args.runs))
        universality(Path(args.runs))
        intervention(Path(args.intervene))
    text = buf.getvalue()
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
