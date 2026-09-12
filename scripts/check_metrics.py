#!/usr/bin/env python3
"""Check the pure metric functions in circuits/validate/ablate.py. No GPU, no model.

    python scripts/check_metrics.py

Runs on synthetic Δ tables built with the sign mix the real runs show (ToRL: mean Δ −1.54,
mean |Δ| ≥3.47), because that mix is what broke the previous metrics. Four identities have
to hold exactly, and the realistic case has to reproduce the failure the old code showed:

  circuit = universe     -> faithfulness 1
  circuit = empty        -> faithfulness 0
  ablate nothing         -> collapse 0
  ablate to zero         -> collapse 1

Exits non-zero if any identity is off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circuits.validate.ablate import (faithfulness, k_curve, necessity,   # noqa: E402
                                      specificity_contrast)

TOL = 1e-6
failures: list[str] = []


def check(label: str, got: float, want: float) -> None:
    ok = abs(got - want) < TOL
    print(f"  {'ok ' if ok else 'FAIL'}  {label:<22} {got:.6f}  (expect {want})")
    if not ok:
        failures.append(label)


def main() -> None:
    rng = np.random.default_rng(0)
    n = 615
    base = rng.normal(-1.54, 3.2, n)          # the real sign mix
    alls = rng.normal(0.0, 0.5, n)            # the everything-ablated floor

    df = pd.DataFrame({"base": base, "all_abl": alls})
    df.attrs.update(main="k50", k_random=20, sizes={"k50": 50})
    print(f"synthetic n={n}  mean(Δ)={base.mean():+.3f}  mean|Δ|={np.abs(base).mean():.3f}  "
          f"({np.abs(base).mean() / abs(base.mean()):.1f}x cancellation)\n")

    print("identities")
    df["circ_k50"], df["abl_k50"] = base, alls
    check("faithfulness(all)", faithfulness(df)["faithfulness"], 1.0)
    df["circ_k50"] = alls
    check("faithfulness(empty)", faithfulness(df)["faithfulness"], 0.0)
    df["abl_k50"] = base
    check("collapse(nothing)", necessity(df, k_random=0)["collapse"], 0.0)
    df["abl_k50"] = np.zeros(n)
    check("collapse(to zero)", necessity(df, k_random=0)["collapse"], 1.0)

    # realistic: the circuit kills Δ, random sets of the same size barely move it,
    # and the circuit alone reproduces ~30% of the gap between floor and base.
    df["abl_k50"] = rng.normal(0.0, 0.05, n)
    for j in range(20):
        df[f"rand_{j}"] = base + rng.normal(0, 0.3, n)
    df["circ_k50"] = alls + 0.30 * (base - alls) + rng.normal(0, 0.4, n)

    nec, fai = necessity(df), faithfulness(df)
    print("\nrealistic case")
    print(f"  collapse             {nec['collapse']:.3f}   median per-pair {nec['collapse_median']:.3f}")
    print(f"  random collapse p95  {nec['random_collapse_p95']:.3f}")
    print(f"  above_random         {nec['above_random']}")
    print(f"  faithfulness         {fai['faithfulness']:.3f}   IQR "
          f"{[round(x, 3) for x in fai['iqr']]}")
    print(f"  pooled_signed        {fai['pooled_signed']:.3f}")
    print(f"  pooled_raw (OLD)     {fai['pooled_raw']:.3f}   <- what the old code reported")
    print(f"  denominator signs    +{fai['n_den_positive']} / -{fai['n_den_negative']}, "
          f"{fai['n_excluded']} excluded")
    if not nec["above_random"]:
        failures.append("above_random should be True when the circuit collapses Δ and random does not")
    if abs(fai["faithfulness"] - 0.30) > 0.10:
        failures.append(f"faithfulness {fai['faithfulness']:.3f} should recover the planted 0.30")

    for name, frac, size in (("k10", 0.10, 10), ("k25", 0.20, 25),
                             ("k100", 0.45, 100), ("k200", 0.70, 200)):
        df[f"circ_{name}"] = alls + frac * (base - alls) + rng.normal(0, 0.4, n)
        df[f"abl_{name}"] = rng.normal(0, 0.05 + 0.3 * (1 - frac), n)
        df.attrs["sizes"][name] = size
    curve = k_curve(df)
    print("\nk-curve (planted faithfulness 0.10 / 0.20 / 0.30 / 0.45 / 0.70)")
    print(curve.to_string(index=False))
    if not curve["faithfulness"].is_monotonic_increasing:
        failures.append("k-curve faithfulness is not monotonic in k")

    print("\nspecificity contrast")
    print(" ", {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in specificity_contrast(nec, dict(collapse=0.21, n_pairs=362)).items()})

    if failures:
        print(f"\n{len(failures)} FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
