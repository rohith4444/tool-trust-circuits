#!/usr/bin/env python3
"""Recompute every validation metric from validation_pairs.csv. No GPU, no model.

    python scripts/recompute_metrics.py --dir hf_data/results/torl
    python scripts/recompute_metrics.py --dir hf_data/results/torl --min-den 0.5 --json

The point of the split in circuits/validate/ablate.py: collect_deltas() runs the model once
and writes every Δ; the metrics are pure functions of that table. A metric definition can
therefore be argued about, changed and re-run on a laptop instead of costing another
~18,000 forward passes.

With --write the recomputed block is merged back into validation.json under
`recomputed`, leaving the original numbers in place for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from circuits.validate.ablate import faithfulness, k_curve, necessity   # noqa: E402


def load(path: Path, main: str | None = None, k_random: int | None = None) -> pd.DataFrame:
    """Read the per-pair table and rebuild the attrs the metric functions read.

    Column names carry the structure: `abl_<name>` / `circ_<name>` per circuit, `rand_<j>`
    per random draw. Sizes come from the sibling validation.json when it is there, and
    otherwise from the names themselves (`k50` -> 50), so the CSV alone is enough.
    """
    df = pd.read_csv(path)
    names = [c[len("circ_"):] for c in df.columns if c.startswith("circ_")]
    if not names:
        raise SystemExit(f"{path} has no circ_* columns -- not a validation_pairs.csv")
    sizes = {}
    vj = path.parent / "validation.json"
    if vj.exists():
        meta = json.load(open(vj))
        sizes = meta.get("circuit_sizes", {})
        main = main or (f"k{meta['n_nodes']}" if "n_nodes" in meta else None)
    for n in names:
        sizes.setdefault(n, int(n[1:]) if n.startswith("k") and n[1:].isdigit() else len(df))
    if k_random is None:
        k_random = sum(c.startswith("rand_") for c in df.columns)
    df.attrs.update(main=main or names[0], k_random=k_random, sizes=sizes)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="a run directory containing validation_pairs.csv")
    p.add_argument("--csv", default=None, help="override the csv path")
    p.add_argument("--circuit", default=None, help="which circuit name to report (default: the main one)")
    p.add_argument("--min-den", type=float, default=0.1)
    p.add_argument("--json", action="store_true", help="print the full metric dicts")
    p.add_argument("--write", action="store_true", help="merge the result into validation.json as `recomputed`")
    args = p.parse_args()

    d = Path(args.dir)
    df = load(Path(args.csv) if args.csv else d / "validation_pairs.csv", main=args.circuit)
    name = args.circuit or df.attrs["main"]
    nec = necessity(df, name)
    fai = faithfulness(df, name, args.min_den)

    print(f"{d}  n_pairs={len(df)}  circuit={name} ({nec['circuit_size']} nodes)  "
          f"k_random={nec['k_random']}  min_den={args.min_den}")
    print(f"  |Δ_base| {nec['abs_delta_base']:.3f} -> |Δ_ablated| {nec['abs_delta_ablated']:.3f}")
    print(f"  necessity    collapse {nec['collapse']:.3f} (median {nec['collapse_median']:.3f})  "
          f"random p95 {nec['random_collapse_p95']:.3f}  above_random={nec['above_random']}")
    print(f"  faithfulness {fai['faithfulness']:.3f}  IQR {fai['iqr']}  "
          f"pooled_signed {fai['pooled_signed']:.3f}  pooled_raw {fai['pooled_raw']:.3f}")
    print(f"  denominators +{fai['n_den_positive']} / -{fai['n_den_negative']}, "
          f"{fai['n_excluded']} excluded")

    curve = k_curve(df, args.min_den)
    if len(curve) > 1:
        print("\n" + curve.to_string(index=False))

    if args.json:
        print("\n" + json.dumps(dict(necessity=nec, faithfulness=fai), indent=2, default=str))

    if args.write:
        vj = d / "validation.json"
        obj = json.load(open(vj)) if vj.exists() else {}
        obj["recomputed"] = dict(circuit=name, min_den=args.min_den,
                                 necessity=nec, faithfulness=fai,
                                 k_curve=curve.to_dict("records"))
        with open(vj, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        print(f"\nmerged into {vj} under `recomputed`")


if __name__ == "__main__":
    main()
