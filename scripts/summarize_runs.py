#!/usr/bin/env python3
"""Summarise generation runs: accuracy against ATTC, and the pair-building funnel.

    python scripts/summarize_runs.py
    python scripts/summarize_runs.py --out /workspace/outputs

Reads the saved JSONL directly rather than parsing logs, so nothing is lost to
terminal scrollback and the output fits on one screen.

Two things it answers:

1. The reproduction gate -- do our accuracies match ATTC's published numbers?
   Sharpest on ToRL, which is deterministic at temperature 0; the other two
   sample at temperature 1.0 so treat theirs as a sanity check, not a pass/fail.

2. The Layer 2 funnel -- how many traces survive to become matched pairs. A pair
   needs the model to be WRONG (so X != Y and the ruler does not collapse), a
   parseable tool output to inject into, and a parseable \\boxed{} to read X from.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.models import MODELS  # noqa: E402


def first(v):
    """Fields are lists (one per n_sampling); we run n=1."""
    return v[0] if isinstance(v, list) and v else v


def latest_runs(root):
    """One file per (model, benchmark, config) -- the most recent.

    Filenames are test_{prompt}_{limit}_seed{n}_t{temp}_p{top_p}_s{start}_e{end}_{stamp}.jsonl
    so the same directory accumulates validation runs (limit 20, 50) alongside
    the full run (-1), and re-running a config adds another timestamp rather
    than overwriting. Group by everything except the timestamp, keep the newest.
    """
    groups = {}
    for f in root.glob("*_seed*/*/*.jsonl"):
        model_dir, bench = f.parent.parent.name, f.parent.name
        parts = f.stem.split("_")
        # limit is the 3rd field: test_{prompt}_{limit}_...
        limit = parts[2] if len(parts) > 2 else "?"
        # strip a trailing MM-DD_HH-MM-SS stamp if present
        stamp = ""
        if len(parts) >= 2 and parts[-1].count("-") == 2 and parts[-2].count("-") == 1:
            stamp, parts = f"{parts[-2]}_{parts[-1]}", parts[:-2]
        key = (model_dir, bench, "_".join(parts))
        if key not in groups or stamp > groups[key][0]:
            groups[key] = (stamp, f, limit)
    return sorted((m, b, lim, f) for (m, b, _), (_, f, lim) in groups.items())


def summarise(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        return None

    n = len(rows)
    stat = defaultdict(int)
    for r in rows:
        code = str(first(r.get("code", "")) or "")
        correct = bool(first(r.get("score", False)))

        stat["correct"] += correct
        stat["python"] += "```python" in code
        stat["output"] += "```output" in code
        stat["boxed"] += "\\boxed{" in code
        stat["exec_err"] += first(r.get("report")) is not None

        # Layer 2 eligibility: wrong answer + a tool output to inject into
        # + a parseable own-answer to use as X.
        if (not correct) and "```output" in code and "\\boxed{" in code:
            stat["eligible"] += 1

    # Name every key explicitly: stat is a defaultdict, so a counter that was
    # never incremented (e.g. zero eligible traces in a file) would otherwise be
    # absent and blow up the formatter.
    keys = ("correct", "python", "output", "boxed", "exec_err", "eligible")
    return {"n": n, **{k: stat[k] for k in keys}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    ap.add_argument("--full-only", action="store_true",
                    help="only the protocol runs (num_test_sample = -1), "
                         "excluding the 20/50-problem validation runs")
    args = ap.parse_args()

    root = Path(args.out)
    if not root.exists():
        sys.exit(f"no such directory: {root}")

    found = latest_runs(root)
    if args.full_only:
        found = [r for r in found if r[2] == "-1"]
    if not found:
        sys.exit(f"no run files under {root} (expected <model>_seed<n>/<benchmark>/*.jsonl)")

    hdr = (f"{'model':<11}{'benchmark':<15}{'lim':>5}{'n':>5}{'acc':>7}{'ATTC':>7}"
           f"{'py':>6}{'out':>6}{'box':>6}{'err':>6}{'elig':>6}")
    print(hdr)
    print("-" * len(hdr))

    totals = defaultdict(lambda: defaultdict(int))
    for model_dir, bench, limit, f in found:
        key = model_dir.rsplit("_seed", 1)[0]
        s = summarise(f)
        if not s:
            continue

        n = s["n"]
        acc = 100.0 * s["correct"] / n
        attc = (MODELS.get(key, {}).get("attc_acc") or {}).get(bench)
        attc_s = f"{attc:.1f}" if attc is not None else "-"

        print(f"{key:<11}{bench:<15}{limit:>5}{n:>5}{acc:>6.1f}{'%':<1}{attc_s:>7}"
              f"{100*s['python']//n:>5}%{100*s['output']//n:>5}%"
              f"{100*s['boxed']//n:>5}%{100*s['exec_err']//n:>5}%"
              f"{s['eligible']:>6}")

        for k in ("n", "correct", "eligible"):
            totals[key][k] += s[k]

    print()
    print("pair-building funnel (eligible = wrong + has ```output + has \\boxed):")
    for key, t in totals.items():
        pct = 100.0 * t["eligible"] / t["n"] if t["n"] else 0
        wrong = t["n"] - t["correct"]
        print(f"  {key:<11} {t['n']:>5} traces -> {wrong:>4} wrong -> "
              f"{t['eligible']:>4} eligible ({pct:.0f}% of traces)")

    print()
    print("At 2,836 problems per model, scale the eligible rate above to project")
    print("pair yield. Each eligible trace gives 1 clean + 2 corrupted runs.")


if __name__ == "__main__":
    sys.exit(main())
