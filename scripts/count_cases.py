#!/usr/bin/env python3
"""How often was the tool right, echoed, overridden, or silent?

    python scripts/count_cases.py
    python scripts/count_cases.py --by-benchmark

Read-only; writes nothing and loads no tokenizer, so it runs in seconds. Needs
the gen venv only because grader.math_equal pulls in sympy.

build_pairs.py reports the same breakdown, but only over traces that survived
to become pairs -- after the divergence_none / no_digit_in_Y / too_long drops.
This measures it over every eligible trace, so the two together separate "how
often does the phenomenon occur" from "how much of it can we build pairs from".

The classification itself lives in build_pairs.py and is imported, so these
counts describe the real code path rather than a second copy of the logic.
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "generation"))

from configs.models import MODELS  # noqa: E402
from scripts.summarize_runs import latest_runs  # noqa: E402
from scripts.build_pairs import OUTPUT_BLOCK, classify_tool, tool_answer  # noqa: E402

CASES = [
    ("tool_right", "tool == Y     TOOL-IGNORED -> validation"),
    ("tool_echoed", "tool == X     model followed a wrong tool, no conflict"),
    ("tool_other", "tool != X,Y   model overrode a wrong tool, real conflict"),
    ("no_answer", "unparseable   Done / Timeout / Error"),
]


def table(c, indent=""):
    e = c["eligible"] or 1
    for name, desc in CASES:
        print(f"{indent}  {name:<12}{c[name]:>5}  ({100*c[name]/e:>5.1f}%)  {desc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--by-benchmark", action="store_true")
    args = ap.parse_args()

    from grader import math_equal

    root = Path(args.out)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    runs = [r for r in latest_runs(root) if r[2] == "-1"
            and r[0].rsplit("_seed", 1)[0] in keys]
    if not runs:
        sys.exit(f"no protocol runs (limit -1) under {root}")

    per_model = {k: Counter() for k in keys}
    per_bench = {}

    for model_dir, bench, _, path in runs:
        key = model_dir.rsplit("_seed", 1)[0]
        counters = (per_model[key], per_bench.setdefault((key, bench), Counter()))

        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            code = str((d.get("code") or [""])[0])
            X = str((d.get("pred") or [""])[0]).strip()
            Y = str(d.get("gt", "")).strip()

            for c in counters:
                c["total"] += 1
            if (d.get("score") or [None])[0] is not False:
                continue
            for c in counters:
                c["wrong"] += 1

            blocks = OUTPUT_BLOCK.findall(code)
            if not blocks or "\\boxed{" not in code:
                continue

            # The same last-block choice build_pairs.py makes.
            case = classify_tool(tool_answer(blocks[-1].strip()), X, Y, math_equal)
            for c in counters:
                c["eligible"] += 1
                c["multi_block"] += len(blocks) > 1
                c[case] += 1

    for key in keys:
        c = per_model[key]
        if not c["total"]:
            continue
        print(f"\n{'=' * 78}\n{key}\n{'=' * 78}")
        print(f"  {c['total']} traces -> {c['wrong']} wrong -> {c['eligible']} eligible"
              f"   ({c['multi_block']} with >1 output block)")
        print()
        table(c)
        conflict = c["tool_right"] + c["tool_other"]
        print(f"\n  real conflict (right+other): {conflict}"
              f"  ({100*conflict/(c['eligible'] or 1):.1f}% of eligible)")

        if args.by_benchmark:
            print()
            for (k, bench), b in sorted(per_bench.items()):
                if k == key and b["eligible"]:
                    print(f"  {bench}  ({b['eligible']} eligible)")
                    table(b, indent="  ")

    tot = Counter()
    for c in per_model.values():
        tot.update(c)
    print(f"\n{'=' * 78}\nall models: {tot['eligible']} eligible traces\n{'=' * 78}")
    table(tot)
    print("\nvalidation = tool_right only. tool_echoed and tool_other are both")
    print("'discovery' today; only tool_other is a conflict the model actually")
    print("resolved, so widening validation to include it is the lever if")
    print("tool_right alone is too thin to report on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
