#!/usr/bin/env python3
"""Did the answer parser fail on traces we are calling wrong?

    python scripts/check_parse.py
    python scripts/check_parse.py --examples 5

Read-only. Writes nothing, loads no model.

grader.py scores a trace by comparing the parsed `pred` field to `gt`. If that
parse fails, a correct answer is recorded as wrong -- and a wrong answer is
exactly our eligibility criterion, so the trace enters the pair corpus.

DemyAgent repeats its final answer degenerately:

    ...The answer is $\\boxed{694}$.\\n\\n$\\boxed{694}$...     pred=''   gt='694'
    \\boxed{22}22\\n\\boxed{22}22\\n\\boxed{22}22...            pred='2'  gt='22'

Both were right and both were scored wrong. This pulls the last \\boxed{...}
straight out of the trace text and compares THAT to the ground truth, which is
independent of whatever pred holds.

Two things it measures:

  1. How many "wrong" traces were actually right -- inflating the wrong count,
     and with it any accuracy number computed from these files.

  2. How many tool_right (validation) traces are contaminated. A mis-parsed
     pred makes the model look like it disagreed with a correct tool, which is
     indistinguishable from real Tool-Ignored unless you check the raw text.
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
from scripts.build_pairs import (  # noqa: E402
    OUTPUT_BLOCK, classify_tool, same, tool_answer,
)


def last_boxed(text):
    """Contents of the final \\boxed{...}, matching braces rather than regex.

    A regex stopping at the first '}' truncates \\boxed{\\frac{1}{2}} to
    '\\frac{1', which would look like a parse failure of our own making.
    """
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth, out = i + len("\\boxed{"), 1, []
    while j < len(text):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
        j += 1
    return None  # unclosed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    from grader import math_equal

    root = Path(args.out)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    runs = [r for r in latest_runs(root) if r[2] == "-1"
            and r[0].rsplit("_seed", 1)[0] in keys]
    if not runs:
        sys.exit(f"no protocol runs (limit -1) under {root}")

    per_model = {k: Counter() for k in keys}
    examples = {k: [] for k in keys}

    for model_dir, bench, _, path in runs:
        key = model_dir.rsplit("_seed", 1)[0]
        c = per_model[key]

        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            code = str((d.get("code") or [""])[0])
            X = str((d.get("pred") or [""])[0]).strip()
            Y = str(d.get("gt", "")).strip()

            if (d.get("score") or [None])[0] is not False:
                continue
            c["wrong"] += 1
            if not X:
                c["pred_empty"] += 1

            boxed = last_boxed(code)
            if boxed is None:
                c["no_boxed"] += 1
                continue
            boxed = boxed.strip()

            # The trace's own final answer agrees with the ground truth, but the
            # trace was scored wrong. The parse is the only thing in between.
            mis = same(boxed, Y, math_equal)
            if mis:
                c["parse_failure"] += 1

            # Same eligibility gate build_pairs.py applies, so the validation
            # contamination number below is about traces that really would have
            # become pairs.
            blocks = OUTPUT_BLOCK.findall(code)
            if not blocks or "\\boxed{" not in code:
                continue
            c["eligible"] += 1
            if classify_tool(tool_answer(blocks[-1].strip()), X, Y, math_equal) \
                    != "tool_right":
                continue
            c["tool_right"] += 1
            if mis:
                c["tool_right_bogus"] += 1
                if len(examples[key]) < args.examples:
                    examples[key].append((bench, d.get("idx"), X, Y, boxed))

    for key in keys:
        c = per_model[key]
        if not c["wrong"]:
            continue
        w, tr = c["wrong"], c["tool_right"] or 1
        print(f"\n{'=' * 74}\n{key}\n{'=' * 74}")
        print(f"  {w} traces scored wrong")
        print(f"    pred empty          {c['pred_empty']:>5}  ({100*c['pred_empty']/w:>5.1f}%)")
        print(f"    no \\boxed in trace  {c['no_boxed']:>5}  ({100*c['no_boxed']/w:>5.1f}%)")
        print(f"    PARSE FAILURE       {c['parse_failure']:>5}  ({100*c['parse_failure']/w:>5.1f}%)"
              f"   last \\boxed == gt, but scored wrong")
        print(f"\n  {c['tool_right']} of {c['eligible']} eligible are tool_right (validation)")
        print(f"    of those, bogus     {c['tool_right_bogus']:>5}  "
              f"({100*c['tool_right_bogus']/tr:>5.1f}%)   model was right all along")
        clean = c["tool_right"] - c["tool_right_bogus"]
        print(f"    surviving validation{clean:>5}")

        for bench, idx, X, Y, boxed in examples[key]:
            print(f"    e.g. {bench}/{idx}: pred={X!r} gt={Y!r} but trace ends \\boxed{{{boxed}}}")

    tot = Counter()
    for c in per_model.values():
        tot.update(c)
    print(f"\n{'=' * 74}")
    print(f"all models: {tot['parse_failure']} of {tot['wrong']} wrong traces were "
          f"actually right ({100*tot['parse_failure']/(tot['wrong'] or 1):.1f}%)")
    print(f"validation: {tot['tool_right']} tool_right, {tot['tool_right_bogus']} bogus, "
          f"{tot['tool_right'] - tot['tool_right_bogus']} surviving")
    return 0


if __name__ == "__main__":
    sys.exit(main())
