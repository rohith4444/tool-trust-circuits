#!/usr/bin/env python3
"""Measure the pair-building funnel before writing the builder.

    python scripts/measure_pairs.py

Three questions this answers, all of which change the design:

1. How many eligible traces have ONE tool call vs several? With one, there is no
   ambiguity about which output block the model was reacting to. With several we
   have to pick, and picking the last one is an assumption. If single-call traces
   alone give enough volume, restricting to them is a cleaner experiment.

2. What fraction of answers are numeric vs symbolic? Symbolic answers
   ("\\frac{1}{2n+2}", "6+9i") still work for the Delta readout via
   first-divergence tokenisation, but Z fabrication needs a digit to perturb,
   and a long shared prefix makes the readout less clean.

3. Where does X diverge from Y in token space? Index 0 is ideal -- read
   immediately after "\\boxed{". Deeper means teacher-forcing a shared prefix.
   No divergence at all means the pair is unusable.

Writes nothing. Read-only measurement.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.models import MODELS  # noqa: E402
from scripts.summarize_runs import latest_runs  # noqa: E402

OUTPUT_BLOCK = re.compile(r"```output\n(.*?)\n```", re.DOTALL)


def is_numeric(s):
    """Can we treat this as a plain number? Decides whether Z fabrication by
    leading-digit perturbation applies."""
    try:
        float(str(s).replace(",", "").strip())
        return True
    except (ValueError, TypeError):
        return False


def first_divergence(tok, a, b):
    ta = tok.encode(str(a), add_special_tokens=False)
    tb = tok.encode(str(b), add_special_tokens=False)
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            return i
    return None  # one is a token-prefix of the other -- unusable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    runs = [r for r in latest_runs(Path(args.out)) if r[2] == "-1"]
    tokenizers, per_model = {}, {}

    for model_dir, bench, _, path in runs:
        key = model_dir.rsplit("_seed", 1)[0]
        if key not in MODELS:
            continue
        if key not in tokenizers:
            print(f"loading tokenizer for {key}...")
            tokenizers[key] = AutoTokenizer.from_pretrained(MODELS[key]["repo"])
            per_model[key] = Counter()
        tok, c = tokenizers[key], per_model[key]

        for line in open(path):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            code = str((d.get("code") or [""])[0])
            X = str((d.get("pred") or [""])[0])
            Y = str(d.get("gt", ""))
            score = (d.get("score") or [None])[0]

            c["total"] += 1
            if score is not False:
                continue
            c["wrong"] += 1

            blocks = OUTPUT_BLOCK.findall(code)
            if not blocks or "\\boxed{" not in code:
                continue
            c["eligible"] += 1

            # 1. tool-call multiplicity
            c[f"blocks_{min(len(blocks), 4)}"] += 1
            if len(blocks) == 1:
                c["single_block"] += 1

            # 2. answer type
            both_numeric = is_numeric(X) and is_numeric(Y)
            c["numeric" if both_numeric else "symbolic"] += 1

            # Z fabrication needs a digit somewhere to perturb
            if not re.search(r"\d", Y):
                c["no_digit_in_Y"] += 1

            # 3. divergence position
            div = first_divergence(tok, X, Y)
            if div is None:
                c["div_none"] += 1
            else:
                c[f"div_{min(div, 3)}"] += 1
                if both_numeric and len(blocks) == 1:
                    c["CLEAN_single_numeric"] += 1

    for key, c in per_model.items():
        e = c["eligible"] or 1
        print(f"\n{'=' * 64}\n{key}\n{'=' * 64}")
        print(f"  {c['total']} traces -> {c['wrong']} wrong -> {c['eligible']} eligible")
        print(f"\n  tool calls in eligible traces:")
        for n in (1, 2, 3, 4):
            v = c[f"blocks_{n}"]
            lbl = "4+" if n == 4 else str(n)
            print(f"    {lbl} block(s): {v:>4}  ({100*v//e}%)")
        print(f"\n  answer type:")
        print(f"    numeric  : {c['numeric']:>4}  ({100*c['numeric']//e}%)")
        print(f"    symbolic : {c['symbolic']:>4}  ({100*c['symbolic']//e}%)")
        print(f"    no digit in Y (Z unfabricable): {c['no_digit_in_Y']}")
        print(f"\n  X vs Y divergence in token space:")
        for i in (0, 1, 2, 3):
            v = c[f"div_{i}"]
            lbl = "3+" if i == 3 else str(i)
            print(f"    index {lbl}: {v:>4}  ({100*v//e}%)")
        print(f"    NONE (unusable): {c['div_none']}")
        print(f"\n  >>> single-block AND numeric: {c['CLEAN_single_numeric']} "
              f"({100*c['CLEAN_single_numeric']//e}% of eligible)")

    print(f"\n{'=' * 64}")
    tot_e = sum(c["eligible"] for c in per_model.values())
    tot_clean = sum(c["CLEAN_single_numeric"] for c in per_model.values())
    print(f"all eligible: {tot_e}   ->  x2 Z-variants = {tot_e*2} pairs")
    print(f"single-block numeric only: {tot_clean}  ->  x2 = {tot_clean*2} pairs")
    print("\nEAP-IG wants ~300-400 pairs for a stable ranking. If the restricted")
    print("set clears that comfortably, take it -- no ambiguity about which tool")
    print("call the model was reacting to, and a clean numeric Delta readout.")


if __name__ == "__main__":
    sys.exit(main())
