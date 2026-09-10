#!/usr/bin/env python3
"""Does Z actually move the metric? Measures the flaw before fixing it.

    python scripts/check_z_contrast.py --pairs ./hf_pairs/pairs

Read-only, CPU only. Loads tokenisers (a few MB each), never model weights.

THE PROBLEM

Delta is read at `divergence_index` -- the token position where the model's own
answer X first differs from the ground truth Y:

    Delta = logit(Y[div]) - logit(X[div])

Z is currently built by perturbing Y's LAST digit. When Y has more than one
token and div lands earlier, Z shares Y's token at exactly the position the
metric reads:

    X='49'  Y='113'  Z='114'  div=0
      clean:     tool says 113 -> model copies -> token '1'
      corrupted: tool says 114 -> model copies -> token '1'   <-- identical

Both runs push the same token at the readout, so Delta(clean) ~ Delta(corrupted)
and there is no contrast for EAP-IG to attribute. The pair is inert.

It works only when div happens to land on Y's last token -- single-digit
answers, mostly:

    X='12'  Y='8'  Z='9'  div=0    -> logit('8') genuinely collapses. Fine.

THE FIX THIS MEASURES

Perturb the digit at `div` instead of the last one. Same token count (so
activation patching stays position-aligned), but guaranteed to differ where the
metric looks. For each broken pair this script searches for such a Z and
reports whether one exists, which is the same search the fixed fabricate_Zs
will do.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.models import MODELS  # noqa: E402

SHIFTS = (1, -1, 2, -2, 3, -3)


def perturb(s, char_pos, shift):
    """Y with the digit at char_pos shifted, or None if out of range."""
    d = int(s[char_pos]) + shift
    if not (0 <= d <= 9):
        return None
    return s[:char_pos] + str(d) + s[char_pos + 1:]


def find_contrasting_z(tok, Y, X, div, ids_Y):
    """A Z that keeps Y's token count but differs from Y at `div`.

    Tries every digit position rather than assuming digits map 1:1 to tokens --
    Qwen splits plain integers digit-by-digit, but Y can be symbolic
    (\\frac{1}{2n+2}) where the mapping is not positional.
    """
    for pos in (i for i, ch in enumerate(Y) if ch.isdigit()):
        for shift in SHIFTS:
            cand = perturb(Y, pos, shift)
            if cand is None or cand == Y or cand == X:
                continue
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == len(ids_Y) and div < len(ids) and ids[div] != ids_Y[div]:
                return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="./hf_pairs/pairs",
                    help="directory of *_discovery.jsonl / *_validation.jsonl")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    pair_dir = Path(args.pairs)
    files = sorted(pair_dir.glob("*.jsonl"))
    if not files:
        sys.exit(f"no pair files under {pair_dir}")

    tokenizers = {}
    stats = defaultdict(Counter)
    examples = defaultdict(list)

    for f in files:
        split = "validation" if "validation" in f.name else "discovery"
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            key = p["model"]
            if key not in tokenizers:
                print(f"loading tokenizer for {key}...")
                tokenizers[key] = AutoTokenizer.from_pretrained(MODELS[key]["repo"])
            tok = tokenizers[key]

            X, Y, Z = p["X"], p["Y"], p["Z"]
            div = p["readout"]["divergence_index"]
            ids_Y = tok.encode(Y, add_special_tokens=False)
            ids_Z = tok.encode(Z, add_special_tokens=False)

            c = stats[key]
            c["total"] += 1
            c[f"{split}_total"] += 1
            c[f"ylen_{min(len(ids_Y), 4)}"] += 1
            c[f"div_{min(div, 3)}"] += 1

            if len(ids_Y) != len(ids_Z):
                # Breaks activation patching: positions after the injected value
                # would no longer align between the clean and corrupted runs.
                c["length_mismatch"] += 1

            if div >= len(ids_Z) or ids_Z[div] == ids_Y[div]:
                c["BROKEN"] += 1
                c[f"{split}_broken"] += 1
                fixed = find_contrasting_z(tok, Y, X, div, ids_Y)
                if fixed:
                    c["fixable"] += 1
                    if len(examples[key]) < args.examples:
                        examples[key].append((X, Y, Z, div, fixed))
                else:
                    c["unfixable"] += 1
                    if len(examples[f"{key}_bad"]) < args.examples:
                        examples[f"{key}_bad"].append((X, Y, Z, div, None))
            else:
                c["ok"] += 1

    grand = Counter()
    for key, c in stats.items():
        grand.update(c)
        n = c["total"] or 1
        print(f"\n{'=' * 76}\n{key}   {c['total']} pairs "
              f"({c['discovery_total']} discovery, {c['validation_total']} validation)"
              f"\n{'=' * 76}")
        print(f"  OK       {c['ok']:>5}  ({100*c['ok']/n:>5.1f}%)   Z differs from Y at the readout")
        print(f"  BROKEN   {c['BROKEN']:>5}  ({100*c['BROKEN']/n:>5.1f}%)   Z shares Y's token there -- no contrast")
        print(f"    fixable   {c['fixable']:>5}   a same-length Z exists that diverges at div")
        print(f"    unfixable {c['unfixable']:>5}   no digit perturbation works")
        print(f"  length mismatch Y vs Z: {c['length_mismatch']}")
        print(f"\n  broken by split: discovery={c['discovery_broken']}  "
              f"validation={c['validation_broken']}")
        print(f"  Y token length: " + "  ".join(
            f"{i}{'+' if i == 4 else ''}={c[f'ylen_{i}']}" for i in range(1, 5)))
        print(f"  divergence idx: " + "  ".join(
            f"{i}{'+' if i == 3 else ''}={c[f'div_{i}']}" for i in range(4)))

        for X, Y, Z, div, fixed in examples[key]:
            print(f"    FIX  X={X!r} Y={Y!r} div={div}   {Z!r} -> {fixed!r}")
        for X, Y, Z, div, _ in examples[f"{key}_bad"]:
            print(f"    NOFIX X={X!r} Y={Y!r} div={div}   Z={Z!r}")

    n = grand["total"] or 1
    print(f"\n{'=' * 76}")
    print(f"all models: {grand['total']} pairs")
    print(f"  OK       {grand['ok']:>5}  ({100*grand['ok']/n:.1f}%)")
    print(f"  BROKEN   {grand['BROKEN']:>5}  ({100*grand['BROKEN']/n:.1f}%)"
          f"   of which {grand['fixable']} fixable, {grand['unfixable']} not")
    print(f"\nBroken pairs are inert, not wrong -- Delta simply does not move, so")
    print(f"they contribute no signal to EAP-IG. Fixing means rebuilding with a Z")
    print(f"perturbed at `div`; unfixable ones get dropped like any other gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
