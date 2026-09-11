#!/usr/bin/env python3
"""What shape are the ground-truth answers, and can we fake each one?

    python scripts/count_answer_shapes.py --pairs /workspace/outputs/pairs

Read-only. Loads tokenisers only, never model weights. Under a minute.

BACKGROUND

The score compares the truth Y against the model's own answer X, at the first
character position where they differ -- call it the score position:

    X=49  Y=113  ->  differ at position 0     score = logit('1') - logit('4')
    X=30  Y=315  ->  share the '3'            score = read at position 1

The fake tool value Z has to differ from Y AT THAT POSITION. Otherwise both the
clean and corrupted runs write the same character there and the score is
identical in both -- the corruption is invisible to the metric.

    Y=113  Z=114  ->  both start '1'  ->  score unchanged  ->  pair is inert
    Y=113  Z=213  ->  '1' vs '2'      ->  score drops      ->  pair works

So whether a pair is salvageable comes down to one question: is the character
at the score position a digit we can change?

FOUR BUCKETS

    A  single digit          8               already works, nothing to change
    B  multi digit,          113, 315,       FIXABLE -- change the digit at the
       digit at score pos    1.25, 25000     score position
    C  non-digit at          -1, \\sqrt{51},  CANNOT FIX -- no digit change
       score position        6,600           reaches that position; drop
    D  no digits at all      even, (B)       already dropped at build time

Reported per model and per split, because validation is the thin one and its
survival is what decides whether the rebuild is worth doing as-is.
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.models import MODELS  # noqa: E402

BUCKETS = [
    ("A single digit", "8", "already works"),
    ("B fixable", "113 -> 213", "change digit at score pos"),
    ("C non-digit", "-1, \\sqrt{51}", "cannot fix, drop"),
]


def find_fake(tok, Y, X, pos, ids_Y):
    """A Z that changes one digit of Y, keeps Y's token count, and differs from
    Y at the score position. Returns it, or None if no such Z exists.

    Tries every digit position rather than assuming digit i is token i -- Qwen
    splits plain integers digit-by-digit, but Y can be symbolic
    (\\frac{1}{2n+2}) where that mapping does not hold. Shifts are +-1 first, so
    the value found is the smallest change the constraints allow.
    """
    for i, ch in enumerate(Y):
        if not ch.isdigit():
            continue
        for shift in (1, -1, 2, -2, 3, -3):
            d = int(ch) + shift
            if not (0 <= d <= 9):
                continue
            cand = Y[:i] + str(d) + Y[i + 1:]
            if cand == Y or cand == X:
                continue
            # A printed number carries no leading zero: "013" is a shape no
            # sandbox would emit.
            if len(cand) > 1 and cand[0] == "0" and cand[1].isdigit():
                continue
            ids = tok.encode(cand, add_special_tokens=False)
            if len(ids) == len(ids_Y) and pos < len(ids) and ids[pos] != ids_Y[pos]:
                return cand
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/workspace/outputs/pairs")
    ap.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    files = sorted(Path(args.pairs).glob("*.jsonl"))
    if not files:
        sys.exit(f"no pair files under {args.pairs}")

    tokenizers = {}
    counts = defaultdict(Counter)          # (model, split) -> bucket counts
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
            pos = p["readout"]["divergence_index"]
            ids_Y = tok.encode(Y, add_special_tokens=False)
            ids_Z = tok.encode(Z, add_special_tokens=False)

            c = counts[(key, split)]
            c["total"] += 1

            already_ok = (len(ids_Z) > pos and ids_Z[pos] != ids_Y[pos])
            if already_ok:
                c["A single digit" if len(ids_Y) == 1 else "A already works"] += 1
                continue

            fake = find_fake(tok, Y, X, pos, ids_Y)
            if fake:
                c["B fixable"] += 1
                if len(examples[key]) < args.examples:
                    examples[key].append((X, Y, Z, pos, fake))
            else:
                c["C non-digit"] += 1
                if len(examples[key + "_C"]) < args.examples:
                    examples[key + "_C"].append((X, Y, Z, pos, None))

    def row(label, c):
        t = c["total"] or 1
        a = c["A single digit"] + c["A already works"]
        b, cc = c["B fixable"], c["C non-digit"]
        print(f"  {label:<13}{c['total']:>7}{a:>10} ({100*a/t:>4.0f}%)"
              f"{b:>10} ({100*b/t:>4.0f}%){cc:>9} ({100*cc/t:>4.0f}%)"
              f"{a+b:>10} ({100*(a+b)/t:>4.0f}%)")

    hdr = (f"  {'':<13}{'pairs':>7}{'A works':>17}{'B fixable':>17}"
           f"{'C drop':>16}{'USABLE':>17}")

    grand = Counter()
    for key in MODELS:
        if not any(m == key for m, _ in counts):
            continue
        print(f"\n{'=' * 82}\n{key}\n{'=' * 82}")
        print(hdr)
        tot = Counter()
        for split in ("discovery", "validation"):
            c = counts.get((key, split))
            if c:
                row(split, c)
                tot.update(c)
        row("both", tot)
        grand.update(tot)

        for X, Y, Z, pos, fake in examples[key]:
            print(f"    B  X={X!r} Y={Y!r} score at pos {pos}:  {Z!r} -> {fake!r}")
        for X, Y, Z, pos, _ in examples[key + "_C"]:
            print(f"    C  X={X!r} Y={Y!r} score at pos {pos} is not a digit  (Z={Z!r})")

    print(f"\n{'=' * 82}\nALL MODELS\n{'=' * 82}")
    print(hdr)
    row("everything", grand)
    a = grand["A single digit"] + grand["A already works"]
    print(f"\n  of the {a} in A, {grand['A single digit']} are single-digit answers")
    print(f"  ({grand['A single digit'] and 'the 8 -> 9 case, already correct'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
