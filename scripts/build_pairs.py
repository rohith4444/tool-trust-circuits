#!/usr/bin/env python3
"""Layer 2: build matched clean/corrupted pairs from the generated traces.

    # inspect 10 per model, write nothing
    python scripts/build_pairs.py --limit 10 --dry-run

    # full build
    python scripts/build_pairs.py

Each pair is two texts differing in exactly one place -- the value inside the
last ```output block -- plus the coordinates for reading Delta:

    RULER (fixed, identical in both runs):  Delta = logit(Y) - logit(X)

    CLEAN      ```output  Y  ```  + probe   tool backs the correct value
    CORRUPTED  ```output  Z  ```  + probe   tool backs a wrong value

X is the model's own answer, Y the ground truth, Z a value we fabricate. The
ruler never swaps its two tokens; only the injected value changes. The gap in
Delta between the two runs is the tool value's causal effect, which is what
EAP-IG ranks components by and what patching flips.

Eligibility needs X != Y, so only traces the model got WRONG qualify -- with a
correct answer the ruler collapses to logit(Y) - logit(Y).
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "generation"))

from configs.models import MODELS  # noqa: E402
from scripts.summarize_runs import latest_runs  # noqa: E402

# Their reinsertion format, from math_eval.py:328
#     exec_result = f"\n```output\n{exec_result}\n```\n"
OUTPUT_BLOCK = re.compile(r"```output\n(.*?)\n```", re.DOTALL)

# One probe for all three models. Every prompt instructs the model to answer in
# \boxed{}, and the prompt is still in context (we cut after the output block,
# not before it), so this is an in-distribution continuation everywhere. Shared
# rather than per-model so the universality comparison has one fewer variable.
PROBE = "\nThe final answer is \\boxed{"

Z_VARIANTS = 2


def parse_number(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return None


def fabricate_Z(Y, X, variant):
    """A plausible wrong tool output.

    Must differ from Y (it is the wrong value) and from X (otherwise the
    corrupted run becomes "tool agrees with the model", a different condition).
    Perturbs the leading digit so it diverges from Y at the first token --
    Qwen splits numbers into single digits, so a leading-digit change is a
    first-token change.
    """
    s = str(Y)
    m = re.search(r"\d", s)
    if not m:
        return None
    i = m.start()
    d = int(s[i])
    for shift in ((1, 2, 3) if variant == 0 else (2, 3, 1)):
        nd = (d + shift) % 10
        if nd == d:
            continue
        cand = s[:i] + str(nd) + s[i + 1:]
        if cand != str(Y) and cand != str(X):
            return cand
    return None


def first_divergence(tok, X, Y):
    """Where the two answers first differ in token space -- the Delta readout.

    Returns (index, forced_prefix). Any shared prefix is appended to the probe,
    identically in both runs, so the pair stays matched.
    """
    tx = tok.encode(str(X), add_special_tokens=False)
    ty = tok.encode(str(Y), add_special_tokens=False)
    for i in range(min(len(tx), len(ty))):
        if tx[i] != ty[i]:
            return i, tok.decode(ty[:i]), ty[i], tx[i]
    return None, None, None, None  # one is a token-prefix of the other


def build_for_model(key, runs, tok, limit, drops, math_equal):
    cfg = MODELS[key]
    pairs = []

    for model_dir, bench, _, path in runs:
        n_from_file = 0
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

            blocks = list(OUTPUT_BLOCK.finditer(code))
            if not blocks or "\\boxed{" not in code:
                drops[key]["no_output_or_boxed"].append((d.get("idx"), X, Y, ""))
                continue

            div, prefix, tok_Y, tok_X = first_divergence(tok, X, Y)
            if div is None:
                drops[key]["divergence_none"].append((d.get("idx"), X, Y, code[-200:]))
                continue

            Zs = [z for z in (fabricate_Z(Y, X, v) for v in range(Z_VARIANTS)) if z]
            if not Zs:
                drops[key]["no_digit_in_Y"].append((d.get("idx"), X, Y, ""))
                continue

            # Inject into the LAST block: it is the result the model saw most
            # recently before committing to \boxed{}. 67-88% of eligible traces
            # have only one block, so for most there is no choice to make.
            last = blocks[-1]
            head = code[:last.start()]
            tool_value = last.group(1).strip()

            def text_with(v):
                return f"{head}```output\n{v}\n```\n{PROBE}{prefix}"

            n_tok = len(tok.encode(text_with(Y), add_special_tokens=False))
            if n_tok > cfg["max_model_len"] - 32:
                drops[key]["too_long"].append((d.get("idx"), X, Y, f"{n_tok} tokens"))
                continue

            # The split: was the tool actually right? Deterministic, no judge.
            tv = parse_number(tool_value)
            yv = parse_number(Y)
            if tv is not None and yv is not None:
                tool_correct = abs(tv - yv) < 1e-6
            else:
                try:
                    tool_correct = bool(math_equal(tool_value, Y))
                except Exception:
                    tool_correct = None

            for vi, Z in enumerate(Zs):
                pairs.append({
                    "pair_id": f"{key}_{bench}_{d.get('idx')}_z{vi}",
                    "model": key, "benchmark": bench, "idx": d.get("idx"),
                    "X": X, "Y": Y, "Z": Z,
                    "clean_text": text_with(Y),
                    "corrupted_text": text_with(Z),
                    "readout": {"forced_prefix": prefix, "tok_Y": tok_Y,
                                "tok_X": tok_X, "divergence_index": div},
                    # validation = real Tool-Ignored (tool was right, model wrong).
                    # unparseable tool output -> discovery, conservatively.
                    "split": "validation" if tool_correct else "discovery",
                    "tool_value": tool_value, "tool_correct": tool_correct,
                    "single_block": len(blocks) == 1,
                    "numeric": parse_number(X) is not None and yv is not None,
                    "n_output_blocks": len(blocks),
                    "n_tokens": n_tok,
                })

            n_from_file += 1
            if limit and n_from_file >= limit:
                break
    return pairs


def show(p):
    print(f"\n{'=' * 72}\n{p['pair_id']}   split={p['split']}   "
          f"blocks={p['n_output_blocks']}  numeric={p['numeric']}  "
          f"tokens={p['n_tokens']}\n{'=' * 72}")
    print(f"  X (model)  = {p['X']!r}")
    print(f"  Y (truth)  = {p['Y']!r}")
    print(f"  Z (fake)   = {p['Z']!r}")
    print(f"  tool said  = {p['tool_value']!r}  -> correct={p['tool_correct']}")
    r = p["readout"]
    print(f"  Delta = logit(tok {r['tok_Y']}) - logit(tok {r['tok_X']}) "
          f"at divergence index {r['divergence_index']}, "
          f"forced prefix {r['forced_prefix']!r}")
    print(f"\n  --- CLEAN, last 400 chars ---\n  "
          + p["clean_text"][-400:].replace("\n", "\n  "))
    print(f"\n  --- CORRUPTED, last 400 chars ---\n  "
          + p["corrupted_text"][-400:].replace("\n", "\n  "))

    # The pair is only valid if the two texts differ in exactly one place.
    c, x = p["clean_text"], p["corrupted_text"]
    i = next((k for k in range(min(len(c), len(x))) if c[k] != x[k]), None)
    print(f"\n  first byte difference at {i} "
          f"(head identical: {'YES' if i and i > len(c) - 300 else 'CHECK'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="traces per file; 0 = all")
    ap.add_argument("--dry-run", action="store_true",
                    help="print pairs in full, write nothing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    from transformers import AutoTokenizer
    from grader import math_equal

    root = Path(args.out)
    all_runs = [r for r in latest_runs(root) if r[2] == "-1"]
    keys = [k.strip() for k in args.models.split(",") if k.strip()]

    drops = defaultdict(lambda: defaultdict(list))
    by_model = {}

    for key in keys:
        runs = [r for r in all_runs if r[0].rsplit("_seed", 1)[0] == key]
        if not runs:
            continue
        print(f"loading tokenizer for {key}...")
        tok = AutoTokenizer.from_pretrained(MODELS[key]["repo"])
        by_model[key] = build_for_model(key, runs, tok, args.limit, drops, math_equal)

    # ---- report ----
    for key, pairs in by_model.items():
        splits = Counter(p["split"] for p in pairs)
        tags = Counter()
        for p in pairs:
            tags["single_block"] += p["single_block"]
            tags["numeric"] += p["numeric"]
            tags[f"div_{min(p['readout']['divergence_index'], 3)}"] += 1

        print(f"\n{'=' * 72}\n{key}: {len(pairs)} pairs "
              f"({len(pairs)//Z_VARIANTS} traces x {Z_VARIANTS} Z-variants)")
        print(f"  discovery={splits['discovery']}  validation={splits['validation']}")
        print(f"  single_block={tags['single_block']}  numeric={tags['numeric']}")
        print(f"  divergence: " + "  ".join(
            f"idx{i}{'+' if i == 3 else ''}={tags[f'div_{i}']}" for i in range(4)))

        d = drops[key]
        if d:
            print(f"  DROPPED: " + "  ".join(f"{k}={len(v)}" for k, v in d.items()))
            for reason, items in d.items():
                for idx, X, Y, extra in items[:3]:
                    print(f"    [{reason}] idx={idx} X={X!r} Y={Y!r} {extra[:80]!r}")

    if args.dry_run:
        for key, pairs in by_model.items():
            for p in pairs[:2 * Z_VARIANTS]:
                show(p)
        print("\n\nDRY RUN -- nothing written.")
        return 0

    pair_dir = root / "pairs"
    pair_dir.mkdir(exist_ok=True)
    for key, pairs in by_model.items():
        for split in ("discovery", "validation"):
            sel = [p for p in pairs if p["split"] == split]
            if not sel:
                continue
            path = pair_dir / f"{key}_{split}.jsonl"
            with open(path, "w") as f:
                for p in sel:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print(f"wrote {len(sel):>5}  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
