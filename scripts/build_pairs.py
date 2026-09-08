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
import math
import os
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


def tool_answer(block):
    """The value a tool output block actually reports.

    Sandbox output is rarely a bare number. Models label their prints and print
    several lines, so the block looks like:

        'Valid colorings count: 22\\nTotal colorings: 256\\nm + n: 139'
        'Total melting: 8 cm\\nNew length after 5:00 PM: 12 cm'
        'Timeout Error'
        'Done'                  <- code ran but printed nothing

    Taking the block verbatim made almost every comparison against ground truth
    fail, which pushed nearly all pairs into 'discovery' and left the validation
    split at ~6% when the judge independently measures Tool-Ignored at 10-20%.

    Strategy: whole block first (bare numbers), then the last line, then the
    last number anywhere in it -- the answer is conventionally printed last.
    """
    block = (block or "").strip()
    if not block or block in ("Done",) or "Error" in block or "Timeout" in block:
        return None

    if parse_number(block) is not None:
        return block

    last_line = block.splitlines()[-1].strip()
    if parse_number(last_line) is not None:
        return last_line
    if ":" in last_line:
        after = last_line.rsplit(":", 1)[1].strip()
        # strip trailing units, e.g. "12 cm"
        m = re.match(r"^(-?[\d,.]+)", after)
        if m and parse_number(m.group(1)) is not None:
            return m.group(1)

    nums = re.findall(r"-?\d+(?:\.\d+)?", block)
    return nums[-1] if nums else None


def last_boxed(text):
    """Contents of the final \\boxed{...}, matching braces rather than regex.

    Returns None when the brace never closes, which means generation stopped
    mid-answer. A regex stopping at the first '}' would instead truncate
    \\boxed{\\frac{1}{2}} to '\\frac{1' and call a complete answer broken.
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
    return None  # unclosed: the trace was cut off


def same(a, b, math_equal):
    """Numeric first, symbolic fallback -- the ladder grader.py uses.

    Tolerance is RELATIVE. An absolute 1e-6 made every pair of tiny numbers
    equal: gt='1.325e-27' against a model answer of '0.000' compared True, so a
    trace where the model was flatly wrong was recorded as one where it agreed
    with the tool. isclose with abs_tol=0 scales the tolerance to the operands,
    which is what "these are the same answer" should mean.
    """
    av, bv = parse_number(a), parse_number(b)
    if av is not None and bv is not None:
        return math.isclose(av, bv, rel_tol=1e-9, abs_tol=0.0)
    try:
        return bool(math_equal(a, b))
    except Exception:
        return False


def classify_tool(answer, X, Y, math_equal):
    """What relationship did the tool have to the model and the truth?

    Four outcomes, and only the first is Tool-Ignored:

      tool_right   tool == Y   tool was right, model was wrong. The model saw
                               the correct value and overrode it. This is the
                               phenomenon; these become the validation split.
      tool_echoed  tool == X   the model reported what the tool said. Both are
                               wrong because the CODE was wrong -- no trust
                               arbitration happened at all.
      tool_other   neither     the tool said something else and the model
                               overrode it. A real conflict; the tool merely
                               happened to be wrong too.
      no_answer                "Done", "Timeout Error", nothing parseable.

    Y is tested before X so a trace can never fall in two cases. Before this
    existed the code compared against Y only, which collapsed the middle two
    into one undifferentiated "discovery" bucket -- 'the model followed a wrong
    tool' and 'the model overrode a wrong tool' are opposite behaviours, and
    lumping them hid how much of the corpus contains no conflict whatsoever.
    """
    if answer is None:
        return "no_answer"
    if same(answer, Y, math_equal):
        return "tool_right"
    if same(answer, X, math_equal):
        return "tool_echoed"
    return "tool_other"


def is_same_number(a, b):
    """Numeric equality when both parse, string equality otherwise.

    Guards Z against colliding with X. A plain string compare misses
    X="113.0" vs cand="113": different text, same value, and the corrupted run
    would silently become "tool agrees with the model" -- a different
    experimental condition wearing the label of this one.
    """
    av, bv = parse_number(a), parse_number(b)
    if av is not None and bv is not None:
        return abs(av - bv) < 1e-6
    return str(a) == str(b)


def fabricate_Zs(Y, X, n=Z_VARIANTS):
    """Up to n DISTINCT plausible wrong tool outputs: near Y, same magnitude.

    Each must differ from Y (it is the wrong value), from X (see
    is_same_number), and from the others.

    Perturbs the LAST digit, not the first. An earlier version changed the
    leading digit "so Z diverges from Y at the first token" -- but Z never
    appears in the ruler (Delta = logit(Y) - logit(X)), so its tokenisation is
    irrelevant. That constraint bought nothing and cost realism: 113 -> 213 is
    nearly double, and a model can dismiss it as absurd without any trust
    arbitration. We would then be localising "detect an implausible number"
    rather than "decide whether to believe the tool".

    Near-miss keeps the contrast intact -- in the corrupted run the tool
    endorses Z, so the model leans toward Z and logit(Y) falls regardless of
    how close Z is.

    The single shared ladder replaces two per-variant ladders that ran in
    opposite directions. Those collided whenever Y's last digit was 0 or 9 --
    one direction left the 0-9 range, so both variants fell through to the same
    shift and emitted byte-identical pairs under different ids. That is ~20% of
    answers, silently double-weighting those traces in any average.

    Editing the last digit rather than incrementing arithmetically keeps the
    change local to the string, so it works on answers that are not plain
    integers: 1.25 -> 1.26 (last decimal place), \\frac{1}{2n+2} ->
    \\frac{1}{2n+3}, 6+9i -> 6+8i. Answers with no digit at all (\\pi, x) yield
    nothing and the trace is dropped.
    """
    s = str(Y)
    digits = [i for i, ch in enumerate(s) if ch.isdigit()]
    if not digits:
        return []
    i = digits[-1]  # trailing digit: preserves order of magnitude
    d = int(s[i])
    out = []
    # Interleaved so the first two are Y+1 and Y-1 in the ordinary case.
    for shift in (1, -1, 2, -2, 3, -3):
        nd = d + shift
        if not (0 <= nd <= 9):
            continue
        cand = s[:i] + str(nd) + s[i + 1:]
        if cand == s or cand in out or is_same_number(cand, X):
            continue
        out.append(cand)
        if len(out) == n:
            break
    return out


def first_divergence(tok, X, Y):
    """Where the two answers first differ in token space -- the Delta readout.

    Returns (index, forced_prefix_text, tok_Y, tok_X, forced_prefix_ids). Any
    shared prefix is appended to the probe, identically in both runs, so the
    pair stays matched.

    The ids come back alongside the text because the text is decoded here and
    re-encoded as part of the full prompt, and that round trip is not
    guaranteed to preserve token boundaries -- the tokeniser can merge the
    prefix with the preceding "\\boxed{". The caller checks the ids survived.
    """
    tx = tok.encode(str(X), add_special_tokens=False)
    ty = tok.encode(str(Y), add_special_tokens=False)
    for i in range(min(len(tx), len(ty))):
        if tx[i] != ty[i]:
            return i, tok.decode(ty[:i]), ty[i], tx[i], list(ty[:i])
    # One is a token-prefix of the other: no position reads them apart.
    return None, None, None, None, None


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
                # Never eligible -- counted separately from genuine drops, which
                # are traces that passed eligibility and then failed a check.
                drops[key]["_not_eligible"].append((d.get("idx"), X, Y, ""))
                continue

            # Generation stopped mid-answer: "\boxed{" is present so the trace
            # passes the substring eligibility test, but the brace never closes.
            # extract_answer walks those braces too, so pred is a fragment --
            # '2' where the model was writing '22'. X feeds the ruler, so a
            # fragment there makes Delta meaningless whichever split the pair
            # lands in. 104 DemyAgent traces, ~0 for the other two.
            if last_boxed(code) is None:
                drops[key]["boxed_unclosed"].append(
                    (d.get("idx"), X, Y, code[-80:]))
                continue

            div, prefix, tok_Y, tok_X, prefix_ids = first_divergence(tok, X, Y)
            if div is None:
                drops[key]["divergence_none"].append((d.get("idx"), X, Y, code[-200:]))
                continue

            Zs = fabricate_Zs(Y, X)
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

            ids = tok.encode(text_with(Y), add_special_tokens=False)
            n_tok = len(ids)
            if n_tok > cfg["max_model_len"] - 32:
                drops[key]["too_long"].append((d.get("idx"), X, Y, f"{n_tok} tokens"))
                continue

            # The prompt must end in exactly the tokens we mean to teacher-force,
            # or Delta is read at the wrong position. prefix was decoded from Y's
            # ids and re-encoded here as part of a longer string, and that round
            # trip can merge it with the preceding "\boxed{". A no-op when
            # div == 0 (empty prefix), which is the common case. Checking the
            # clean text covers both runs: Z is the same character length as Y,
            # so everything after the output block tokenises identically.
            if div > 0 and ids[-div:] != prefix_ids:
                drops[key]["prefix_retokenised"].append(
                    (d.get("idx"), X, Y, f"div={div} prefix={prefix!r}"))
                continue

            # The split: was the tool actually right? Deterministic, no judge.
            # Compare the *extracted* answer, not the raw block -- see tool_answer().
            answer = tool_answer(tool_value)
            yv = parse_number(Y)
            case = classify_tool(answer, X, Y, math_equal)
            tool_correct = None if case == "no_answer" else (case == "tool_right")

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
                    "tool_value": tool_value, "tool_answer": answer,
                    "tool_correct": tool_correct,
                    # tool_right / tool_echoed / tool_other / no_answer. The
                    # split only distinguishes the first from the rest; this
                    # keeps the distinction the split throws away, so analysis
                    # can ask whether the circuit behaves differently on traces
                    # that contained a real conflict (tool_right, tool_other)
                    # than on ones where nothing was arbitrated (tool_echoed).
                    "tool_case": case,
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
    print(f"  tool block = {p['tool_value']!r}")
    print(f"  tool answer= {p['tool_answer']!r}  -> correct={p['tool_correct']}")
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
    args = ap.parse_args()

    # No --seed: Z generation is a deterministic function of (Y, X), and the
    # trace order is fixed by the input files. Nothing here samples.
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
        # Cases are a property of the trace, not the pair -- both Z-variants of
        # a trace share one. Count unique traces so the breakdown is not
        # silently doubled. (Not len(pairs)//Z_VARIANTS: fabricate_Zs returns
        # "up to" n, so a trace can contribute fewer.)
        cases, seen = Counter(), set()
        for p in pairs:
            tags["single_block"] += p["single_block"]
            tags["numeric"] += p["numeric"]
            tags[f"div_{min(p['readout']['divergence_index'], 3)}"] += 1
            k = (p["benchmark"], p["idx"])
            if k not in seen:
                seen.add(k)
                cases[p["tool_case"]] += 1

        n_tr = len(seen) or 1
        print(f"\n{'=' * 72}\n{key}: {len(pairs)} pairs from {len(seen)} traces")
        print(f"  discovery={splits['discovery']}  validation={splits['validation']}")
        print("  tool vs model/truth, per trace:")
        for name in ("tool_right", "tool_echoed", "tool_other", "no_answer"):
            print(f"    {name:<12}{cases[name]:>5}  ({100*cases[name]/n_tr:>5.1f}%)")
        conflict = cases["tool_right"] + cases["tool_other"]
        print(f"    real conflict (right+other): {conflict} "
              f"({100*conflict/n_tr:.1f}%)")
        print(f"  single_block={tags['single_block']}  numeric={tags['numeric']}")
        print(f"  divergence: " + "  ".join(
            f"idx{i}{'+' if i == 3 else ''}={tags[f'div_{i}']}" for i in range(4)))

        d = drops[key]
        real = {k: v for k, v in d.items() if not k.startswith("_")}
        print(f"  scanned past (never eligible): {len(d.get('_not_eligible', []))}")
        if real:
            print(f"  DROPPED after eligibility: "
                  + "  ".join(f"{k}={len(v)}" for k, v in real.items()))
            for reason, items in real.items():
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
