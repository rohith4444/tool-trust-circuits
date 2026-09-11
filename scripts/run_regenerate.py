#!/usr/bin/env python3
"""Stage 5 -- End-task evaluation: re-run the benchmarks with an intervention active.

    # the gate: 20 traces, intervention OFF, must reproduce the baseline
    python scripts/run_regenerate.py --model torl --method none \\
        --traces /workspace/outputs/torl_seed0 --limit 20 \\
        --out outputs/regen/torl_alpha0 -v

    # head-scaling, all benchmarks
    python scripts/run_regenerate.py --model torl --method scale \\
        --circuit /workspace/results/torl/circuit.json --policy attn_role --alpha 5 \\
        --traces /workspace/outputs/torl_seed0 \\
        --out outputs/regen/torl_scale -v

Writes trace-shaped JSONL mirroring the input layout, so the existing scorers apply
unchanged:

    python scripts/summarize_runs.py --out outputs/regen/torl_scale   # accuracy
    python scripts/run_judge.py     --out outputs/regen/torl_scale    # Tool-Ignored
    python scripts/count_cases.py   --out outputs/regen/torl_scale    # tool_case

Using the same scorers as the baseline is the point: no part of the before/after
difference can then come from measuring the two arms differently.

THE GATE

`--method none` regenerates with no intervention. At temperature 0 that should reproduce
the baseline traces. It will not be byte-identical -- vLLM and TransformerLens use
different kernels and bf16 rounding, and greedy decoding can diverge where two logits are
close -- so the check is whether most traces reach the same final answer:

    18-20 of 20 agree  ->  numerics, proceed
    ~10 of 20          ->  look closer
    2  of 20           ->  a bug: prompt, stop rule, or injection

It also decides how much has to run. If the baseline reproduces, compare existing vLLM
traces against intervened TL traces. If it does not, that comparison confounds the engine
with the intervention and the baseline has to be regenerated under TL as well.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "generation"))

from circuits.cli import add_common, push_outputs, setup_logging, write_json  # noqa: E402
from circuits.hooks import set_seed  # noqa: E402
from circuits.intervene import regenerate, scale  # noqa: E402
from configs.models import MODELS  # noqa: E402


def load_traces(root: Path, limit: int | None):
    """(benchmark, rows) from the baseline run, newest file per benchmark."""
    from scripts.summarize_runs import latest_runs
    out = []
    for model_dir, bench, lim, path in latest_runs(root.parent):
        if lim != "-1" or model_dir != root.name:
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        out.append((bench, rows[:limit] if limit else rows))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common(p, model_arg=True, pairs_arg=False)   # this stage reads traces, not pairs
    p.add_argument("--traces", required=True,
                   help="baseline run dir, e.g. /workspace/outputs/torl_seed0")
    p.add_argument("--method", required=True, choices=("none", "steer", "scale"),
                   help="none = no intervention, the reproduction gate")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--direction", help="directions.pt (method=steer)")
    p.add_argument("--layer", type=int, help="layer to steer at (method=steer)")
    p.add_argument("--circuit", help="circuit.json (method=scale)")
    p.add_argument("--policy", default="attn_role", choices=("dla_sign", "attn_role"))
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--all-nodes", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="problems per benchmark")
    p.add_argument("--max-func-call", type=int, default=regenerate.MAX_FUNC_CALL)
    p.add_argument("--max-tokens-per-call", type=int, default=regenerate.MAX_TOKENS_PER_CALL)
    p.add_argument("--no-cache", action="store_true",
                   help="re-run the whole sequence every token instead of using a KV "
                        "cache. Hundreds of times slower and OOMs on long turns; only for "
                        "checking the cached path produces identical text")
    args = p.parse_args()
    setup_logging(args.verbose)
    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from grader import math_equal
    from parser import extract_answer
    from python_executor import PythonExecutor
    from trajectory import extract_program

    from circuits.models.load import load_model
    model = load_model(args.model, device=args.device, dtype=args.dtype)
    prompt_type = MODELS[args.model]["prompt_type"]
    executor = PythonExecutor(get_answer_from_stdout=True)

    # ---- the intervention ----
    if args.method == "none":
        hooks_at = lambda cut, n: []                      # noqa: E731 -- the gate
        label = "none (reproduction gate)"
    elif args.method == "steer":
        vec = torch.load(args.direction)["delta_split"][args.layer].to(model.cfg.device)
        hooks_at = regenerate.hooks_factory("steer", layer=args.layer, vec=vec, alpha=args.alpha)
        label = f"steer layer {args.layer} alpha {args.alpha}"
    else:
        circuit = scale.load_circuit(args.circuit)
        nodes = scale.select(circuit, confirmed_only=not args.all_nodes)
        scales = scale.policy(nodes, args.alpha, args.beta, args.policy)
        counts = scale.policy_counts(scales, args.alpha, args.beta)
        hooks_at = regenerate.hooks_factory("scale", scales=scales)
        label = (f"scale {args.policy} alpha {args.alpha} beta {args.beta} "
                 f"({counts['n_up']} up, {counts['n_down']} down)")
    print(f"intervention: {label}\nprompt_type: {prompt_type}\n")

    benches = load_traces(Path(args.traces), args.limit)
    if not benches:
        raise SystemExit(f"no protocol traces under {args.traces}")

    totals = {"n": 0, "correct": 0, "same_as_baseline": 0, "t": 0.0}
    for bench, rows in benches:
        bdir = out / Path(args.traces).name / bench
        bdir.mkdir(parents=True, exist_ok=True)
        written, t0 = [], time.time()
        for row in rows:
            gt = str(row.get("gt", ""))
            r = regenerate.run_one(model, executor, extract_program,
                                   row.get("question", ""), prompt_type, hooks_at,
                                   args.max_func_call, args.max_tokens_per_call,
                                   use_cache=not args.no_cache)
            pred = extract_answer(r["code"], bench)
            try:
                score = bool(math_equal(pred, gt))
            except Exception:
                score = False
            base_pred = (row.get("pred") or [""])[0]
            written.append(dict(
                idx=row.get("idx"), question=row.get("question", ""), gt=gt,
                gt_cot=row.get("gt_cot", ""),
                # Lists to match the baseline schema (n_sampling>1 support), so
                # summarize_runs / run_judge / count_cases read these unchanged.
                code=[r["code"]], pred=[pred], score=[score], report=[None],
                n_tool_calls=r["n_tool_calls"], stopped_because=r["stopped_because"],
                intervened=r["intervened"], baseline_pred=base_pred,
                same_as_baseline=str(pred).strip() == str(base_pred).strip(),
            ))
            totals["n"] += 1
            totals["correct"] += score
            totals["same_as_baseline"] += written[-1]["same_as_baseline"]

        path = bdir / f"regen_{args.method}.jsonl"
        with open(path, "w") as f:
            for w in written:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        dt = time.time() - t0
        totals["t"] += dt
        acc = sum(w["score"][0] for w in written) / max(len(written), 1)
        same = sum(w["same_as_baseline"] for w in written) / max(len(written), 1)
        print(f"{bench:<16}{len(written):>5} traces  acc {acc:>5.1%}  "
              f"same-as-baseline {same:>5.0%}  {dt/max(len(written),1):>5.1f}s/trace")

    n = max(totals["n"], 1)
    summary = dict(model=args.model, method=args.method, label=label,
                   n=totals["n"], accuracy=totals["correct"] / n,
                   same_as_baseline=totals["same_as_baseline"] / n,
                   seconds_per_trace=totals["t"] / n, total_seconds=totals["t"])
    write_json(out / "regen_summary.json", summary)
    print(f"\n{totals['n']} traces  accuracy {summary['accuracy']:.1%}  "
          f"same-as-baseline {summary['same_as_baseline']:.0%}  "
          f"{summary['seconds_per_trace']:.1f}s/trace")
    if args.method == "none" and summary["same_as_baseline"] < 0.8:
        print("\nWARNING: the gate did not reproduce the baseline. Either the loop differs "
              "from math_eval.py, or TL and vLLM diverge enough that the baseline must be "
              "regenerated here too -- otherwise before/after confounds engine with "
              "intervention.")
    if not args.no_push:
        push_outputs(out, args.hf_repo, "regenerate")


if __name__ == "__main__":
    main()
