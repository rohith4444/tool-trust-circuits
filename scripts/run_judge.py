#!/usr/bin/env python3
"""Layer 1 labelling -- run ATTC's two Appendix B judges over our traces.

    # sample first: 20 cases per file, dump raw responses to eyeball
    python scripts/run_judge.py --limit 20 --verbose

    # full run
    python scripts/run_judge.py

Adapted from ATTC's tool_analysis/analyze_tool_ignore.py and
analyze_trust_or_not.py. The prompts and sampling parameters are theirs
verbatim; what we changed is the plumbing:

  - they hardcode one input file and load the model inside main(). We have 18
    trace files, and loading a 32B model per file would cost an hour of pure
    load time. This loads once and loops.
  - they append to the output file, so re-running duplicates. We overwrite.
  - they print "invalid type!" and drop responses with no parseable verdict.
    Silently dropping cases biases the rate, so we count and report them.

Both judges run on FALSE cases only -- Tool-Ignored is defined as a failure
mode, so a correct answer cannot exhibit it. That is ATTC's choice and it is
why their rates are "% of errors" rather than "% of problems".
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.models import MODELS  # noqa: E402
from scripts.summarize_runs import latest_runs  # noqa: E402

JUDGE_MODEL = "Qwen/Qwen3-32B"

# Prompts verbatim from ATTC, including their leading whitespace and typos.
JUDGES = {
    "tool_ignore": {
        "source": "tool_analysis/analyze_tool_ignore.py",
        "max_tokens": 1024,
        "prompt": """
    I will give you a jsonl formatted data, which is a case where the model answered incorrectly."question" is the question answered by the model, "answer" is the standard answer, "solution" is the standard problem-solving process, "code" is the content of the model's answer, and "pred" is the final answer of the model.
    You need to analyze the model's answer content based on this information. If there is no Python code segment(after "```python") in the model's answer, output "\\boxed{2}". If the return result of the Python code (after "```output") is correct (including correct answer and correct thinking), output "\\boxed{1}". Otherwise, output "\\boxed{0}". Your answer is limited to these three types and must be included in "\\boxed{}, don't output extra text". The cases that need to be analyzed are as follows:\n
    """,
        # Names from their own statistics dict. type 1 is Tool-Ignored: the model
        # was wrong while the tool was right.
        "labels": {"0": "normal", "1": "TOOL-IGNORED", "2": "no tools used"},
        "headline": "1",
    },
    "trust_or_not": {
        "source": "tool_analysis/analyze_trust_or_not.py",
        "max_tokens": 2048,
        "prompt": """
     I will give you a jsonl formatted data, which is a case where the model answered."question" is the question answered by the model, "answer" is the standard answer, "solution" is the standard problem-solving process, "code" is the content of the model's answer, and "pred" is the final answer of the model.
     You need to analyze the model's answer content based on this information.If there is a contradiction between the inference of the model and the execution result of Python code (after "```output") ,the model chooses to believe in the inference,output "\\boxed{0}".If there is a contradiction between the inference of the model and the execution result of Python code(after "```output"), the model chooses to believe in the code execution result,output "\\boxed{1}".If neither of these situations exist,output "\\boxed{2}".Your answer is limited to these three types and must be included in "\\boxed{}", don't output extra text.The cases that need to be analyzed are as follows:\n
    """,
        "labels": {"0": "trusted own reasoning", "1": "trusted the tool", "2": "no conflict"},
        "headline": "0",
    },
}


def extract_boxed(text):
    """Last \\boxed{...} in the response, as ATTC does."""
    m = re.findall(r"\\boxed\{(.*?)\}", text)
    return m[-1] if m else None


def false_cases(path, limit):
    """The record shape ATTC's judges expect, for wrong answers only."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for i, s in enumerate(d.get("score", [])):
            if s is False:
                out.append({
                    "idx": d.get("idx"),
                    "question": d.get("question", ""),
                    "solution": d.get("gt_cot", ""),
                    "answer": d.get("gt", ""),
                    "code": d.get("code", [""])[i] if i < len(d.get("code", [])) else "",
                    "pred": d.get("pred", [""])[i] if i < len(d.get("pred", [])) else "",
                })
        if limit and len(out) >= limit:
            break
    return out[:limit] if limit else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0,
                    help="cases per file; 0 = all. Use a small value to sample "
                         "before committing to the full run.")
    ap.add_argument("--verbose", action="store_true",
                    help="print raw judge responses -- the only way to check the "
                         "verdicts are reasoned rather than pattern-matched")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="override the judge's limit. ATTC passes raw prompts, "
                         "so Qwen3 reasons at length before answering; at their "
                         "1024/2048 caps 20-45%% of verdicts were truncated away.")
    ap.add_argument("--judges", default="tool_ignore",
                    help="comma-separated. Default is tool_ignore alone -- it "
                         "produces the Tool-Ignored rate, the only judged number "
                         "either contribution needs. trust_or_not reproduces "
                         "ATTC's Figure 2, which we cite rather than re-derive.")
    ap.add_argument("--no-thinking", action="store_true",
                    help="wrap the prompt in Qwen3's chat template with "
                         "enable_thinking=False, so it emits a verdict instead of "
                         "reasoning to the token limit. Deviates from ATTC, who "
                         "pass raw text -- validate agreement before relying on it.")
    args = ap.parse_args()

    judges = {k: JUDGES[k] for k in
              (j.strip() for j in args.judges.split(",")) if k in JUDGES}
    if not judges:
        sys.exit(f"unknown judge(s): {args.judges}. known: {list(JUDGES)}")

    from vllm import LLM, SamplingParams  # noqa: E402

    root = Path(args.out)
    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    runs = [r for r in latest_runs(root) if r[2] == "-1"
            and r[0].rsplit("_seed", 1)[0] in keys]
    if not runs:
        sys.exit(f"no protocol runs (limit -1) under {root}")

    mode = "chat template, thinking OFF" if args.no_thinking else "raw prompt (ATTC)"
    print(f"{len(runs)} trace files, judges: {list(judges)}, mode: {mode}")
    print(f"loading {JUDGE_MODEL} (once, reused for every file)...\n")

    tokenizer = None
    if args.no_thinking:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)

    def build(prompt_text):
        if tokenizer is None:
            return prompt_text
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    llm = LLM(model=JUDGE_MODEL, tensor_parallel_size=len(gpus),
              trust_remote_code=True, max_model_len=32768)

    label_dir = root / "labels"
    label_dir.mkdir(exist_ok=True)
    totals = {}

    for model_dir, bench, _, path in runs:
        key = model_dir.rsplit("_seed", 1)[0]
        cases = false_cases(path, args.limit)
        if not cases:
            print(f"{key}/{bench}: no false cases, skipping")
            continue

        for jname, jcfg in judges.items():
            prompts = [build(jcfg["prompt"] + json.dumps(c, ensure_ascii=False, indent=2))
                       for c in cases]
            outs = llm.generate(prompts, SamplingParams(
                temperature=0.6, top_p=0.95, n=1,
                max_tokens=args.max_tokens or jcfg["max_tokens"],
            ))
            outs = sorted(outs, key=lambda x: int(x.request_id))

            counts, invalid, thinking, lengths = Counter(), 0, 0, []
            records = []
            for c, o in zip(cases, outs):
                text = o.outputs[0].text
                verdict = extract_boxed(text)
                lengths.append(len(o.outputs[0].token_ids))
                thinking += "<think>" in text
                if verdict in jcfg["labels"]:
                    counts[verdict] += 1
                else:
                    invalid += 1
                records.append({"idx": c["idx"], "type": verdict, "response": text})

            tag = "_nothink" if args.no_thinking else ""
            out_path = label_dir / f"{key}_{bench}_{jname}{tag}.jsonl"
            with open(out_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            n = sum(counts.values())
            head = jcfg["headline"]
            rate = counts[head] / n if n else 0
            totals[(key, bench, jname)] = (rate, n, invalid)

            dist = "  ".join(f"{jcfg['labels'][k]}={counts[k]}" for k in sorted(jcfg["labels"]))
            print(f"{key}/{bench}/{jname}: {dist}")
            print(f"    {jcfg['labels'][head]} rate = {100*rate:.1f}%   "
                  f"invalid={invalid}/{len(cases)}   "
                  f"tokens median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}"
                  f"{'   <think> in ' + str(thinking) if thinking else ''}")

            if args.verbose:
                for c, r in list(zip(cases, records))[:2]:
                    print(f"\n    --- idx {c['idx']}  gt={c['answer']!r}  pred={c['pred']!r} ---")
                    print("    " + (r["response"] or "")[:700].replace("\n", "\n    "))
                print()

    print("\n" + "=" * 72)
    print("TOOL-IGNORED RATE vs ATTC's released verdicts")
    print("=" * 72)
    for (key, bench, jname), (rate, n, invalid) in sorted(totals.items()):
        if jname != "tool_ignore":
            continue
        target = (MODELS.get(key, {}).get("attc_tool_ignored") or {}).get(bench)
        t = f"{100*target:.0f}%" if target is not None else "-"
        flag = ""
        if target is not None:
            flag = " OK" if abs(rate - target) < 0.05 else "  <-- off"
        print(f"  {key:<11}{bench:<15}ours={100*rate:>5.1f}%  ATTC={t:>5}{flag}")

    bad = [k for k, v in totals.items() if v[2] > 0.05 * max(v[1], 1)]
    if bad:
        print(f"\nWARNING: >5% unparseable verdicts in {len(bad)} run(s). "
              f"Raise --max-tokens; <think> is likely eating the budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
