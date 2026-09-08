#!/usr/bin/env python3
"""Smoke test: what tool-call format does each TIR model actually emit?

    python scripts/smoke_test.py demyagent
    python scripts/smoke_test.py torl
    python scripts/smoke_test.py simpletir

One model per invocation, deliberately -- vLLM does not reliably release GPU
memory, so loading several in one process tends to OOM. Loop it from bash.

This gates the whole dataset pipeline. The exact delimiter strings decide where
generation stops to run code, where the tool answer is parsed from, where Layer 2
pairs inject their value, and where the intervention's post-output positions are.

Prompt format matters as much as the template. ToRL and SimpleTIR descend from
Qwen2.5 *base* models and take a raw completion prompt, which is what the ATTC
appendix shows. DemyAgent descends from Qwen3-4B-*Instruct*, so a raw prompt puts
it off-distribution -- in a first run it ignored tools entirely and just reasoned
in plain text. For it we try raw and chat-templated side by side.

Results are written to a clean file by this script. Do not pipe through tee:
vLLM logs to stderr with carriage returns, which overwrite and destroy the
generated text when the streams are merged.
"""

import os
import sys

# Must precede the vllm import.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

# Prompt templates verbatim from the ATTC appendix; decoding from its Table 4.
MODELS = {
    "torl": {
        "repo": "GAIR/ToRL-7B",
        # ATTC Figure 12
        "template": (
            "A conversation between User and Assistant. The user asks a question, and\n"
            "the Assistant solves it.\n"
            "User:Please integrate natural language reasoning with programs to solve the "
            "problem above, and put your final answer within \\boxed{{}}.\n"
            "{question}\n"
            "Assistant:"
        ),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_model_len": 4096,  # ToRL's real ceiling, inherited from Qwen2.5-Math
        "variants": ["raw"],  # base model -- raw completion
        "expect": "```python / ```output",
    },
    "simpletir": {
        "repo": "ZhenghaiXue/Qwen2.5-7B-SimpleTIR",
        # ATTC Figure 14
        "template": (
            "Solve the following problem step by step. You now have the ability to\n"
            "selectively write executable Python code to enhance your reasoning\n"
            "process. The Python code will be executed by an external sandbox,\n"
            "and the output (after \u201cCode execution result: \u201d) is returned to aid your\n"
            "reasoning and help you arrive at the final answer. The Python code\n"
            "should be complete scripts, including necessary imports.\n\n"
            "Code Format:\n"
            "Each code snippet is wrapped between \\n```python \\n```.\\n You need to use\n"
            "`print()` to output intermediate results.The code execution result is only\n"
            "auxiliary, you can choose to believe or not believe it.\n\n"
            "Answer Format:\n"
            "You should use \\boxed to return your final answer. The last part of your "
            "response should be: \\boxed{{'The final answer goes here.'}}\n\n"
            "User Question:\n"
            "{question}"
        ),
        "temperature": 1.0,
        "top_p": 0.7,
        "max_model_len": 8192,
        "variants": ["raw"],  # base model -- raw completion
        "expect": "```python / Code execution result:",
    },
    "demyagent": {
        "repo": "Gen-Verse/DemyAgent-4B",
        # ATTC Figure 16. The "[math/science domain]" placeholder is verbatim from
        # the paper -- it appears to be an unfilled template in their setup.
        "template": (
            "Analyze and solve the following [math/science domain] problem step by step.\n"
            "Problem: {question}\n"
            "Hint: The tool could be used for more precise and efficient calculations and "
            "could help you to verify your result before you reach the final answer.\n\n"
            "Note: You should first analyze the problem and form a high-level solution "
            "strategy, then utilize the tools to help you solve the problem.\n\n"
            "Answer Format: Do not put units of the final answer inside \\boxed{{}}. The "
            "content of \\boxed{{}} should be the numerical value of the final answer only, "
            "without any units. Remember once you make sure the current answer is your "
            "final answer, do not call the tools again and directly output the final answer "
            "in the following text format, the answer format must be: "
            "\\boxed{{'The final answer goes here.'}}."
        ),
        "temperature": 1.0,
        "top_p": 0.6,
        "max_model_len": 8192,  # config says 262144; capped so the KV cache fits
        "variants": ["raw", "chat"],  # Instruct-based -- chat is the likely one
        "expect": "UNKNOWN -- this is what we are here to find out",
    },
}

# ATTC's own Figure 1 example: the model reasons to 18, the answer is 17. Chosen
# because it is a known tool-conflict case for these exact models.
QUESTION = (
    "The smallest distance between the origin and a point on the graph of "
    "y = (1/2)x^2 - 9 can be expressed as a. Find a^2."
)

# Every convention seen across the ATTC appendix and Qwen chat templates.
CANDIDATES = [
    "```python", "```output", "```", "<code>", "</code>",
    "<interpreter>", "</interpreter>", "<tool_call>", "<tool_response>",
    "<python>", "</python>", "Code execution result",
    "<answer>", "<think>", "\\boxed{",
]


def build_prompts(cfg, tokenizer):
    """One (label, prompt_text) per variant we want to try."""
    raw = cfg["template"].format(question=QUESTION)
    out = []
    for v in cfg["variants"]:
        if v == "raw":
            out.append(("raw", raw))
        elif v == "chat":
            if not getattr(tokenizer, "chat_template", None):
                out.append(("chat", None))  # recorded as skipped
                continue
            out.append((
                "chat",
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            ))
    return out


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        print(f"usage: {sys.argv[0]} [{' | '.join(MODELS)}]")
        return 1

    key = sys.argv[1]
    cfg = MODELS[key]
    out_path = os.path.join(
        os.environ.get("TIR_OUTPUTS", "/workspace/outputs"), f"smoke_{key}.txt"
    )

    report = [
        "=" * 72,
        f"{key}  --  {cfg['repo']}",
        f"expected: {cfg['expect']}",
        f"variants: {cfg['variants']}",
        "=" * 72,
    ]

    tokenizer = AutoTokenizer.from_pretrained(cfg["repo"], trust_remote_code=True)
    prompts = build_prompts(cfg, tokenizer)

    llm = LLM(
        model=cfg["repo"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=2048,
        n=1 if cfg["temperature"] == 0.0 else 2,
    )

    for label, prompt in prompts:
        report += ["", "#" * 72, f"# VARIANT: {label}", "#" * 72]
        if prompt is None:
            report.append("SKIPPED -- tokenizer has no chat_template")
            continue

        report += ["", "--- PROMPT ---", prompt, ""]

        for i, out in enumerate(llm.generate([prompt], params)[0].outputs):
            report += [
                "",
                f"--- OUTPUT {i} ---",
                out.text,
                f"--- END (finish={out.finish_reason}, {len(out.token_ids)} tokens) ---",
                "",
                "--- DELIMITERS FOUND ---",
            ]
            found = [(c, out.text.count(c)) for c in CANDIDATES if c in out.text]
            if found:
                report += [f"  {c!r:<28} x{n}" for c, n in found]
            else:
                report.append("  NONE -- model emitted no recognised tool format")

    text = "\n".join(report)
    with open(out_path, "w") as f:
        f.write(text)

    print(text)
    print(f"\n\n>>> clean copy written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
