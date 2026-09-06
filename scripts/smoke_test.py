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

For ToRL and SimpleTIR we have templates from the ATTC appendix (Figures 12, 14)
and this confirms them -- those figures show what ATTC *told* the model, not
necessarily what the model *emits*, and papers get such details wrong. DemyAgent
is the real unknown: its output delimiter is undocumented in the ATTC appendix,
its model card, its repo README, and its own paper.

The trick for DemyAgent: run it with no tool attached and let it generate past
the code block. A model trained on a fixed format usually hallucinates that
format when no execution result arrives -- which tells us what it expects.
"""

import sys

from vllm import LLM, SamplingParams

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
    "<answer>", "\\boxed{",
]


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        print(f"usage: {sys.argv[0]} [{' | '.join(MODELS)}]")
        return 1

    key = sys.argv[1]
    cfg = MODELS[key]

    print(f"\n{'=' * 72}\n{key}  --  {cfg['repo']}")
    print(f"expected: {cfg['expect']}\n{'=' * 72}\n")

    llm = LLM(
        model=cfg["repo"],
        max_model_len=cfg["max_model_len"],
        gpu_memory_utilization=0.85,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    prompt = cfg["template"].format(question=QUESTION)
    print(f"--- PROMPT ---\n{prompt}\n")

    # No stop strings on purpose: we want everything, including whatever the model
    # hallucinates where a tool result would go. That hallucination is the answer.
    params = SamplingParams(
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        max_tokens=1024,
        n=1 if cfg["temperature"] == 0.0 else 2,
    )

    for i, out in enumerate(llm.generate([prompt], params)[0].outputs):
        print(f"\n--- RAW OUTPUT {i} ---")
        print(out.text)
        print(f"--- END (finish={out.finish_reason}, {len(out.token_ids)} tokens) ---")

        print("\n--- DELIMITERS FOUND ---")
        found = [(c, out.text.count(c)) for c in CANDIDATES if c in out.text]
        for c, n in found:
            print(f"  {c!r:<28} x{n}")
        if not found:
            print("  NONE -- model emitted no recognised tool format")

    return 0


if __name__ == "__main__":
    sys.exit(main())
