"""Model registry -- the single source of truth for every stage.

Generation, pair building, circuit localisation, and the intervention all read
from here, so extending the roster (ICLR) is a config edit rather than new code.

Decoding settings come from ATTC Table 4; the accuracy figures from its Table 1;
the Tool-Ignored rates are computed from the judge verdicts they released in
`case_analysis/output_*.jsonl` -- those are exact, unlike the bar charts in
Figures 3 and 8.

Lineage was confirmed empirically rather than from the papers: ToRL's
`max_position_embeddings` is 4096, matching Qwen2.5-Math-7B, where plain
Qwen2.5-7B is 131072. That is why the attribution control is Math-7B.
"""

# All three TIR models share one delimiter convention. Confirmed from
# math_eval.py, not from the appendix -- ATTC Figure 14 shows
# "Code execution result:" in SimpleTIR's *prompt text*, but the harness puts
# every one of these prompt types in the same stop_words branch:
#
#   stop_words += ["\n\n---", "```output"]        (math_eval.py:183)
#   exec_result = f"\n```output\n{result}\n```\n" (math_eval.py:328)
#
# The model never writes a tool result: generation halts at the opening
# delimiter, the harness executes the code and appends the real output.
DELIMITERS = {
    "code_open": "```python",
    "code_close": "```",
    "output_open": "```output",
    "output_close": "```",
}

# ATTC's run scripts: --max_tokens_per_call 3072 --max_func_call 4.
# Kept verbatim; deviating would break the reproduction gate.
MAX_TOKENS_PER_CALL = 3072
MAX_FUNC_CALL = 4

# Context cap for the circuit stages (circuits/models/load.py). ToRL's pairs fit in
# 4096 tokens, and one shared pair file is used for every model (docs/CIRCUIT.md),
# so no analysis forward ever needs more, whatever a model's max_model_len says.
MAX_CTX = 4096

# Union of ATTC's two benchmark sets. Table 1 (accuracy) omits gsm8k; Figures 3
# and 8 (Tool-Ignored) omit aime24 and amc23. We need both metrics, so we run
# all six. 2,836 problems.
BENCHMARKS = ["gsm8k", "math500", "minerva_math", "olympiadbench", "aime24", "amc23"]

BENCHMARK_SIZES = {
    "gsm8k": 1319,
    "math500": 500,
    "minerva_math": 272,
    "olympiadbench": 675,
    "aime24": 30,
    "amc23": 40,
}

MODELS = {
    "torl": {
        "repo": "GAIR/ToRL-7B",
        "prompt_type": "torl",
        "base": "Qwen/Qwen2.5-Math-7B",
        # Table 4. Temperature 0 makes this deterministic, so one pass is exact
        # and the reproduction gate is a hard number rather than a distribution.
        "temperature": 0.0,
        "top_p": 1.0,
        "n_sampling": 1,
        "seeds": [0],
        # Real ceiling, inherited from Qwen2.5-Math. Note MAX_TOKENS_PER_CALL is
        # 3072 and the prompt grows with each tool round, so long traces will hit
        # this. Count truncations rather than assuming they do not happen.
        "max_model_len": 4096,
        "role": "primary -- circuit, full validation, intervention",
        "attc_acc": {"math500": 82.2, "minerva_math": 33.5, "olympiadbench": 49.9,
                     "aime24": 43.3, "amc23": 65.0, "avg": 54.8},
        "attc_acc_with_method": 60.0,
        # From their released judge verdicts, type 1 = "Do not believe in tool output"
        "attc_tool_ignored": {"gsm8k": 0.12, "math500": 0.11, "olympiadbench": 0.10},
    },
    "simpletir": {
        "repo": "ZhenghaiXue/Qwen2.5-7B-SimpleTIR",
        "prompt_type": "simpleTIR",
        "base": "Qwen/Qwen2.5-7B",
        "temperature": 1.0,
        "top_p": 0.7,
        "n_sampling": 1,
        # One seed is enough for pairs (fixed text, nothing to average). The
        # other two are only needed when we report rates, in the intervention
        # phase -- generating them now would delay DemyAgent for no gain.
        "seeds": [0],
        "seeds_for_rates": [0, 1, 2],
        # config declares 131072; capping restores ~8x batching throughput
        "max_model_len": 16384,
        "role": "universality (within-family) + second intervention model",
        "attc_acc": {"math500": 82.1, "minerva_math": 30.1, "olympiadbench": 47.4,
                     "aime24": 46.7, "amc23": 75.0, "avg": 56.3},
        "attc_acc_with_method": 61.3,
        "attc_tool_ignored": {"gsm8k": 0.17, "math500": 0.20, "olympiadbench": 0.10},
    },
    "demyagent": {
        "repo": "Gen-Verse/DemyAgent-4B",
        "prompt_type": "DemyAgent",
        # Lineage has an SFT step: Qwen3-4B-Instruct-2507 -> Qwen3-4B-RA-SFT -> DemyAgent
        "base": "Gen-Verse/Qwen3-4B-RA-SFT",
        "temperature": 1.0,
        "top_p": 0.6,
        "n_sampling": 1,
        "seeds": [0],
        # config.json declares 262144. vLLM sizes its KV cache to whatever is
        # declared, which collapsed batching to 1.79 concurrent requests against
        # ToRL's 262. 16384 leaves headroom for 3072 tokens x 4 rounds plus the
        # prompt (~13.5k worst case) while restoring ~16x throughput.
        "max_model_len": 16384,
        "role": "universality (cross-family) -- the load-bearing evidence",
        "attc_acc": {"math500": 71.4, "minerva_math": 17.3, "olympiadbench": 51.9,
                     "aime24": 40.0, "amc23": 75.0, "avg": 51.1},
        "attc_acc_with_method": 55.2,
        # Not in ATTC Figures 3/8 -- no published Tool-Ignored number exists for
        # this model, so it serves universality only, never a rate comparison.
        "attc_tool_ignored": None,
    },
}

# Attribution control. Not a universality model -- it answers a question no
# comparison among TIR checkpoints can: did TIR training create the circuit, or
# inherit it from pretraining? Needs zero generation: Layer 2 pairs are fixed
# text, so we feed it ToRL's pairs and read Delta from forward passes.
CONTROL = {
    "repo": "Qwen/Qwen2.5-Math-7B",
    "borrows_pairs_from": "torl",
    "max_model_len": 4096,
    "role": "attribution control -- ToRL's direct parent",
}


def generation_runs(model_keys=None, benchmarks=None, for_rates=False):
    """Every (model, seed) generation run to perform.

    for_rates=True adds the extra seeds needed for stable reported rates on
    temperature>0 models. Leave it False for pair building.
    """
    out = []
    for key in model_keys or MODELS:
        cfg = MODELS[key]
        seeds = cfg.get("seeds_for_rates", cfg["seeds"]) if for_rates else cfg["seeds"]
        for seed in seeds:
            out.append({
                "key": key,
                "seed": seed,
                "benchmarks": benchmarks or BENCHMARKS,
                **cfg,
            })
    return out
