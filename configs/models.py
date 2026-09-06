"""Shared model registry.

The README names `configs/models.py` as the file shared by everything downstream.
Keys are the short names used by scripts/smoke_test.py on the dataset branch, so
dataset files and `--model` flags agree. Repo ids come from scripts/verify_models.py.
"""

MODELS = {
    "torl": {
        "repo": "GAIR/ToRL-7B",
        "family": "qwen2.5-math",
        "role": "PRIMARY - circuit + full intervention",
    },
    "simpletir": {
        "repo": "ZhenghaiXue/Qwen2.5-7B-SimpleTIR",
        "family": "qwen2.5",
        "role": "universality (within-family) + 2nd intervention",
    },
    "demyagent": {
        "repo": "Gen-Verse/DemyAgent-4B",
        "family": "qwen3",
        "role": "universality - cross-family (Qwen3)",
    },
    "base": {
        "repo": "Qwen/Qwen2.5-Math-7B",
        "family": "qwen2.5-math",
        "role": "attribution control - ToRL's direct parent",
    },
}

# TransformerLens allocates an [n_ctx, n_ctx] mask; 131072 (Qwen2.5-7B) would be 17 GB.
MAX_CTX = 8192

# Tool-output delimiters as the ATTC harness writes them for every markdown-fence
# prompt type (torl, simpleTIR, DemyAgent): "\n```output\n<result>\n```\n".
# The dataset side's smoke test decides what each model emits natively; if a model
# differs, only these two strings (or the dataset's prefix/suffix fields) change.
OUTPUT_OPEN = "\n```output\n"
OUTPUT_CLOSE = "\n```\n"
