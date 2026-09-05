#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface-hub>=0.25",
#     "transformers>=4.51",  # Qwen3 support (DemyAgent-4B); bump if configs fail
# ]
# ///
"""Step 0: verify the models before provisioning any GPU.

Reads config + tokenizer metadata from the HuggingFace API. Downloads no weights.

    uv run scripts/verify_models.py

What it answers, and why each matters:
  - Does the repo resolve, and is it gated?   -> whether we need an HF token
  - Total weight size                         -> how big the network volume must be
  - Architecture / layers / heads             -> TransformerLens support, CKA depth mapping
  - Tokenizer identity                        -> can the base model reuse ToRL's pairs?
  - How numbers tokenize                      -> where we read the Delta metric
  - Chat template tool markers                -> the injection point delimiters
"""

import sys
import traceback

from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer

# The final roster -- these are the four we download. ~54 GB total.
#
# Lineage, confirmed by the max_position_embeddings signature (4096 is
# Qwen2.5-Math's, vs 131072 for plain Qwen2.5-7B):
#     Qwen2.5-Math-7B --RL--> ToRL-7B
#     Qwen2.5-7B      --RL--> SimpleTIR-7B
#     Qwen3-4B-Instruct-2507 --SFT--> Qwen3-4B-RA-SFT --RL--> DemyAgent-4B
MODELS = [
    ("GAIR/ToRL-7B", "PRIMARY - circuit + full intervention"),
    ("ZhenghaiXue/Qwen2.5-7B-SimpleTIR", "universality (within-family) + 2nd intervention"),
    ("Gen-Verse/DemyAgent-4B", "universality - cross-family (Qwen3)"),
    ("Qwen/Qwen2.5-Math-7B", "attribution control - ToRL's direct parent"),
]

# Standby if DemyAgent's output delimiter can't be pinned down. No official
# ReTool-4B checkpoint has surfaced yet -- the search below is to confirm that.
SEARCH_TERMS = ["ReTool-4B"]

# Tool-call delimiters we know from the ATTC appendix, plus generic chat-template ones.
TOOL_MARKERS = [
    "```python", "```output", "<code>", "<interpreter>",
    "<tool_call>", "<tool_response>", "<python>", "Code execution result",
]

# The numbers our Delta metric will compare. Pairs are (own answer, tool answer).
NUMBER_PAIRS = [("50", "42"), ("18", "17"), ("42", "99"), ("7", "9"), ("123", "223")]

# Encoded by every model to test whether two models share a tokenizer.
TOKENIZER_PROBE = "The answer is \\boxed{42}. ```output\n17.0\n```"

api = HfApi()


def first_divergence(tokenizer, a: str, b: str):
    """Where two answer strings first differ in token space.

    This is the position we read Delta at. Returns (index, tokens_a, tokens_b),
    or (None, ...) if one string is a token-prefix of the other.
    """
    ta = tokenizer.encode(a, add_special_tokens=False)
    tb = tokenizer.encode(b, add_special_tokens=False)
    for i in range(min(len(ta), len(tb))):
        if ta[i] != tb[i]:
            return i, ta, tb
    return None, ta, tb


def weight_size_gb(info) -> float:
    total = 0
    for f in info.siblings or []:
        if f.rfilename.endswith((".safetensors", ".bin")) and f.size:
            total += f.size
    return total / 1e9


def check_model(repo_id: str, role: str) -> dict | None:
    print(f"\n{'=' * 72}\n{repo_id}\n  role: {role}\n{'=' * 72}")

    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as e:
        print(f"  REPO NOT ACCESSIBLE: {type(e).__name__}: {e}")
        return None

    gated = info.gated or False
    print(f"  exists      : yes")
    print(f"  gated       : {gated}  {'<-- needs HF token + accepted terms' if gated else ''}")
    print(f"  license     : {(info.card_data or {}).get('license', 'not declared')}")
    print(f"  downloads   : {info.downloads}")
    size = weight_size_gb(info)
    print(f"  weight size : {size:.1f} GB" if size else "  weight size : unknown")

    try:
        cfg = AutoConfig.from_pretrained(repo_id)
        print(f"  arch        : {cfg.architectures}")
        print(f"  model_type  : {cfg.model_type}")
        print(f"  layers      : {getattr(cfg, 'num_hidden_layers', '?')}")
        print(f"  attn heads  : {getattr(cfg, 'num_attention_heads', '?')}")
        print(f"  kv heads    : {getattr(cfg, 'num_key_value_heads', '?')}")
        print(f"  hidden size : {getattr(cfg, 'hidden_size', '?')}")
        print(f"  vocab size  : {getattr(cfg, 'vocab_size', '?')}")
        print(f"  max pos     : {getattr(cfg, 'max_position_embeddings', '?')}")
    except Exception as e:
        print(f"  CONFIG FAILED: {type(e).__name__}: {e}")
        cfg = None

    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=False)
    except Exception as e:
        print(f"  TOKENIZER FAILED: {type(e).__name__}: {e}")
        return {"repo_id": repo_id, "config": cfg, "probe": None}

    # Chat template -- often encodes the tool-call format we need for injection.
    template = getattr(tok, "chat_template", None)
    if template:
        found = [m for m in TOOL_MARKERS if m in template]
        print(f"  chat template: {len(template)} chars")
        print(f"  tool markers : {found if found else 'NONE FOUND'}")
    else:
        print(f"  chat template: none")

    # Number tokenization -- decides where Delta is read.
    print("  number pairs (Delta read position):")
    for a, b in NUMBER_PAIRS:
        idx, ta, tb = first_divergence(tok, a, b)
        pa = [tok.decode([t]) for t in ta]
        pb = [tok.decode([t]) for t in tb]
        where = f"index {idx}" if idx is not None else "PREFIX CLASH - unusable"
        print(f"    {a:>4} {pa} vs {b:>4} {pb}  ->  {where}")

    probe = tok.encode(TOKENIZER_PROBE, add_special_tokens=False)
    return {"repo_id": repo_id, "config": cfg, "probe": tuple(probe)}


def search_unresolved():
    print(f"\n\n{'#' * 72}\n# UNRESOLVED REPO IDS - candidates from HF search\n{'#' * 72}")
    for term in SEARCH_TERMS:
        print(f"\n  search: {term!r}")
        try:
            hits = list(api.list_models(search=term, limit=15, sort="downloads"))
        except Exception as e:
            print(f"    SEARCH FAILED: {e}")
            continue
        if not hits:
            print("    no results")
            continue
        for m in hits:
            print(f"    {m.id:<55} downloads={m.downloads}")


def compare_tokenizers(results):
    print(f"\n\n{'#' * 72}\n# TOKENIZER IDENTITY\n{'#' * 72}")
    print("\n  Matters because the base-model control reuses ToRL's pairs only if")
    print("  the tokenizers are identical.\n")
    groups: dict[tuple, list[str]] = {}
    for r in results:
        if r and r.get("probe"):
            groups.setdefault(r["probe"], []).append(r["repo_id"])
    for i, (_, members) in enumerate(groups.items(), 1):
        print(f"  group {i}:")
        for m in members:
            print(f"    - {m}")


def main():
    results = []
    for repo_id, role in MODELS:
        try:
            results.append(check_model(repo_id, role))
        except Exception:
            print(f"  UNEXPECTED ERROR on {repo_id}:")
            traceback.print_exc()
            results.append(None)

    search_unresolved()
    compare_tokenizers(results)

    ok = sum(1 for r in results if r)
    print(f"\n\n{'#' * 72}")
    print(f"# {ok}/{len(MODELS)} models resolved")
    print(f"{'#' * 72}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
