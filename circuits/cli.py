"""Shared argparse / IO helpers for scripts/run_*.py."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .data.pairs import build_pairs, load_examples


RESULTS_REPO = "tool-trust-circuits/tool-trust-results"


def add_common(p: argparse.ArgumentParser, model_arg: bool = True) -> None:
    if model_arg:
        p.add_argument("--model", required=True, help="registry name (torl, simpletir, demyagent, base) or HF repo id")
    p.add_argument("--pairs", required=True, help="jsonl pair file: a docs/DATASET.md file (pairs/<model>_<split>.jsonl) or the segment format in circuits/data/pairs.py::PAIR_FILE_SCHEMA")
    p.add_argument("--label", default=None, help="keep only pairs with this label (accept|override; dataset files map tool_case -> label)")
    p.add_argument("--max-tokens", type=int, default=None, help="left-truncate pairs longer than this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, help="torch dtype name (default bfloat16)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--hf-repo", default=RESULTS_REPO, help="HF dataset repo the finished --out dir is pushed to")
    p.add_argument("--no-push", action="store_true", help="do not push --out to --hf-repo when done")
    p.add_argument("-v", "--verbose", action="store_true")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("transformer_lens").setLevel(logging.WARNING)


def get_pairs(model, args, examples: list[dict] | None = None, max_tokens: int | None = "default"):
    examples = load_examples(args.pairs, args.label) if examples is None else examples
    mt = args.max_tokens if max_tokens == "default" else max_tokens
    pairs = build_pairs(model.tokenizer, examples, max_tokens=mt, model_name=getattr(model, "tt_name", None))
    if not pairs:
        raise SystemExit("no usable pairs")
    return examples, pairs


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def push_outputs(out: str | Path, repo: str, stage: str) -> str | None:
    """Upload the finished --out directory to the private results dataset; returns the path in the repo.

    `outputs/circuits/torl` lands at `torl/`, `outputs/x` at `x/`, anything else at its own path.
    Skips with a warning when HF_TOKEN is unset so a run without a token still finishes.
    """
    if not os.environ.get("HF_TOKEN"):
        print(f"WARNING: HF_TOKEN not set -- {out} was not pushed to {repo}")
        return None
    from huggingface_hub import HfApi

    out = Path(out)
    rel = out.as_posix().removeprefix("outputs/circuits/").removeprefix("outputs/").strip("/")
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out), path_in_repo=rel, repo_id=repo, repo_type="dataset",
                      commit_message=f"{stage}: {rel}")
    print(f"pushed {out} -> https://huggingface.co/datasets/{repo}/tree/main/{rel}")
    return rel
