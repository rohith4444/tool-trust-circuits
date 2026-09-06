"""Shared argparse / IO helpers for scripts/run_*.py."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .data.pairs import build_pairs, load_examples


def add_common(p: argparse.ArgumentParser, model_arg: bool = True) -> None:
    if model_arg:
        p.add_argument("--model", required=True, help="registry name (torl, simpletir, demyagent, base) or HF repo id")
    p.add_argument("--pairs", required=True, help="jsonl pair file (schema: tooltrust/data/pairs.py::PAIR_FILE_SCHEMA)")
    p.add_argument("--label", default=None, help="keep only pairs with this label (accept|override)")
    p.add_argument("--max-tokens", type=int, default=None, help="left-truncate pairs longer than this")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None, help="torch dtype name (default bfloat16)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("-v", "--verbose", action="store_true")


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("transformer_lens").setLevel(logging.WARNING)


def get_pairs(model, args, examples: list[dict] | None = None, max_tokens: int | None = "default"):
    examples = load_examples(args.pairs, args.label) if examples is None else examples
    mt = args.max_tokens if max_tokens == "default" else max_tokens
    pairs = build_pairs(model.tokenizer, examples, max_tokens=mt)
    if not pairs:
        raise SystemExit("no usable pairs")
    return examples, pairs


def write_json(path: str | Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
