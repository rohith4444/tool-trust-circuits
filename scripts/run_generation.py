#!/usr/bin/env python3
"""Drive ATTC's math_eval.py from our model registry.

    # port check -- must reproduce the 20-problem baseline exactly
    python scripts/run_generation.py --models torl --benchmarks math500 --limit 20

    # config validation across all three
    python scripts/run_generation.py --limit 50

    # full run, one model per GPU
    python scripts/run_generation.py --models torl      --gpu 0
    python scripts/run_generation.py --models simpletir --gpu 1

Every decoding setting comes from configs/models.py, which mirrors ATTC Table 4.
Nothing is hardcoded here -- deviating from their settings would invalidate the
reproduction gate, so the only knobs exposed are which models, which benchmarks,
how many problems, and which GPU.
"""

import argparse
import os
import resource
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from configs.models import (  # noqa: E402
    BENCHMARKS, MAX_FUNC_CALL, MAX_TOKENS_PER_CALL, MODELS,
)


def raise_fd_limit():
    """PythonExecutor pools one process per CPU, and evaluate() opens a second
    pool. On a 32-vCPU host that blows past the default 1024 descriptors and
    dies with 'Too many open files' -- after generation has finished but before
    grading, so you lose the whole run.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = min(65536, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def run(key, benchmarks, limit, seed, gpu, out_root, dry_run):
    cfg = MODELS[key]
    out_dir = Path(out_root) / f"{key}_seed{seed}"

    cmd = [
        sys.executable, "-u", str(REPO / "generation" / "math_eval.py"),
        "--model_name_or_path", cfg["repo"],
        "--data_dir", str(REPO / "data"),
        "--output_dir", str(out_dir),
        "--data_names", ",".join(benchmarks),
        "--split", "test",
        "--prompt_type", cfg["prompt_type"],
        "--num_test_sample", str(limit),
        "--seed", str(seed),
        "--temperature", str(cfg["temperature"]),
        "--n_sampling", str(cfg["n_sampling"]),
        "--top_p", str(cfg["top_p"]),
        "--start", "0", "--end", "-1",
        "--save_outputs",
        "--max_tokens_per_call", str(MAX_TOKENS_PER_CALL),
        "--use_vllm",
        "--max_func_call", str(MAX_FUNC_CALL),
        "--overwrite",
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    # Dummy, matching their run scripts. Nothing calls OpenAI; this only existed
    # so a client object could be constructed at import time, and we removed that.
    env.setdefault("OPENAI_API_KEY", "0")

    print(f"\n{'=' * 72}")
    print(f"{key}  seed={seed}  gpu={gpu}")
    print(f"  {cfg['repo']}  prompt_type={cfg['prompt_type']}")
    print(f"  temp={cfg['temperature']} top_p={cfg['top_p']} n={cfg['n_sampling']}")
    print(f"  benchmarks: {','.join(benchmarks)}   limit={limit}")
    print(f"  -> {out_dir}")
    print(f"{'=' * 72}\n")
    print(" ".join(cmd), "\n")

    if dry_run:
        return 0
    # cwd matters: their modules use flat imports (from utils import ...)
    return subprocess.call(cmd, env=env, cwd=str(REPO / "generation"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS),
                    help=f"comma-separated. default: all ({','.join(MODELS)})")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--limit", type=int, default=-1,
                    help="problems per benchmark; -1 = all")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=os.environ.get("TIR_OUTPUTS", "/workspace/outputs"))
    ap.add_argument("--for-rates", action="store_true",
                    help="also run the extra seeds needed for stable reported "
                         "rates on temperature>0 models (intervention phase only)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"file descriptor limit: {raise_fd_limit()}")

    keys = [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}. known: {list(MODELS)}")

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    failures = []
    for key in keys:
        cfg = MODELS[key]
        seeds = cfg.get("seeds_for_rates", cfg["seeds"]) if args.for_rates else cfg["seeds"]
        for seed in seeds:
            rc = run(key, benchmarks, args.limit, seed, args.gpu, args.out, args.dry_run)
            if rc != 0:
                failures.append(f"{key}/seed{seed} (exit {rc})")

    if failures:
        print("\nFAILED: " + ", ".join(failures))
        return 1
    print("\nall runs completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
