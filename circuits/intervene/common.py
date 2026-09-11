"""Shared by both intervention methods: which run to read, and deduplicating it."""
from __future__ import annotations

import hashlib
import random

import torch


def run_ids(pair, run: str) -> torch.Tensor:
    """The token ids for one side of a pair.

    Interventions run on `clean` -- the tool reports the truth and the model writes its
    own wrong answer anyway, which is the failure we are trying to fix and, for
    `tool_right` pairs, the trace as it actually happened. `corrupt` is the manufactured
    condition localization needed for its contrast; fixing the model's response to a
    fabricated value is not the goal.
    """
    return pair.clean_ids if run == "clean" else pair.corrupt_ids


def dedupe(pairs, run: str = "clean") -> list:
    """Pairs with distinct token ids for this run.

    On clean runs this halves the work: both Z-variants of a trace carry the same clean
    text, since Z appears only in the corrupt one. A ToRL discovery file of 712 pairs is
    356 distinct clean runs. Without this every trace is counted twice -- harmless to a
    mean, but it doubles every reported n, and an average over "34 held-out traces" that
    is really 17 is the kind of number that reaches a paper.

    On corrupt runs it is automatically a no-op, so no caller has to remember it.
    """
    seen, out = set(), []
    for pair in pairs:
        key = hashlib.sha1(run_ids(pair, run).cpu().numpy().tobytes()).hexdigest()
        if key not in seen:
            seen.add(key)
            out.append(pair)
    return out


def subsample(pairs, n: int | None, seed: int = 0) -> list:
    """A fixed random subset, for sweeping only.

    Choosing a layer or an (alpha, beta) is hyperparameter selection: a 50-point grid
    over 356 runs is ~18k forwards, and the ranking of configs is stable long before the
    mean is. The held-out report always uses every pair -- only the search is subsampled.
    Seeded, so a rerun sweeps the same subset and the choice is reproducible.
    """
    if not n or n >= len(pairs):
        return pairs
    rng = random.Random(seed)
    return rng.sample(pairs, n)
