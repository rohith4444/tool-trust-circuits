#!/usr/bin/env python3
"""The paper's two figures, from artifacts already on disk. No GPU, no model.

    python scripts/make_figures.py --results /workspace/results \\
        --runs /workspace/repo/outputs/circuits --out figures/

Writes circuit_map.pdf and kcurve.pdf (plus .png for quick viewing).

FIGURE 1 -- circuit map
    x = layer, y = measured recovery at post-output positions, one marker per confirmed
    head, ToRL and SimpleTIR overlaid. Chosen over a layer x head-index grid because index
    position carries no meaning while recovery does, and overlaying the two same-architecture
    models puts the universality result in the same panel as the circuit: the heads that
    matter are the same heads, at the same depth, with the same effect size.

FIGURE 2 -- faithfulness vs circuit size
    Backs the claim that sub-unity faithfulness is not an artifact of choosing too few
    heads. The dashed line is the random-circuit null, which is the only thing that makes
    the curve's height interpretable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = ("torl", "simpletir", "demyagent")
LABEL = {"torl": "ToRL-7B", "simpletir": "SimpleTIR-7B", "demyagent": "DemyAgent-4B"}
COLOR = {"torl": "#1f77b4", "simpletir": "#d62728", "demyagent": "#2ca02c"}
MARKER = {"torl": "o", "simpletir": "^", "demyagent": "s"}


def confirmed_heads(results: Path, model: str) -> list[dict]:
    c = json.load(open(results / model / "circuit.json"))
    return [n for n in c["nodes"] if n["above_noise"] and n["type"] == "a" and n["layer"] != 0]


def circuit_map(results: Path, out: Path, models=("torl", "simpletir")) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    shared = None
    per = {}
    for m in models:
        heads = confirmed_heads(results, m)
        per[m] = {n["name"] for n in heads}
        xs = [n["layer"] for n in heads]
        ys = [abs(float(n["rec_postout"])) for n in heads]
        ax.scatter(xs, ys, s=46, marker=MARKER[m], facecolors="none",
                   edgecolors=COLOR[m], linewidths=1.3, label=LABEL[m], zorder=3)
    if len(models) == 2:
        shared = per[models[0]] & per[models[1]]
    # annotate the heads both models found, which are the ones the text names
    if shared:
        heads = {n["name"]: n for n in confirmed_heads(results, models[0])}
        top = sorted((heads[s] for s in shared), key=lambda n: -abs(float(n["rec_postout"])))[:4]
        for n in top:
            ax.annotate(n["name"], (n["layer"], abs(float(n["rec_postout"]))),
                        textcoords="offset points", xytext=(5, 4), fontsize=7,
                        color="#333333")
    ax.set_xlabel("layer", fontsize=9)
    ax.set_ylabel("recovery of $\\Delta$\n(activation patching)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    if shared:
        ax.set_title(f"{len(shared)} of {len(per[models[0]] | per[models[1]])} confirmed heads "
                     "found by both models", fontsize=8.5, pad=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"circuit_map.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/circuit_map.pdf  ({len(shared or [])} shared heads annotated)")


def kcurve(runs: Path, out: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    nulls = []
    for m in MODELS:
        f = runs / f"final_{m}" / "validation.json"
        if not f.exists():
            print(f"  skip {m}: missing {f}")
            continue
        d = json.load(open(f))
        rows = sorted(d.get("k_curve", []), key=lambda r: r["k"])
        if not rows:
            continue
        ax.plot([r["k"] for r in rows], [r["faithfulness"] for r in rows],
                marker=MARKER[m], color=COLOR[m], linewidth=1.6, markersize=5,
                label=LABEL[m], zorder=3)
        fr = d.get("faithfulness_random", {})
        if fr.get("n"):
            nulls.append(fr["mean"])
    if nulls:
        ax.axhline(max(nulls), linestyle="--", color="#777777", linewidth=1.1, zorder=2)
        ax.annotate("random circuits of the same size",
                    xy=(0.98, max(nulls)), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7.5, color="#555555")
    ax.set_xlabel("circuit size (confirmed heads)", fontsize=9)
    ax.set_ylabel("faithfulness\n($\\Delta$ recovered by the circuit alone)", fontsize=9)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"kcurve.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {out}/kcurve.pdf")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="/workspace/results")
    p.add_argument("--runs", default="/workspace/repo/outputs/circuits")
    p.add_argument("--out", default="figures")
    args = p.parse_args()
    try:
        import matplotlib                                              # noqa: F401
    except ImportError:
        raise SystemExit("matplotlib is not installed in this environment:\n"
                         "  uv pip install --python $VENV_ANALYSIS matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    circuit_map(Path(args.results), out)
    kcurve(Path(args.runs), out)


if __name__ == "__main__":
    main()
