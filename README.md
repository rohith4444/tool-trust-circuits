# tool-trust-circuits

Mechanistic analysis of **tool trust** in tool-integrated reasoning (TIR) models: where
inside the model the decision is made to accept a returned tool output or override it with
the model's own earlier answer, and whether that decision can be steered.

Code for the workshop paper *Localizing and Steering a Tool-Trust Circuit in
Tool-Integrated Reasoning Agents* (NeurIPS 2026 EvoRobust, non-archival).

## Pipeline

Five stages. Each writes to `--out` and pushes that directory to a private HF dataset
repo; every stage reads the previous stage's files, so they can be run days apart on
different pods.

| Stage | Script | Reads | Writes |
|-------|--------|-------|--------|
| 0 Generation | `run_generation.py` | benchmarks | traces (jsonl) |
| 1 Pairs | `build_pairs.py` | traces | matched clean/corrupted pairs |
| 2 Localize | `run_localize.py` | pairs | `node_scores.csv`, `patch_results.csv`, `circuit.json` |
| 3 Validate | `run_validate.py` | pairs + `circuit.json` | `validation.json`, `validation_pairs.csv` |
| 4 Universality | `run_universality.py` | pairs + all `circuit.json` | CKA matrices, `overlap.json` |
| 5 Intervene | `run_scale.py`, `run_steer.py` | pairs + `circuit.json` | sweeps, held-out eval |

```sh
# localize, validate, and intervene on one model
python scripts/run_localize.py  --model torl --pairs pairs/torl_discovery.jsonl --out outputs/circuits/torl
python scripts/run_validate.py  --model torl --pairs pairs/torl_discovery.jsonl \
    --circuit outputs/circuits/torl/circuit.json --only-above-noise --rank-by rec_postout \
    --protect-layers 0 --protect-types m --specificity --out outputs/circuits/torl_final
python scripts/run_scale.py     --model torl --circuit outputs/circuits/torl/circuit.json \
    --pairs pairs/torl_discovery.jsonl --eval-pairs pairs/torl_validation.jsonl --out outputs/intervene/torl
```

`scripts/paper_numbers.py` reads every artifact above and prints each number the paper
quotes; `scripts/make_figures.py` regenerates both figures from the same files.

## Reproducing the paper

Every number and figure in the paper is computed from `results/`, which is committed here
(1.2 MB of JSON and CSV). No GPU, no model download, no HuggingFace account.

```sh
uv venv .venv-tools && uv pip install --python .venv-tools pandas numpy matplotlib
.venv-tools/bin/python scripts/paper_numbers.py     # every number, in paper order
.venv-tools/bin/python scripts/make_figures.py      # both figures into figures/
```

Both default to reading `results/`, so neither takes an argument. Point `--results`,
`--runs` and `--intervene` at a fresh run directory to read that instead.

`paper_numbers.py` prints every quoted number in five blocks, in paper order. What backs
what:

| Paper | Artifact |
|-------|----------|
| Δ validity (98.0/96.3/91.8% lowered, sign at chance) | `results/{model}/pairs_summary.csv` |
| Circuit size, layer range, head roles | `results/{model}/circuit.json`, `node_scores.csv`, `patch_results.csv` |
| Validation table (necessity, faithfulness, specificity, self-repair) | `results/final_{model}/validation.json` |
| Faithfulness-vs-k curve | `results/final_{model}/faithfulness_k.csv` |
| Specificity by region | `results/final_{model}/damage.csv` |
| Universality: 9 of 12 heads shared, Jaccard 0.70 | `results/{model}/circuit.json` |
| CKA between models | `results/final_universality/cka_*.npy` |
| Head scaling, α=4 | `results/intervene/final_torl_scale/` |
| Steering, layer 18 | `results/intervene/final_torl_steer/` |

## Layout

```
circuits/           the library: hooks, pair construction, and one module per stage
  data/pairs.py       clean/corrupted pair construction and the Δ readout
  hooks.py            node ids, caching, replace/scale/steer hooks
  localize/           EAP-IG screen, activation patching, DLA and attention roles
  validate/ablate.py  necessity, faithfulness, specificity, self-repair, controls
  intervene/          head scaling, steering, alpha tuning, TIR-loop regeneration
  universality/       linear CKA and node overlap
scripts/            one runnable entry point per stage, plus checks and reporting
generation/         ATTC's harness, vendored (see generation/ATTRIBUTION.md)
configs/models.py   the model registry: repos, prompt types, decoding, ATTC's numbers
docs/               dataset, circuit, validation and intervention notes
envs/               three pinned environments (see below)
results/            the artifacts every paper number is computed from (1.2 MB)
figures/            the paper's two figures, regenerable from results/
```

`generation/` is vendored third-party code and keeps its upstream licence. Everything
else is MIT; see [LICENSE](LICENSE).

## Environments

Three, because vLLM and TransformerLens pin conflicting `torch`/`transformers` versions:

| Env | Contents | Where |
|-----|----------|-------|
| `tools` | `huggingface-hub`, `transformers`, `datasets` | laptop |
| `gen` | `vllm`, ATTC's grader | GPU pod, CUDA only |
| `analysis` | `transformer-lens`, `torch`, EAP-IG | GPU pod |

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv venv .venv-tools && uv pip install --python .venv-tools -r envs/tools.txt
```

On a pod, `source scripts/pod_env.sh` sets the caches, both venv paths, and the git SSH
key. It has to be sourced *after* pulling, or newly added variables will be missing from
the shell.

## Data

What is in this repo:

- `results/` — the 1.2 MB of JSON and CSV every paper number is computed from. Numbers and
  token ids only; no problem statements and no generated text.
- `data/` — ATTC's released benchmark files, committed so the reproduction gate scores the
  same records they scored.
- `figures/` — the paper's two figures.

What is not, and why:

- **Traces (85 MB) and matched pairs (21 MB) are in private HuggingFace dataset repos.**
  They contain full model generations across six benchmarks, and we have not released
  them. Stages 2 through 5 read pairs, so running them needs that access.
- **Large tensors** (`means.pt`, `graph.pt`, CKA features, 730 MB) are not committed. Those
  are re-run inputs, not results.
- Model weights are pulled from HF to the pod's volume cache and never stored here.

To regenerate the pairs from scratch instead, run stage 0 and stage 1. That is 2,836
problems × 3 models of vLLM generation followed by `build_pairs.py`, which needs one
80 GB GPU and reproduces `pairs/` deterministically from the seeds in `configs/models.py`.
`circuits/cli.py::push_outputs` uploads each finished `--out` directory if you set
`HF_TOKEN`; it only warns when the token is missing, so check its output if you expect a
push.

## Status

Accepted at the NeurIPS 2026 EvoRobust workshop (non-archival); this is the camera-ready
code release. Next work is on the problems listed under *What it does not show*, and on
widening the setting beyond a code interpreter to context-versus-prior conflict more
generally.
