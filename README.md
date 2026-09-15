# tool-trust-circuits

Mechanistic analysis of **tool trust** in tool-integrated reasoning (TIR) models: where
inside the model the decision is made to accept a returned tool output or override it with
the model's own earlier answer, and whether that decision can be steered.

The failure this studies is *Tool Ignored*, named by ATTC
([Xu et al., arXiv:2604.08281](https://arxiv.org/abs/2604.08281)): a code interpreter
returns the correct value, it contradicts what the model already reasoned, and the model
keeps its own wrong answer. ATTC fixes it with an external controller. We locate the
decision inside the network instead.

Code for the workshop paper *Localizing and Steering a Tool-Trust Circuit in
Tool-Integrated Reasoning Agents* (NeurIPS 2026 EvoRobust, non-archival).

## What it found

On ToRL-7B, SimpleTIR-7B and DemyAgent-4B, the decision is carried by **14-20 late-layer
attention heads**, 1-3% of each model's components. Measured against controls matched to
the circuit in size *and* head-to-MLP composition, the circuit is **necessary** for the
decision, **specific** to it, and **partially sufficient** (faithfulness 0.19-0.33). ToRL
and SimpleTIR, localized independently, share 9 of their 12 highest-recovery heads.
Scaling those heads, or adding a difference-of-means steering direction, shifts the
decision toward the tool on held-out traces where the model really did override a correct
result, without adding a token or changing the prompt.

Full numbers and the ablation-scope sensitivity table: [docs/VALIDATION.md](docs/VALIDATION.md).

## What it does not show

Stated here because it is the first thing to fix, not because it is hidden:

- **No end-task result.** Activation hooks need TransformerLens, an unintervened
  TransformerLens run reproduces our vLLM traces on only 77% of problems, and conflict
  cases are too rare to rate on a subsample. See `docs/INTERVENTION.md`.
- **Δ is a logit-level preference, not the emitted answer.** On 10 of 17 held-out
  Tool-Ignored traces the model already favours the tool at the readout and still writes
  its own answer.
- **Node-level, not edge-level.** Components are patched, connections are not, so this is
  a validated set of heads rather than a wiring diagram.
- One corruption operator, one seed per model, and 17/11/14 held-out problems.

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

## Three things worth knowing before reading the code

**Δ is read at the divergence token, and Z never enters it.** Δ = logit(Y) − logit(X) at
the first position where the ground truth Y and the model's own answer X differ, with the
shared prefix teacher-forced. The same two token ids are used in both the clean and the
corrupted run. The fabricated wrong value Z changes what the model *sees*, never what is
measured.

**A corrupted value that matches Y at the readout is inert.** Both runs then predict the
same token and Δ is identical in both. Our first build perturbed the last digit, which
fails this for 80% of pairs; the corpus was rebuilt to perturb at the divergence index
instead. `scripts/check_z_contrast.py` is the check.

**Controls must match composition, not just size.** A model has ~28 MLPs against ~784
heads, so a uniform random set of a circuit's size contains almost no MLPs. Measured
against unmatched controls our necessity margin read 10.8×; matched, it read 1.1×. The
same correction applies to faithfulness and specificity, and each one changed a reported
result. `circuits/validate/ablate.py::matched_draw`.

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
```

## Environments

Three, because vLLM and TransformerLens pin conflicting `torch`/`transformers` versions:

| Env | Contents | Where |
|-----|----------|-------|
| `tools` | `huggingface-hub`, `transformers`, `datasets` | laptop |
| `gen` | `vllm`, `math_verify` | GPU pod, CUDA only |
| `analysis` | `transformer-lens`, `torch`, EAP-IG | GPU pod |

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv venv .venv-tools && uv pip install --python .venv-tools -r envs/tools.txt
```

On a pod, `source scripts/pod_env.sh` sets the caches, both venv paths, and the git SSH
key. It has to be sourced *after* pulling, or newly added variables will be missing from
the shell.

## Data

Model weights are pulled from HF to the pod's volume cache and never stored here. Traces,
pairs, labels and results live in private HF dataset repos; `circuits/cli.py::push_outputs`
uploads each finished `--out` directory. Only code, docs and figures are in git.

## Status

The workshop submission is done. Next work is on the problems listed under *What it does
not show*, and on widening the setting beyond a code interpreter to context-versus-prior
conflict more generally.
