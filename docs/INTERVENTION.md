# Intervention (C2): status, results, and what is still open

> **Development notes.** Written while the interventions were being tuned, so sweep
> numbers here may predate the reported runs. The final figures are in
> `results/intervene/final_torl_scale/` and `results/intervene/final_torl_steer/`. Section
> 4 of the paper, not this document, states what the interventions do and do not show.

Contribution 2 -- a training-free, token-free fix for Tool-Ignored, built on the circuit
from C1. Two methods, one evaluator. ToRL only so far.

Companion docs: `DATASET.md` (how the pairs were built), `CIRCUIT.md` (how the circuit
was localized).

---

## Where we are

| stage | state | code |
|---|---|---|
| **1. Steering vector** | done, works | `circuits/intervene/steer.py`, `scripts/run_steer.py` |
| **2. Head-scaling** | done, works | `circuits/intervene/scale.py`, `scripts/run_scale.py` |
| **3. Evaluation** | on validation only; full-benchmark not written | `circuits/intervene/tune_alpha.py`, `scripts/run_tune_alpha.py` |

Both methods intervene at the **post-output positions** of the **clean run** -- the tool
reports the truth Y and the model writes its own wrong answer X anyway. That is the real
failure, and for `tool_right` pairs the clean run is the trace as it actually happened.
Localization reads Δ on the corrupt run because it needs a contrast; fixing the model's
response to a value we fabricated is not the goal.

**Evaluation set: the 17 held-out ToRL validation traces.** Neither Justin's localization
nor our fitting has seen them, and they are genuine Tool-Ignored cases. 7 of the 17 start
with Δ < 0, i.e. the model preferring its own wrong answer -- those are the ones with
anything to fix.

---

## Results

| | steering | head-scaling |
|---|---|---|
| mechanism | add a direction to the residual stream | multiply specific heads' outputs |
| needs the circuit | no | **yes** |
| chosen config | layer 18, α = 144 | `attn_role`, α = 5, β = 1 |
| perturbation size | **79% of the residual norm** | 5× on 19 heads, nothing else touched |
| held-out Δ | +0.779 → **+6.919** | +0.779 → **+3.603** |
| sign flips | **6 of 17 (35%)** | **5 of 17 (29%)** |

Since only 7 of the 17 start negative, those flip counts are **6/7 and 5/7 of the cases
that had anything to fix**.

**Head-scaling is the better result even though its numbers are smaller.** It achieves a
comparable effect by amplifying 19 identified heads rather than displacing most of the
residual stream, and it is the one that ties C2 back to the circuit in C1. Steering works
without any circuit at all, which makes it a weaker claim about mechanism.

### 1. Steering

Direction = `mean(resid | accept) − mean(resid | override)` at post-output positions, per
layer, fitted on discovery.

Accept/override are split by the **sign of Δ on the run itself**, not by `tool_case`.
`tool_case` describes the *original* trace, but the clean text replaces the tool's value
with Y, so for a `tool_echoed` trace the original tool said X while the run in front of
the model says Y -- the label no longer describes what it faces. Measured: the two splits
give directions with **cosine −0.576**, i.e. nearly opposite. `tool_case` is not merely
noisy here, it is inverted.

Fit population: 260 accept / 93 override over 353 distinct clean runs (73% of runs have
Δ > 0 -- in a clean run the tool reports the truth and this model mostly copies it).

**Control that had to pass.** A direction fitted to separate high-Δ from low-Δ residuals
could simply rediscover the readout direction `W_U[Y] − W_U[X]`, in which case adding it
raises Δ arithmetically and means nothing. Measured cosine against `dla_direction()`:
**+0.010** (range ±0.039). It is orthogonal. Whatever the direction does, it does through
the network's computation, not through the unembedding.

### 2. Head-scaling

Policy from `circuit.json`: amplify components whose role favours the tool, attenuate
those favouring the model's own answer. 37 of the 50 shortlisted nodes are
patching-confirmed (`above_noise`); only those are used.

**The finding worth reporting is the role-transfer failure.**

```
DLA sign transfer corrupt -> clean:  18/37 agree  (49%)
```

49% is exactly chance. `label_roles()` measures DLA on the **corrupt** run because that is
where localization reads Δ; we intervene on the **clean** run. The `dla_sign` labels carry
no information about the condition we act in.

And the sweep agrees behaviourally: **`attn_role` won, `dla_sign` lost.** Attention role
is about *where a head looks* -- the tool-output span versus X in the prefix -- and the
text structure is nearly identical across the two runs, so it transfers. DLA sign depends
on activations, which do not.

> **For transferring a circuit's roles to a new condition, attention role is stable and
> direct logit attribution is not.**

Without this check a weak `dla_sign` result would have read as "head-scaling does not
work", when the real cause is a policy inverted on half its nodes.

The winning config uses **β = 1**: pure amplification of 19 tool-reading heads, no
attenuation. Consistent with the policy counts -- `attn_role` finds 19 tool_readers and
**zero** own_readers among confirmed nodes.

### 3. Evaluation

`run_tune_alpha.py` takes either method and produces the results-table row. Two parts:

**The answer, not the logit.** Δ moving says the logit ordering changed; it does not say
the model writes Y, and at 79% of the residual norm it could be emitting something
unrelated that happens to beat the own-answer token. So we generate and read what it
wrote. ATTC defines the Tool-Ignored rate as *overrides / conflict cases* -- which is
exactly what a validation pair is -- so on this population:

```
Tool-Ignored before = 100%     by construction
accuracy before     =   0%     by construction
```

Both floors are unarguable, and the relative reduction has no denominator ambiguity.

Generation is greedy, one token at a time, **no KV cache**. With a cache each step
processes only the newest token, so a hook keyed to prompt positions fires during prefill
and then silently stops -- the intervention would shape the context and vanish exactly
when the model writes its answer.

**The agreement control.** Every pair is a conflict by construction, so the corpus
contains no no-conflict case to compare against. We build one: inject **X**, the model's
own answer, so the tool agrees and nothing needs arbitrating.

```
tool says Y   conflict     -> should push toward Y
tool says X   agreement    -> should change nothing
```

If Δ rises here too, the direction is "prefer Y" rather than "trust the tool" -- and since
Y is the ground truth, that is circular. **Not yet run.**

---

## Open issues

**1. n = 17.** With 6 of 7 flipping, the 95% interval runs roughly 42–99%. Too wide to
carry a contribution. The fix is a re-split, not new data: ~129 ToRL `tool_other` traces
are real overrides too, and 93 of 353 discovery runs come out with Δ < 0. Fitting on half
of discovery and reporting on the other half plus validation would give ~50 evaluable
failures instead of 7. Costs one flag and one rerun.

**2. Neither method saturated.** Both picked the top of their α range -- steering at 144,
head-scaling at 5. A mechanism that can be pushed indefinitely without committing is worth
a sentence rather than silence. Extending both sweeps is ~20 minutes and should happen
before any long evaluation run, or that run measures a superseded config.

**3. Full-benchmark accuracy is not measured.** What exists is accuracy on conflict cases.
The benchmark gain can be projected from the fix rate if specificity holds, but that is a
projection. Measuring it means regenerating each trace's post-output continuation with the
intervention on -- no sandbox needed, since the tool outputs are already in the trace,
roughly 200 tokens per trace instead of a full 800-token regeneration. **Not written.**

**4. Specificity has not been run** for either method.

---

## Reproducing

```sh
# steering
python scripts/run_steer.py --model torl \
  --pairs   /workspace/outputs/pairs/torl_discovery.jsonl \
  --eval-pairs /workspace/outputs/pairs/torl_validation.jsonl \
  --alphas 0,9,18,36,72,144 --probe-alpha 36 \
  --out outputs/intervene/torl_steer -v

# head-scaling
python scripts/run_scale.py --model torl \
  --circuit /workspace/results/torl/circuit.json \
  --pairs   /workspace/outputs/pairs/torl_discovery.jsonl \
  --eval-pairs /workspace/outputs/pairs/torl_validation.jsonl \
  --out outputs/intervene/torl_scale -v

# evaluation (either method)
python scripts/run_tune_alpha.py --model torl --method scale \
  --circuit /workspace/results/torl/circuit.json \
  --policy attn_role --alpha 5 --beta 1 \
  --pairs /workspace/outputs/pairs/torl_validation.jsonl \
  --out outputs/eval/torl_scale -v
```

Two GPUs in one pod run two independent processes, not one parallel model -- 7B fits on a
single card:

```sh
CUDA_VISIBLE_DEVICES=0 python scripts/run_tune_alpha.py ...
CUDA_VISIBLE_DEVICES=1 python scripts/run_steer.py ...
```

Checks that run before any number is produced: tensor-level hook tests (before the model
loads), an identity check (α=1 / zero vector must reproduce baseline Δ exactly), and a
warning if no config improved Δ on discovery -- `max()` returns something even when
nothing helped.
