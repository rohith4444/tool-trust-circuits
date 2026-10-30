# Circuit validation: what went wrong and what changed

> **Development notes.** This is the record of how the metrics were fixed, including
> intermediate configurations that were tried and discarded. For the final reported
> numbers, read `results/final_{model}/validation.json` or run
> `scripts/paper_numbers.py --results results --runs results`.

Written after the 2026-09-12 debugging session. The numbers here are ToRL unless stated.
Every configuration mentioned is reproducible from `scripts/run_validate.py` flags.

## 0. The metrics were broken first

Δ is not positive by construction. On the corrupt runs `mean(Δ) = −1.54` while
`mean|Δ| = 4.46`, so pooling signed sums across pairs cancelled more than half the signal.

- `necessity.above_random` compared a signed drop (`base − ablated`, negative whenever
  Δ_base is negative) against a positive percentile, so it could not return `True` however
  strong the circuit was.
- `faithfulness = Σcircuit / Σbase` divided two cancelled sums: **−0.43** on SimpleTIR,
  **−0.49** on DemyAgent. Those were not low scores, they were cancellation.

Fixed by computing every statistic per pair and aggregating by magnitude
(`circuits/validate/ablate.py`). That produced the first readable numbers:

    necessity    0.778  vs random 0.133   ->  5.9x
    faithfulness 0.412 @ k=50,  0.831 @ k=200
    specificity  0.012                    ->  failing

Three of those four were still wrong, for three separate reasons.

## 1. The circuit was never confirmed

The pipeline is: **EAP-IG screens -> activation patching confirms -> the confirmed set is
the circuit.** Patching ran and wrote a verdict for every shortlisted node. We then
validated the *unconfirmed* shortlist: the filter (`--only-above-noise`) existed and was
never passed.

What patching had already found, against a noise floor of 0.0011:

    m0      rec_postout  0.000    not confirmed
    a0.h3                0.001    not confirmed
    m1-m5           ~0.000-0.001  not confirmed
    ---
    m26                  0.318    confirmed
    a23.h11              0.237    confirmed

13 of ToRL's 50 were unconfirmed, and the consequence was structural: **23 of the model's
28 MLPs** sat in the "circuit", a near-contiguous block m0-m8 and m17-m27. That is the
backbone, not a mechanism. `m0` was EAP rank 1 or 2 on all three models -- attribution is
largest for components every path runs through.

## 2. The ablation removed the input, not a mechanism

Mean-ablating `m0` replaces token-specific embedding processing with a constant: it
deletes the text, not a computation. Ablating all 28 MLPs deletes the model's ability to
compute anything. Measured:

    faithfulness with m0 inside the circuit      0.412
    faithfulness with m0 ablated                 0.069
    faithfulness, heads-only, all MLPs ablated   0.019

None of those measure a circuit; they measure how much model is left.

## 3. The control was not a control

Random sets were drawn uniformly from all 812 components. A model has `n_layers` MLPs
against `n_layers * n_heads` heads -- 28 against 784 -- so a uniform 37-node sample holds
**~1.3 MLPs**. The circuit held **17**. The comparison was "a set with 17 MLPs versus a set
with 1 MLP", not "these components versus other components".

Running the same circuit against composition-matched controls:

    circuit                      collapse 0.768
    matched random, same recipe  collapse 0.706   ->  margin 1.09x
    unmatched random (reported)  collapse 0.133   ->  margin 5.9x

The 5.9x was an artifact. Any 17 MLPs collapse Δ by 71%.

## 4. Specificity tested the wrong region, against nothing

It ablated the circuit at post-output positions -- where the model assimilates the tool
result, i.e. the circuit's own job. Removing the conflict does not make that region
unrelated work. And there was no random baseline, so `0.012` had no reference point.

## The four fixes

| # | fix | flag |
|---|---|---|
| 1 | use only patching-confirmed nodes | `--only-above-noise` |
| 2 | layer 0 and all MLPs are substrate: never ablated, never circuit | `--protect-layers 0 --protect-types m` |
| 3 | random controls matched on head/MLP composition | now default (`matched_draw`) |
| 4 | specificity in a region with no tool output, against matched random | `--damage-regions preout,postout` |

Fix 2 follows Wang et al.: the circuit is a set of attention heads, only non-circuit heads
are ablated, MLPs and embeddings stay intact. It is also required for the comparison the
paper makes -- IOI's 87% was measured under that scope, and a faithfulness figure from a
different scope is not comparable to it.

## Trial run after the fixes (200 pairs, 5 control sets)

| | ToRL | SimpleTIR |
|---|---|---|
| circuit | 20 heads (2.5% of model) | 19 heads (2.4%) |
| necessity | 0.299 vs matched 0.065 -> **4.6x** | 0.338 vs 0.058 -> **5.8x** |
| faithfulness | **0.374** | **0.359** |
| specificity, no-tool region | 0.963 vs 0.974 -> **0.99** | 0.951 vs 0.968 -> **0.98** |
| specificity, post-output | 0.933 vs 0.978 -> 0.95 | 0.900 vs 0.962 -> 0.94 |

A pre-registered check held: fix 2 can only move faithfulness, because necessity and
specificity ablate the heads alone and MLPs were already intact for them. Necessity came
back identical (0.299 / 0.065) and specificity identical (0.963 / 0.974); only faithfulness
moved, 0.019 -> 0.374.

## Sensitivity table (report this, not just the final row)

| circuit | size | necessity (matched) | faithfulness | specificity |
|---|---|---|---|---|
| top-50 EAP, unconfirmed | 6.2% | -- | 0.412 | 0.012 fail |
| top-200 EAP, unconfirmed | 24.6% | -- | 0.831 | -- |
| 37 confirmed + protect L0 | 4.6% | 1.09x **fail** | 0.640 | floored |
| **20 confirmed heads, MLPs as substrate** | **2.5%** | **4.6x pass** | **0.374** | **0.99 pass** |

Only the last row passes every control it is measured against, and it does so on two
models independently. Publishing the sensitivity table rather than the final row alone is
what separates "we adopted a published convention for a stated reason" from "we searched
configurations until one passed" -- the configurations were searched, so the search
belongs in the paper.

## What the result says

A small set of late-layer attention heads is **necessary** for the tool-trust decision and
**specific** to it, but only **partially sufficient** (0.37). The MLP stack carries the
rest of the effect and is not selective -- any comparable set of MLPs does the same work.
Heads select, MLPs compute.

ToRL and SimpleTIR converge on 11 of their top 12 heads independently -- `a22.h3`,
`a23.h11`, `a25.h12`, `a26.h26`, `a23.h19`, `a26.h25` -- which is the universality result.
