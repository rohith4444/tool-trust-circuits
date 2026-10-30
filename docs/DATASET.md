# Layer 2 matched pairs — build provenance and schema

> **Development notes.** This describes the *first* pair build. The corpus was later
> rebuilt to perturb the corrupted value at the divergence index rather than the last
> digit, which changed every count below. The paper uses the rebuilt corpus: **712 / 516 /
> 216 discovery pairs** (ToRL / SimpleTIR / DemyAgent) and **34 / 22 / 28 held-out pairs**,
> the latter being 17 / 11 / 14 problems at two pairs each. The gate structure and schema
> in sections 2 onward still apply; the intermediate counts do not. The README and
> `results/` are authoritative for anything the paper reports.

Six JSONL files at `outputs/pairs/`, also on HuggingFace at
`tool-trust-circuits/tool-trust-circuits` (private, `repo_type=dataset`) under `pairs/`.

```
torl_discovery.jsonl        912 pairs      torl_validation.jsonl        42 pairs
simpletir_discovery.jsonl   664 pairs      simpletir_validation.jsonl   26 pairs
demyagent_discovery.jsonl   250 pairs      demyagent_validation.jsonl   30 pairs
                          -----                                       ----
                          1,826                                         98      = 1,924 pairs
```

Regenerate with `python scripts/build_pairs.py`. Everything below is printed by
that script, so the numbers can be re-derived rather than taken on trust.

---

## 1. All traces → wrong traces (8,508 → 2,691)

Six benchmarks, 2,836 problems, three models, one sample per problem at the
decoding settings in `configs/models.py` (ATTC Table 4).

```
gsm8k 1319 + math500 500 + minerva_math 272 + olympiadbench 675
      + aime24 30 + amc23 40  =  2,836   ×3 models  =  8,508 traces
```

A trace is **wrong** when `score[0] is False`. `score` is written at generation
time by `grader.py:math_equal(pred, gt)` — deterministic, no LLM involved. (The
Qwen3-32B judge is a separate Layer 1 artifact and has no bearing on anything
here.)

| model | traces | wrong |
|---|---|---|
| torl | 2,836 | 694 |
| simpletir | 2,836 | 789 |
| demyagent | 2,836 | 1,208 |
| | **8,508** | **2,691** |

Only wrong traces can become pairs. The readout is `logit(Y) − logit(X)`, so a
trace where the model was already right gives `X == Y` and the metric collapses
to zero by construction.

## 2. Wrong → eligible (2,691 → 1,318 → 1,190)

Three gates. The first is the substring test; the next two remove traces that
are in the wrong pool for reasons that have nothing to do with the model
disagreeing with anything.

**Gate 2 — has a tool output and a boxed answer.** `"```output" in code and
"\boxed{" in code`. Leaves **1,318** (torl 595, simpletir 423, demyagent 300).

**Gate 3 — the boxed answer actually closes.** `last_boxed(code) is not None`.
An unclosed brace means generation stopped mid-answer. `parser.py:extract_answer`
walks those same braces, so `pred` comes back as a fragment — `'2'` where the
model was writing `'22'`. Removes **100**, all DemyAgent.

**Gate 3b — the model wasn't actually right.** Drops traces whose final
`\boxed{...}` equals the ground truth despite being scored wrong. These are
`extract_answer` failures: one ToRL trace had a whole run-together sentence as
`pred` while its `\boxed{0}` matched `gt` exactly. Removes **28** (27 DemyAgent,
1 ToRL).

| model | wrong | gate 2 | truncated | actually correct | **eligible** |
|---|---|---|---|---|---|
| torl | 694 | 595 | 0 | 1 | **594** |
| simpletir | 789 | 423 | 0 | 0 | **423** |
| demyagent | 1,208 | 300 | 100 | 27 | **173** |
| | 2,691 | 1,318 | 100 | 28 | **1,190** |

DemyAgent loses most of its pool because it frequently doesn't terminate: 68% of
its wrong traces contain no `\boxed{` at all and a further 9% have one that never
closes. That is a generation-length property of the model, not a parsing bug —
verified with `scripts/check_parse.py`, which found genuine parse failures at
only 1.1% corpus-wide.

## 3. Eligible → traces that yield pairs (1,190 → 962)

Four more gates, all about whether a usable Δ readout can be constructed.

**`divergence_none` (80).** X and Y must differ *in token space* at some
position. When one is a token-prefix of the other — `X='36.36' Y='36'`,
`X='3000' Y='300'`, `X='' Y='106'` — no single position separates them and there
is nothing to read.

**`no_digit_in_Y` (16).** Z is fabricated by perturbing a digit of Y. Answers
like `even`, `(B)`, `ellipse`, `SRAC=LRAC` have none.

**`too_long` (0).** Would exceed the model's `max_model_len`.

**`prefix_retokenised` (132).** Where X and Y share a leading token prefix, that
prefix is teacher-forced by appending it to the probe. The prefix is decoded
from Y's token ids and then re-encoded as part of the full prompt, and that round
trip can merge it with the preceding `\boxed{`. The build checks the encoded
prompt actually ends in the intended token ids and drops the pair otherwise —
without this the readout would silently land one position off. Fires on ~11% of
otherwise-good traces, mostly `prefix='-'`, `prefix='\\'`, `prefix='\frac{'`.

| model | eligible | div_none | no_digit | prefix | **traces** | **pairs** |
|---|---|---|---|---|---|---|
| torl | 594 | 38 | 3 | 76 | **477** | 954 |
| simpletir | 423 | 22 | 9 | 47 | **345** | 690 |
| demyagent | 173 | 20 | 4 | 9 | **140** | 280 |
| | 1,190 | 80 | 16 | 132 | **962** | **1,924** |

Every surviving trace produced exactly 2 pairs (954 = 477×2 etc.), so the two
Z-variants are always distinct.

## 4. Pair construction and the discovery / validation split

### The pair

One trace → two texts differing at exactly one place, the value inside the
**last** ` ```output ` block:

```
CLEAN      ```output  Y  ```  + probe     the tool backs the correct value
CORRUPTED  ```output  Z  ```  + probe     the tool backs a wrong value
```

Everything before that block is byte-identical, earlier output blocks included.
The probe is the same string for all three models:

```python
PROBE = "\nThe final answer is \\boxed{"
```

It is in-distribution everywhere because every prompt instructs the model to
answer in `\boxed{}` and the prompt is still in context — the text is cut *after*
the output block, not before it.

**Z** is a near-miss: the last digit of Y, shifted, everything else copied.
`113 → 114 / 112`, `1.25 → 1.26 / 1.24`, `\frac{1}{2n+2} → \frac{1}{2n+3}`. Near
rather than far on purpose — `113 → 213` is nearly double and a model can reject
it as absurd without any trust arbitration, which would localise "detect an
implausible number" instead of "decide whether to believe the tool". Z is
guaranteed ≠ Y, ≠ X (numerically, not just as a string), and ≠ the other variant.

### The split

Every pair is a conflict *by construction* — clean injects Y, corrupted injects
Z, neither of which is X. The split is about the **original** trace:

| `tool_case` | condition | split | count (traces) |
|---|---|---|---|
| `tool_right` | tool output == Y | **validation** | 49 |
| `tool_echoed` | tool output == X | discovery | 418 |
| `tool_other` | tool output ∉ {X, Y} | discovery | 391 |
| `no_answer` | `Done` / `Timeout Error` / unparseable | discovery | 104 |

`tool_right` is real Tool-Ignored: the tool was correct and the model overrode
it. **For those traces the clean run reproduces what actually happened** — the
tool really did output Y. For every other case the clean run is counterfactual:
the tool said `49` and we overwrote it with `113`.

That is the whole point of the split — **discover the circuit on manufactured
conflicts, validate it on real ones.**

Validation is small: **49 traces / 98 pairs** (torl 21, simpletir 13,
demyagent 15). That is the phenomenon, not a defect — rejecting a *correct* tool
is rare, because when the tool is right the model usually reports it and the
trace is then correct, so gate 1 discards it.

`tool_other` (391 traces) is the robustness lever: those are conflicts the model
genuinely resolved, just with a tool that happened to also be wrong. They stay
tagged in the discovery files, so widening validation needs no rebuild —
filter on `tool_case`, don't regenerate.

---

## 5. Schemas

### Pair record (`outputs/pairs/*.jsonl`, one JSON object per line)

| field | type | notes |
|---|---|---|
| `pair_id` | str | `{model}_{benchmark}_{idx}_z{0,1}` — unique, verified no duplicates |
| `model` | str | `torl` \| `simpletir` \| `demyagent` |
| `benchmark` | str | one of the six |
| `idx` | int | problem index within that benchmark |
| `X` | str | the model's own answer (`pred` from the trace) |
| `Y` | str | ground truth (`gt`) |
| `Z` | str | fabricated wrong tool value |
| `clean_text` | str | **full prompt**, tool block = Y, ending in the probe |
| `corrupted_text` | str | identical except the block = Z |
| `readout` | obj | see below |
| `split` | str | `discovery` \| `validation` |
| `tool_case` | str | `tool_right` \| `tool_echoed` \| `tool_other` \| `no_answer` |
| `tool_value` | str | raw text of the original last output block |
| `tool_answer` | str \| null | number extracted from it |
| `tool_correct` | bool \| null | `tool_answer == Y`; null when unparseable |
| `single_block` | bool | trace had exactly one output block (78% do) |
| `numeric` | bool | both X and Y parse as numbers |
| `n_output_blocks` | int | |
| `n_tokens` | int | length of `clean_text` |

`readout`:

| field | type | notes |
|---|---|---|
| `tok_Y` | int | token id to read the positive logit at |
| `tok_X` | int | token id to read the negative logit at |
| `divergence_index` | int | position within the answer where X and Y first differ |
| `forced_prefix` | str | shared prefix, **already appended to both texts** |

### Source trace record (`outputs/{model}_seed0/{benchmark}/*.jsonl`)

Produced by the vendored ATTC harness. Fields are lists because the harness
supports `n_sampling > 1`; we run `n=1`, so take index `[0]`.

| field | notes |
|---|---|
| `idx` | problem index |
| `question` | problem statement |
| `gt` | ground truth answer |
| `gt_cot` | reference solution |
| `code` | **the full generated trace** — reasoning, ` ```python ` blocks, ` ```output ` blocks |
| `pred` | parsed final answer (`parser.py:extract_answer`) |
| `score` | `math_equal(pred, gt)` |
| `report` | execution error, or null |

---

## 6. Computing Δ

```python
Δ = logit(tok_Y) − logit(tok_X)
```

The same two token ids in **both** runs. Never swapped, never re-derived per
run — only the injected value changes between clean and corrupted.

```python
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

p = json.loads(open("pairs/torl_validation.jsonl").readline())
tok = AutoTokenizer.from_pretrained("GAIR/ToRL-7B")   # must match p["model"]
model = AutoModelForCausalLM.from_pretrained("GAIR/ToRL-7B", torch_dtype=torch.bfloat16)

def delta(text):
    ids = tok(text, return_tensors="pt").input_ids
    logits = model(ids).logits[0, -1]                  # next-token distribution
    return (logits[p["readout"]["tok_Y"]] - logits[p["readout"]["tok_X"]]).item()

gap = delta(p["clean_text"]) - delta(p["corrupted_text"])
```

Three things that will silently give wrong answers:

1. **`tok_Y` / `tok_X` are tokeniser-specific.** Load the tokeniser for
   `p["model"]`, not a shared one. ToRL and SimpleTIR have different bases.
2. **Don't append anything to `clean_text`.** The probe and `forced_prefix` are
   already in it; it ends mid-answer on purpose. Read at the final position.
3. **Don't add special tokens** when encoding, and don't re-tokenise
   `forced_prefix` separately — the build already verified the encoded prompt
   ends in the intended ids, and re-deriving it can shift the readout.

`gap` is the tool value's causal effect on the model's answer: how much the
correct-tool run favours the truth over the model's own answer, relative to the
wrong-tool run. That is what EAP-IG ranks components by and what patching flips.

---

## 7. Known limitations

- **Only the last output block is inspected.** 22% of traces (216 of 962) have
  more than one. If a model ignored a correct result at call 2 and erred
  differently at call 4, we classify on call 4 and miss it. This is part of why
  our `tool_right` rate (5.0%) sits below ATTC's judged Tool-Ignored rate
  (10–20%) — their judge reads the whole trace.
- **Validation n=49** — about 20 per model. Pool for headline numbers and report
  per-model with confidence intervals; `tool_other` is the robustness lever.
- **DemyAgent accuracy is below ATTC's published figures** (olympiadbench 28.0 vs
  51.9, aime24 3.3 vs 40.0). Root cause is non-termination: 77% of its wrong
  traces have no usable boxed answer. `max_model_len=16384`, which we set for
  throughput, is the leading suspect and is untested. DemyAgent is the
  universality model whose accuracy we never report, so this does not affect any
  published number — but it does mean its 173 eligible traces are a biased
  sample of its behaviour.
- **Multi-block traces keep their earlier output values.** Only the last block is
  swapped, so in those the model may already have seen the true value earlier.
  Filter on `single_block` if that matters for a given analysis.
