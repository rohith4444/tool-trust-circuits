# Matched-pair dataset: what the circuit pipeline expects

Input to `scripts/run_localize.py`, `run_validate.py`, `run_universality.py`
(`--pairs <file>`). Parsed by `circuits/data/pairs.py`; everything token-level is derived
there, per model, so this file contains text only.

## Format

jsonl, one matched clean/corrupt pair per line.

| Field | Type | Meaning |
|---|---|---|
| `problem_id` | str | Unique per line. Row key for cross-model alignment. |
| `model` | str | `torl` \| `simpletir` \| `demyagent`. Recorded, not filtered on. |
| `prefix` | str | Trace text from the start **through the tool-output opening delimiter** (`\n```output\n`). Contains the reasoning answer. |
| `own_answer` | str | The model's reasoning answer, as written in `prefix`. |
| `tool_clean` | str | Correct tool output (agrees with `own_answer`). |
| `tool_corrupt` | str | Plausible wrong tool output. |
| `suffix` | str | Text from the **closing delimiter** (`\n```\n`) through `\boxed{`. Same for both runs. |
| `ground_truth` | str | Optional. |
| `label` | str | Optional: `accept` \| `override`. Used only by `--label`. |

Extra fields are kept as metadata.

Example line:

```json
{"problem_id": "math500-17", "model": "torl",
 "prefix": "...so a^2 = 18. Let me verify.\n```python\nprint(...)\n```\n```output\n",
 "own_answer": "18", "tool_clean": "18", "tool_corrupt": "17",
 "suffix": "\n```\nThe final answer is \\boxed{", "ground_truth": "17", "label": "override"}
```

## How a pair is used

```
text_clean   = prefix + tool_clean   + suffix + shared
text_corrupt = prefix + tool_corrupt + suffix + shared
```

`shared` is the common token prefix of `own_answer` and `tool_corrupt` (Qwen tokenizes digits
one at a time, so "18" vs "17" share "1"). The last position's logits then predict the first
divergent token, and Δ = logit(tool) − logit(own) is read there on the corrupt run. The clean
run is the patching source only; `tool_clean` never enters Δ.

## Requirements

Text
- Delimiters live in `prefix` / `suffix`; nothing is inserted between the fields.
- One injection point per pair: the value after the final opening delimiter. Earlier code and
  output blocks stay verbatim inside `prefix`.
- Raw completion text. No BOS, no chat template, unless it is part of `prefix`.
- `own_answer` appears verbatim in `prefix` as a standalone number. If missing, the pair still
  runs but gets no attention-role label.

Values
- `own_answer` and `tool_corrupt` are in the surface form the model writes inside `\boxed{}`.
  They must differ at some token: "18" vs "18.0" never diverges and is dropped.
- `tool_clean` and `tool_corrupt` tokenize to the same length (same digit count). Otherwise
  the pair is dropped. Prefer near-miss corruptions.
- Same `suffix` for both runs. It fixes the continuation; the metric is "answer now", not the
  model's free continuation after the tool output.

Size
- A pair must fit the model context: 4096 tokens for ToRL, 8192 for the others.
  `--max-tokens` left-truncates, cutting the start of the problem.
- The EAP-IG stage truncates to `--eap-max-tokens` (default 2000) on its own.
- Validation pools over all pairs passed. Dozens of pairs minimum; ablation means computed from
  a handful still carry the signal.
- Accept and override pairs are pooled unless `--label` is given.

Universality
- CKA needs the same texts fed to every model. Either one shared file used for all models, or
  per-model files sharing `problem_id`s. Rows are aligned by `problem_id` and mean-pooled per pair.
- Node overlap is only meaningful between same-architecture models (ToRL, SimpleTIR).

## Silently dropped pairs

Logged at DEBUG (`-v`), counted in the "built N/M pairs" line:
own/tool token-prefix clash; clean/corrupt value lengths differ; tokens differ outside the
value span; truncation would cut the value or every occurrence of `own_answer`.
