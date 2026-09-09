# Circuit pipeline: dataloader and resulting dataset

What `scripts/run_localize.py`, `run_validate.py` and `run_universality.py` feed to the
model, how the loader in `circuits/data/pairs.py` builds it from the released pair files,
and how many pairs come out. Self-contained; `docs/DATASET.md` has the build provenance
of the files themselves if you want it.

---

## 1. Pair

Every pair comes from one math trace where the model called a tool and then gave a
**wrong** final answer. Three values matter:

| symbol | meaning |
|---|---|
| **X** | the model's own final answer, the wrong one it actually wrote |
| **Y** | the ground truth |
| **Z** | a fake tool value: Y with its last digit shifted (`192 → 193`) |

The trace is cut right after its last tool-output block and finished with a fixed probe,
`The final answer is \boxed{`. Two texts are made that differ in exactly one place, the
value inside that block:

```
clean      ...```output\n 192 \n```\n\nThe final answer is \boxed{      tool says the truth Y
corrupt    ...```output\n 193 \n```\n\nThe final answer is \boxed{      tool says the fake Z
```

The measurement is the model's very next token. Qwen writes one digit per token, so the
model commits to an answer at the first digit where X and Y differ. Two token ids are
fixed per pair, `tok_Y` and `tok_X` (that first differing digit of each), and the metric is

```
Δ = logit(tok_Y) − logit(tok_X)        read at the last position, same two ids in both runs
```

"Right now, how much does the model prefer the truth over its own answer?" Where X and Y
share leading digits, that shared prefix is already appended after `\boxed{` in both
texts (the file calls it `forced_prefix`), so the next token is always the differing one.

The pipeline reads Δ on the **corrupt** run and uses the **clean** run as the source for
activation patching. A component's effect is

```
recovery = (Δ_patched − Δ_corrupt) / (Δ_clean − Δ_corrupt)
```

how much of the correct-tool preference it restores when its clean activation is swapped
into the corrupt run. EAP-IG ranks components by the same clean-minus-corrupt difference.

## 2. Dataloader

Files are `pairs/<model>_<split>.jsonl`, one JSON object per line, with `X`, `Y`, `Z`,
`clean_text`, `corrupted_text`, `readout` (`tok_Y`, `tok_X`, `divergence_index`,
`forced_prefix`), `model`, `pair_id`, `tool_case` and a few tags. The loader
(`load_examples` → `build_pair`) turns each line into a `Pair` of token-id tensors:

1. **Split the text.** `clean_text` is cut at the last `` ```output\n ``. What precedes it
   (including the fence) is the prefix; the remainder starts with Y and ends with
   `forced_prefix`, and what sits between them is the suffix. The pieces must
   re-concatenate to both stored texts byte for byte, or the loader raises. A wrong cut
   would read Δ one token off, silently.
2. **Re-derive everything token-level with the loaded model's tokenizer:** the token span
   of the injected value, the first divergent token of X vs Y (so `tok_X`, `tok_Y`,
   the shared prefix), the post-output positions, and every occurrence of X in the prefix
   (for attention-role labelling). The file's stored ids are not trusted blindly.
3. **Check against the file.** When the file's `model` is the loaded model, the
   recomputed `tok_Y / tok_X / divergence_index` must equal the stored `readout`. A
   mismatch means the wrong tokenizer is loaded and the run stops. For any other model
   the stored readout is ignored (see §5, universality).
4. **Keep only true minimal pairs at the readout** (§3).

The two texts, tokenized, have the same length and differ only inside the value span.
Every field of the row is carried along as `Pair.meta`, so results can be sliced by
`tool_case`, `split`, `single_block` and so on afterwards.

## 3. Minimal pairs

Δ reads a single token: the first digit where X and Y differ, which is the very first
digit for 75% of pairs. Z changes Y's **last** digit. For any truth longer than one digit
those are different positions:

```
X = 96    Y = 192    Z = 193        readout digit: '1' (Y) vs '9' (X)
clean      tool says 192   →  a model that trusts the tool writes '1'
corrupt    tool says 193   →  a model that trusts the tool writes '1'    same token
```

Both runs push the same next token, so Δ_clean ≈ Δ_corrupt and there is no difference
for EAP-IG to attribute or for patching to recover. The pair is not minimal *at the
token we read*, even though the texts differ by one character.

So the loader keeps a pair only if, at the divergence index, **Z's token differs from Y's
and from X's**. The second clause drops a few pairs where the fake happens to back the
model's own answer, which is a different experiment. What survives are pairs where Y's
last digit is also its first differing digit: single-digit truths (`X=2 Y=7 Z=8`) or
multi-digit ones whose leading digits match X (`X=12 Y=13 Z=14`).

## 4. Resulting dataset

Counted by running the loader on the released files with each model's own tokenizer
(run any script with `-v` to see every drop and its reason):

| file | pairs in file | usable pairs | traces | `tool_echoed` / `tool_other` / `tool_right` / `no_answer` |
|---|---|---|---|---|
| torl_discovery | 912 | **174** | 87 | 80 / 62 / – / 32 |
| torl_validation | 42 | **13** | 8 | – / – / 13 / – |
| simpletir_discovery | 664 | **128** | 67 | 76 / 41 / – / 11 |
| simpletir_validation | 26 | **8** | 4 | – / – / 8 / – |
| demyagent_discovery | 250 | **50** | 26 | 24 / 19 / – / 7 |
| demyagent_validation | 30 | **4** | 2 | – / – / 4 / – |
| | 1,924 | **377** | 194 | |

Each trace contributes two fake values (Z = last digit +1 and −1), which usually pass
or fail together, so a surviving trace is mostly one clean text against two corrupt
texts. Usable pairs skew toward small answers; keep that in mind when reading role
labels.

`tool_case` says what the *original* tool output was before we replaced it:
`tool_echoed` = it matched X, `tool_right` = it matched Y (the model overrode a correct
tool; those files are the validation split), `tool_other` = neither, `no_answer` =
nothing parseable. ToRL discovery is enough for a first EAP-IG ranking. Validation is
thin; `tool_other` pairs in discovery are real conflicts the model resolved against a
wrong tool and serve as the robustness set.

**Getting the other 80% back is a dataset change, not a loader change:** have
`fabricate_Zs` in `scripts/build_pairs.py` shift the digit at `divergence_index` rather
than the last digit. Every pair would then be readable, with Z a larger perturbation
(`192 → 292`). Not done; the released files are used as-is.

## 5. Labels and universality

**`--label`** filters on `accept` / `override`. Dataset rows carry `tool_case`; the loader
translates:

| `tool_case` | `label` | reading |
|---|---|---|
| `tool_echoed` | `accept` | the model repeated what the tool said |
| `tool_right`, `tool_other` | `override` | the model went its own way |
| `no_answer` | none | nothing to arbitrate; excluded by any `--label` |

**Universality** compares models on the same texts, aligned by `pair_id`. The per-model
files hold different traces and share no ids, so pass **one** file to all `--models`.
Use ToRL's: its pairs fit in 4096 tokens, so they fit every model, and only ToRL's
stored readout is checked against the file. Rows are mean-pooled per pair, so the three
tokenizers need not agree on token counts. (In practice all three are Qwen tokenizers
and produce identical ids on these texts.)

## 6. Drop reasons (logged at DEBUG, counted in `built N/M pairs`)

1. **injected values share the readout token** — §3. Nearly every drop.
2. **corrupt value backs the own answer at the readout** — §3. A few per file.
3. **own/ruler token-prefix clash** — X is a token-prefix of Y or vice versa. The build
   already removed these; should not fire.
4. **clean/corrupt lengths differ** — Y and Z tokenize to different lengths. Same
   character length by construction, so rare.
5. **ids differ outside the value span / shared prefix does not survive** — the file's
   own model passed this check at build time.
6. **truncation** — `--max-tokens` or `--eap-max-tokens` would cut the value span or every
   occurrence of X in the prefix.

**Segment format.** A hand-built row with `problem_id, model, prefix, own_answer,
tool_clean, tool_corrupt, suffix` (optional `ground_truth`, `label`) is still accepted, for
synthetic inputs. There `tool_corrupt` carries the positive logit and `tool_clean` is
expected to agree with `own_answer`. Same `build_pair`, same drop rules.

## 7. First run

```sh
python scripts/run_localize.py --model torl --pairs pairs/torl_discovery.jsonl --out outputs/circuits/torl -v
```

- Log: `built 174/912 pairs`, drops almost all reason 1.
- `pairs_summary.csv`: `delta_clean` above `delta_corrupt` on nearly every row,
  `top1_clean` mostly `tool`. On `torl_validation.jsonl` the clean run is the real trace
  where the model overrode a correct tool, so `delta_clean` should often still be
  negative there: the sign matches observed behaviour.
- `localize_checks.json`: `sign_flip ok`, patch-all recovery ≈ 1, patch-none = 0,
  `dla_sum.ok`.
