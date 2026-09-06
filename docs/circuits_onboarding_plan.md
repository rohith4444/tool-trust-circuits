# Onboarding Plan: Circuits (1 day)
### Circuit Analysis of Tool Trust in Tool-Integrated Reasoning

**Scope:** Circuit Localization, Circuit Validation, Universality — the code only.
**Not today:** anything touching data. Dataset integration happens after teammate B's dataset is finished.
**Assumed done:** models load in TransformerLens, hooks work.

Every function takes a model, clean/corrupt token-ID tensors, and `(own_tok, tool_tok, measure_pos, post_output_positions)`. Where those come from is not today's concern; use a couple of synthetic tensors to exercise the code.

Model loading (`models/load.py`): bf16, `fold_ln=False`, `center_writing_weights=False`, `center_unembed=False`, `model.cfg.ungroup_grouped_query_attention = True` (Qwen2.5 is GQA; EAP-IG needs ungrouped heads).

```
tooltrust/
  models/load.py
  localize/delta.py       # metric
  localize/eap.py         # EAP-IG screen
  localize/patch.py       # activation patching + random controls
  localize/roles.py       # attention-pattern + DLA labeling
  validate/ablate.py      # necessity / self-repair / faithfulness
  universality/cka.py     # CKA + node overlap
  scripts/run_localize.py, run_validate.py, run_universality.py   # --model flag
```

---

## 1. Circuit Localization (4 h)

### 1a. Metric → `localize/delta.py` (45 min)

- `delta(logits, own_tok, tool_tok, measure_pos)` = `logits[measure_pos, tool_tok] − logits[measure_pos, own_tok]`. Differentiable (EAP-IG backprops through it).
- `delta_curve(logits, ..., post_output_positions)` — Δ at every post-output index.
- Δ is read on the **corrupt** run (tool ≠ own). The clean run is the patching source. Recovery ceiling `Δ_clean_ref` = corrupt run with all post-output activations patched from clean.
- Check: Δ flips sign when `own_tok` and `tool_tok` are swapped.

### 1b. EAP-IG screen → `localize/eap.py` (1.5 h)

Wrap `hannamw/EAP-IG`; `ioi.ipynb` (Llama-3 8B) is the closest template.

| Library | Ours |
|---|---|
| dataloader | yields `(clean_ids, corrupt_ids, label)`, `label = (tool_tok, own_tok, measure_pos)` |
| `metric` | `delta`, batched over the label |
| `Graph.from_model(model)` | node-level (`attribute_node.py`); edges are stretch |
| `attribute(..., method='EAP-IG-inputs', ig_steps=5)` | `ig_steps` as a flag |
| `graph.apply_topn(n)` | `n` as a flag, default 50 |

- Batch size 1 or bucket by length. fp32 backward if memory allows. `patching` intervention, not `mean-positional`.
- **Output:** `node_scores.csv` — `layer, type, index, score`, descending.
- Check: finite scores, every head/MLP exactly once, fixed seed reproduces the file.

### 1c. Activation patching + controls → `localize/patch.py` (1.25 h)

- `patch_node(model, clean_cache, corrupt_ids, node, positions) -> Δ`: hook overwrites `hook_z[:, :, head]` or `hook_mlp_out` with the cached clean activation at `positions`.
- `recovery = (Δ_patched − Δ_corrupt) / (Δ_clean_ref − Δ_corrupt)`.
- Per shortlisted node: (i) all positions, (ii) post-output only, (iii) reverse direction.
- `random_control(k, n_draws=20, seed)` → noise distribution; keep nodes above its 95th percentile.
- **Output:** `patch_results.csv` — `layer, type, index, rec_all, rec_postout, rec_reverse, above_noise`.
- Check: patching every node at post-output positions → recovery ≈ 1.0; patching nothing → 0.0.

### 1d. Role labeling → `localize/roles.py` (30 min)

- **Attention role:** mean attention from post-output queries to the `<output>…</output>` span vs the reasoning-answer span → `tool_reader` / `own_reader` / `neither`.
- **DLA sign:** head output at `measure_pos` · `(W_U[:, tool_tok] − W_U[:, own_tok])`, final-LN scale applied.
- **Depth:** early / mid / late third.
- **Output:** `circuit.json` — per node `{layer, type, index, eap_score, rec_all, rec_postout, rec_reverse, attn_role, dla_sign, depth}`. **Hand-off file for Interventions.**

---

## 2. Circuit Validation (2 h) → `validate/ablate.py`

Mean-ablation default (means over whatever inputs are passed, cached by hash); zero-ablation behind a flag.

| Function | Does | Output field |
|---|---|---|
| `ablate(model, inputs, nodes, mode)` | replace `nodes` with mean/zero at post-output positions; return Δ | — |
| `necessity(circuit, k_random=20, seed)` | Δ drop: circuit vs 20 random same-size sets | `necessity_gap`, `random_dist` |
| `self_repair(circuit, max_iters=3)` | after ablation, recompute DLA on non-circuit heads; add spikes; iterate | `repair_heads`, `n_iters`, `converged` |
| `faithfulness(circuit)` | mean-ablate everything except circuit; % Δ recovered (`evaluate_graph`) | `faithfulness` |

- Specificity needs agreement traces + benchmark accuracy → `NotImplementedError` stub with a TODO.
- **Output:** `validation.json`.
- Checks: ablating empty set = baseline Δ; ablating all nodes → Δ ≈ 0; `faithfulness(all_nodes) ≈ 1.0`.

---

## 3. Universality (1 h) → `universality/cka.py`

- `run_localize.py` / `run_validate.py` take `--model {ToRL,SimpleTIR,VerlTool}`; nothing else differs. If it does, fix it.
- `cka(model_a, model_b, inputs, layers) -> [L, L]`: linear CKA on `resid_post` at post-output positions, same inputs. Cache activations once per model.
- `node_overlap(circuit_a, circuit_b)`: Jaccard over `(layer, type, index)`. Cheap, and more telling than CKA for models sharing a base.
- **Output:** `cka_<A>_<B>.npy`, `overlap.json`.
- Checks: `cka(m, m)` diagonal = 1; self-overlap = 1.0.

---

## 4. End-of-day deliverables

| File | Produces | Check |
|---|---|---|
| `models/load.py` | 3 models, correct cfg flags | GQA ungrouped, no folding |
| `localize/delta.py` | Δ, Δ curve | sign flips on token swap |
| `localize/eap.py` → `node_scores.csv` | ranked nodes | complete, finite, reproducible |
| `localize/patch.py` → `patch_results.csv` | recovery + noise floor | patch-all ≈ 1, patch-none = 0 |
| `localize/roles.py` → `circuit.json` | role-labeled circuit | every node has all fields |
| `validate/ablate.py` → `validation.json` | necessity, self-repair, faithfulness | empty = baseline, all ≈ 1 |
| `universality/cka.py` → `cka_*.npy`, `overlap.json` | cross-model similarity | self-CKA diagonal = 1 |
| `scripts/run_*.py` | one CLI per stage | runs for all 3 models on synthetic inputs |

Nothing produced today is a result. It is a pipeline that runs.

---

## 5. After this day

**Review (you):**
- `patch.py` — recovery formula, position masks, control seeding. This is where plausible-looking wrong circuits come from.
- `ablate.py` — means are computed at the same positions they're applied to.
- `roles.py` — DLA applies the final-LN scale.

**Then:** once the dataset is finished, write the loader that feeds it into these functions, run `run_localize → run_validate → run_universality` per model, and implement specificity. Hand `circuit.json` to Interventions.

**Escalate if:** faithfulness <50%; no clean elbow in `node_scores.csv`; self-repair doesn't converge in 3 iterations; ungrouped-GQA + fp32 backward doesn't fit on one 80 GB card.
