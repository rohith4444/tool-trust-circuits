# Attribution

The code in this directory is vendored from the ATTC repository and is **not our
work**. It is the evaluation harness that produced the results in:

> Xu, Ji, Luo, Li, Li, Li, Li & Zhang.
> *When to Trust Tools? Adaptive Tool Trust Calibration For Tool-Integrated Math
> Reasoning.* arXiv:2604.08281

- **Source:** https://github.com/00Dreamer00/ATTC (via fork `rohith4444/ATTC`)
- **Commit:** `3258c40bde8020958bbd66f5a5067d66bab0ae52`
- **Copied:** 2026-09-06

ATTC's harness is itself derived from the Qwen2.5-Math / ToRA evaluation
pipeline, as its own file headers note.

## Why vendored rather than cloned

Two reasons. The intervention (Contribution 2) modifies the generation loop to
apply steering hooks, so both arms of the before/after comparison must come from
the same codebase -- a patch maintained against an external clone would risk the
baseline and the intervention diverging. And a self-contained repo means one
clone reproduces the whole pipeline.

## What we copied

| Path | Purpose |
|---|---|
| `math_eval.py` | the TIR loop: generate, stop at the tool marker, execute, reinsert, repeat |
| `python_executor.py` | sandbox that runs model-written code |
| `data_loader.py` | benchmark loading and schema normalisation |
| `parser.py` | answer extraction from raw generations |
| `grader.py` | symbolic/numeric answer checking against ground truth |
| `evaluate.py`, `trajectory.py`, `utils.py`, `model_utils.py` | supporting |
| `judge/analyze_tool_ignore.py` | Appendix B Tool-Ignored labelling |
| `judge/analyze_trust_or_not.py` | Appendix B contradiction labelling |

Deliberately **not** copied: `octotools/` (agent-tools library, unreachable
after the modifications below), `search_executor.py` (search-engine TIR, we are
code-only), the SGLang and confidence-method variants, `case_analysis/`, and the
13 benchmarks we do not evaluate on.

`data/` holds their released files for our six benchmarks, unmodified. We use
theirs rather than fetching from source so that our reproduction gate compares
against the same records they scored -- these benchmarks have several public
variants with different subsets and answer normalisation.

## Our modifications

Four. The first two remove dead imports -- neither changes behaviour on any
benchmark we evaluate, and together they mean `octotools` is never imported. The
third adds an optional argument that defaults to upstream behaviour; the fourth
restores a timestamped filename they had written but disabled.

### 1. `grader.py` -- commented out the `ChatOpenAI` import and instantiation

```python
# from octotools.engine.openai import ChatOpenAI
# llm_engine = ChatOpenAI(model="gpt-4o", enable_cache=True)
```

`llm_engine` is referenced only inside `model_eval()`, which `math_equal_process()`
calls only when `data_name == 'gpqa_diamond_mcq'` -- a multiple-choice science
benchmark not in our set. The module-level instantiation made it an import-time
dependency on `octotools`, which pulls a large transitive tree.

`math_equal()`, the function we actually use, is pure sympy and never touches it.
If anything did reach `model_eval()` it would now raise `NameError` rather than
grade silently wrong -- the safe failure direction.

### 2. `math_eval.py` -- made the `PerplexitySearch` import lazy

```python
elif "search-r1" in args.prompt_type or "octo-sci" in args.prompt_type:
    from search_executor import PerplexitySearch  # lazy: search TIR only
    executor = PerplexitySearch()
```

`PerplexitySearch` is constructed only for the `search-r1` and `octo-sci` prompt
types. Ours are `torl`, `simpleTIR`, and `DemyAgent`, which take the
`PythonExecutor` branch. Moving the import to the call site preserves search TIR
for anyone who wants it while keeping `octotools` off our import path.

### 3. `math_eval.py` -- added an optional `--max_model_len`

Upstream never sets this, so vLLM initialises with whatever each model's config
declares. That is 4096 for ToRL, but **131072 for SimpleTIR and 262144 for
DemyAgent** -- and vLLM sizes its KV cache to the declared context, which
collapses batching:

| model | declared context | max concurrency |
|---|---|---|
| ToRL | 4,096 | 262.00x |
| SimpleTIR | 131,072 | 8.18x |
| DemyAgent | 262,144 | 1.79x |

A ~150x throughput gap on identical hardware. We cap SimpleTIR and DemyAgent at
16,384, which leaves headroom for `max_tokens_per_call` (3072) x `max_func_call`
(4) plus the prompt -- roughly 13.5k worst case.

The argument defaults to `None`, in which case the `LLM(...)` call is upstream's
verbatim. This is a throughput change only; it does not alter what the model
sees, since no prompt approaches even 8k tokens.

### 4. `math_eval.py` -- re-enabled the output-filename timestamp

Upstream computes `dt_string` but writes a deterministic filename, having
commented out the timestamped variant. The effect is that re-running a config
(all our runs pass `--overwrite`) silently replaces the previous result, so a
run that later turns out to be wrong cannot be compared against its replacement.

We restored their timestamped line and widened the format from `%m-%d_%H-%M` to
`%m-%d_%H-%M-%S`, since minute resolution still collides for two runs started in
the same minute.

```
test_torl_-1_seed0_t0.0_p1_s0_e-1_09-07_04-15-32.jsonl
```

Safe for their resume logic: `out_file_prefix` stops before start/end/timestamp,
and processed samples are collected via `startswith(out_file_prefix)`, so
timestamped files are still found and merged.

**This is the one modification that changes output rather than imports or
throughput**, and it imposes a rule on everything downstream: tools must select
the **most recent** file per (model, benchmark, config). Without that rule the
timestamp merely trades "which limit?" for "which timestamp?" -- and a timestamp
says nothing about which run is authoritative. `scripts/summarize_runs.py`
implements this in `latest_runs()`.

## Runtime notes

Their `requirements.txt` pins `vllm==0.8.3`; we run **0.28.0**. The API surface
they use (`LLM(...)`, `llm.generate(prompts, SamplingParams(...))`) is unchanged
across that gap, and a 20-problem run reproduced end to end.

`latex2sympy2` hard-pins `antlr4-python3-runtime==4.11.1`, which downgrades from
the 4.13.2 that `math-verify` wants. We dropped `math-verify` -- their
`grader.py` is what we grade with, so it was redundant.

`PythonExecutor` opens a process pool sized to `cpu_count()`, and a second pool
is created during `evaluate()`. On a 32-vCPU host this exceeds the default file
descriptor limit. Run scripts must set `ulimit -n 65536`.
