# tool-trust-circuits

Circuit analysis of tool trust in tool-integrated reasoning (TIR) models.

Localize and causally validate the internal circuit that decides whether to
**accept** a returned tool output or **override** it with the model's own
reasoning, then derive a training-free steering fix.

Baseline: ATTC (Xu et al., [arXiv:2604.08281](https://arxiv.org/abs/2604.08281)).

## Status

Setup. Nothing generated yet.

## Environments

Three, but only one is buildable off-GPU:

| Env | Contents | Where |
|-----|----------|-------|
| `tools` | `huggingface-hub`, `transformers`, `datasets` | laptop |
| `gen` | `vllm`, `math_verify` | RunPod (CUDA only) |
| `analysis` | `transformer-lens`, `torch` | RunPod |

`gen` and `analysis` are built on the pod, where they can actually be tested.
They are kept separate because vLLM and TransformerLens pin conflicting
`torch`/`transformers` versions.

Requires [uv](https://docs.astral.sh/uv/) (`brew install uv`).

```sh
uv venv .venv-tools
uv pip install --python .venv-tools -r envs/tools.txt
```

## Step 0 — verify models before provisioning any GPU

```sh
uv run scripts/verify_models.py
```

Declares its own dependencies inline (PEP 723), so uv builds the environment
automatically. Reads config and tokenizer metadata from the HF API — downloads
no weights, needs no GPU, takes about a minute.

It reports, per model: whether the repo resolves and is gated, total weight size
(to size the network volume), architecture and layer/head counts (for
TransformerLens support and CKA depth mapping), how numbers tokenize (which sets
where Δ is read), and whether the chat template documents the tool-call
delimiters.

Two things it is specifically meant to settle:

- **DemyAgent-4B's tool-call delimiters** are undocumented in the ATTC appendix,
  its model card, and its repo README. If the chat template does not reveal
  them, the cross-family universality model is at risk — and there is no
  official ReTool-4B to fall back on.
- **SimpleTIR-7B's repo ID** has not been resolved from the paper or GitHub, so
  the script searches HF for candidates.

## Storage

| Artifact | Home |
|----------|------|
| Model weights (~53 GB) | HF → RunPod volume cache. Never stored by us. |
| Raw traces (~150 MB) | RunPod volume → private HF dataset repo |
| Pair files, labels, results, figures | git |

RunPod pods are ephemeral; a **network volume** carries data between the
generation pod (3×48 GB) and the analysis pod (1×80 GB).

## Branches

Short-lived, one per stage, merged as each lands — not long-lived personal
branches. `configs/models.py` is shared by everything downstream, so ownership
of it should be agreed before parallel work starts.
