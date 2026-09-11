"""Model loading: a TransformerLens HookedTransformer built from the fine-tune's own HF snapshot.

Why not `HookedTransformer.from_pretrained`: it resolves `model_name` through
`get_official_model_name`, which raises for anything off its list (GAIR/ToRL-7B,
Qwen2.5-Math-7B, DemyAgent-4B are all absent). The usual workaround,
`from_pretrained("Qwen/Qwen2.5-7B", hf_model=torl)`, reads the *official* model's config:
rope_theta=1e6 against weights trained at 1e4 (Qwen2.5-Math lineage) -- silently wrong.
The loader functions below accept a local directory and read its config.json.

Flags per the onboarding plan: bf16 on GPU, fold_ln=False, center_writing_weights=False,
center_unembed=False. `ungroup_grouped_query_attention` is only needed by EAP-IG and is
toggled by `tooltrust.hooks.eap_mode`.
"""
from __future__ import annotations

import gc
import logging

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens import loading_from_pretrained as loading

from configs.models import MAX_CTX, MODELS
from circuits.models import tl_patch

log = logging.getLogger(__name__)

IGNORE = ["*.md", "*.gguf", "original/*", "*.pth", "*.h5", "*.msgpack", "*.ot"]


def resolve_repo(name: str) -> str:
    return MODELS[name]["repo"] if name in MODELS else name


def snapshot(name: str) -> str:
    return snapshot_download(resolve_repo(name), ignore_patterns=IGNORE)


def _hf_rope_theta(hf_cfg) -> float | None:
    if getattr(hf_cfg, "rope_theta", None) is not None:
        return float(hf_cfg.rope_theta)
    rp = getattr(hf_cfg, "rope_parameters", None)
    if isinstance(rp, dict) and rp.get("rope_theta") is not None:
        return float(rp["rope_theta"])
    return None


def load_model(name: str, device: str | None = None, dtype=None, n_ctx: int | None = None) -> HookedTransformer:
    """Load a registry name (torl, simpletir, demyagent, base) or a raw HF repo id."""
    tl_patch.apply()
    repo = resolve_repo(name)
    path = snapshot(name)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    hf_cfg = AutoConfig.from_pretrained(path)
    if n_ctx is None:
        n_ctx = min(int(getattr(hf_cfg, "max_position_embeddings", MAX_CTX)), MAX_CTX)

    # n_ctx is passed explicitly: TransformerLens 2.x hardcodes 2048 for Qwen2 and Qwen3.
    cfg = loading.get_pretrained_model_config(
        path, dtype=dtype, device=device, n_ctx=n_ctx, default_prepend_bos=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(path)
    state_dict = loading.get_pretrained_state_dict(path, cfg, dtype=dtype)
    model = HookedTransformer(cfg, tokenizer=tokenizer, move_to_device=False)
    model.load_and_process_state_dict(
        state_dict, fold_ln=False, center_writing_weights=False, center_unembed=False,
        fold_value_biases=False, refactor_factored_attn_matrices=False,
    )
    del state_dict
    gc.collect()
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.tt_name, model.tt_repo, model.tt_path = name, repo, path

    rope = _hf_rope_theta(hf_cfg)
    if rope is not None and int(rope) != int(model.cfg.rotary_base):
        raise RuntimeError(f"rotary_base {model.cfg.rotary_base} != HF rope_theta {rope} for {repo}")
    if model.cfg.n_ctx != n_ctx:
        raise RuntimeError(f"n_ctx {model.cfg.n_ctx} != requested {n_ctx}")
    log.info("loaded %s (%s): %d layers, %d heads, kv=%s, d_model=%d, n_ctx=%d, rope=%s, dtype=%s, device=%s",
             name, repo, cfg.n_layers, cfg.n_heads, cfg.n_key_value_heads, cfg.d_model, cfg.n_ctx,
             cfg.rotary_base, dtype, device)
    return model


@torch.no_grad()
def verify(model: HookedTransformer, text: str = "The answer is \\boxed{4", long_len: int | None = None) -> dict:
    """Feasibility gate: TL logits match HF logits, and a forward at long context works.

    Loads the HF model alongside (same dtype/device) and frees it afterwards. bf16
    tolerance is loose because TL does not upcast the attention softmax.
    """
    device, dtype = model.cfg.device, model.cfg.dtype
    ids = model.tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    tl_logits = model(ids)[0, -1].float()
    hf = AutoModelForCausalLM.from_pretrained(model.tt_path, torch_dtype=dtype).to(device).eval()
    hf_logits = hf(ids).logits[0, -1].float()
    del hf
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    diff = (tl_logits - hf_logits).abs().max().item()
    top1 = bool(tl_logits.argmax() == hf_logits.argmax())
    tol = 1e-3 if dtype == torch.float32 else 0.5

    if long_len is None:
        long_len = min(model.cfg.n_ctx, 4096)
    long_ids = torch.randint(0, model.cfg.d_vocab, (1, long_len), device=device)
    model(long_ids)

    out = dict(model=model.tt_name, max_abs_logit_diff=diff, top1_equal=top1, tol=tol,
               n_ctx=model.cfg.n_ctx, long_forward_len=long_len, rotary_base=model.cfg.rotary_base,
               dtype=str(dtype), ok=bool(top1 and diff < tol))
    log.info("verify %s: %s", model.tt_name, out)
    return out
