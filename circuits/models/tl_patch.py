"""One-line memory fix for TransformerLens 2.18's per-head attention output.

With `use_attn_result=True` (which `eap.attribute_node` asserts), `AbstractAttention.forward`
computes the per-head output through a [batch, pos, head, d_head, d_model] intermediate:

    unhooked_result = (z * w).sum(-2)        # components/abstract_attention.py

For ToRL that is 28 * 128 * 3584 * 2 bytes = 24 MB per token per layer per pass -- 5.9 GiB at
247 tokens, 47 GiB at 2000 -- and the backward materialises it again. An einsum gives the
identical [batch, pos, head, d_model] result as a batched matmul with no expanded tensor.

`apply()` rewrites that single statement in the installed forward and asserts the original
text is present exactly once, so a TransformerLens upgrade fails here instead of silently
running the old code. It is idempotent; `load_model` calls it.
"""
import inspect
import textwrap

from transformer_lens.components import abstract_attention as _aa

OLD = "unhooked_result = (z * w).sum(-2)"
# z: [batch pos head d_head 1], w: [1 1 head d_head d_model] at this point in the source.
NEW = 'unhooked_result = torch.einsum("bphd,hdm->bphm", z.squeeze(-1), w.squeeze(0).squeeze(0))'


def apply() -> None:
    fwd = _aa.AbstractAttention.forward
    if getattr(fwd, "_tt_patched", False):
        return
    src = textwrap.dedent(inspect.getsource(fwd))
    if src.count(OLD) != 1:
        raise RuntimeError(f"transformer_lens attention source changed; expected exactly one {OLD!r}")
    ns = dict(vars(_aa))
    exec(compile(src.replace(OLD, NEW), _aa.__file__, "exec"), ns)
    new = ns["forward"]
    new._tt_patched = True
    _aa.AbstractAttention.forward = new
