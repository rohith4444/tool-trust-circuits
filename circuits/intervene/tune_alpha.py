"""Tiers 2 and 3: does the model change its ANSWER, and only where it should?

TIER 2 -- the answer, not the logit

Delta moving is necessary and not sufficient. It says the logit ordering changed; it does
not say the model writes Y, and at the perturbation sizes steering needs (alpha 144 is
79% of the residual norm) it could be emitting something unrelated that happens to beat
the own-answer token. So we generate and read what it actually wrote.

This is the evaluation, not a proxy for it. ATTC defines the Tool-Ignored rate as
overrides / CONFLICT cases -- the denominator is conflict cases, which is exactly what a
validation pair is. Every one starts as an override (the tool reported the truth, the
model wrote its own wrong answer), so:

    Tool-Ignored before = 100%   by construction
    Tool-Ignored after  = fraction that still does not write Y
    accuracy before     = 0%     by construction
    accuracy after      = fraction that writes Y

What it does NOT give is full-benchmark accuracy: that needs regenerating every problem
under HookedTransformer, ~8 hours a model, and it is not in the deadline. The benchmark
gain can be projected from the fix rate if specificity holds, but it is a projection and
the writeup should say so.

GENERATION

Greedy, one token at a time, no KV cache. With a cache each step processes only the newest
token, and a hook keyed to absolute prompt positions would fire during prefill and then
silently stop -- the intervention would shape the context and vanish exactly when the model
writes its answer. Full forwards are ~10 tokens x a few hundred traces, so the cost of
removing that ambiguity is minutes.

Positions are recomputed per step as everything from the output block onward, so newly
generated tokens are steered too.

TIER 3 -- the agreement control

Every pair is a conflict by construction: clean injects Y, and X != Y because the model was
wrong. So there is no no-conflict case in the corpus to compare against, and the plan's
original control (`tool_echoed` pairs) is not one -- that label describes the ORIGINAL
trace, and after injection those are as much a conflict as any other.

We build the missing case instead: inject X, the model's own answer, so the tool agrees
and there is nothing to arbitrate.

    tool says Y   conflict, tool right   -> intervention should push toward Y
    tool says X   agreement              -> intervention should change nothing

If Delta rises here too, the direction is not "trust the tool", it is "prefer Y" -- and
since Y is the ground truth that is circular, a fix that says "be correct" rather than
"believe the tool". This is the remaining way a working-looking result is uninteresting.
"""
from __future__ import annotations

import logging
import re

import torch

from ..localize.delta import delta, run_delta
from .common import dedupe, run_ids

log = logging.getLogger(__name__)

# The probe leaves the model mid-answer inside \boxed{, so the answer ends at the brace.
STOP = "}"
MAX_NEW = 12


def positions_from(pair, total_len: int) -> list[int]:
    """Post-output positions extended over whatever has been generated so far.

    pair.post_output is fixed at build time and covers the prompt only. During generation
    the sequence grows, and those new tokens are post-output too -- the model is writing
    its answer, which is the moment the intervention is supposed to act on.
    """
    return list(range(pair.output_span[1], total_len))


@torch.no_grad()
def generate(model, ids: torch.Tensor, pair, hooks_fn=None, max_new: int = MAX_NEW) -> str:
    """Greedy continuation, returning only the newly generated text.

    `hooks_fn(pair, positions)` returns forward hooks; it is re-called each step because
    the position list grows. Passing None generates the untouched baseline through the
    identical code path, so before/after differ by the intervention and nothing else.
    """
    cur = ids.to(model.cfg.device)
    if cur.dim() == 1:
        cur = cur[None]
    out: list[int] = []
    for _ in range(max_new):
        hooks = hooks_fn(pair, positions_from(pair, cur.shape[1])) if hooks_fn else []
        logits = model.run_with_hooks(cur, fwd_hooks=list(hooks))
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        text = model.tokenizer.decode(out)
        if STOP in text:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=cur.device)], dim=1)
    return model.tokenizer.decode(out)


def parse_answer(generated: str) -> str:
    """What the model wrote inside \\boxed{, given the prompt ends just after the brace."""
    return generated.split(STOP)[0].strip()


def matches(answer: str, target: str) -> bool:
    """Prefix-tolerant match: generation stops at the brace, so a long target may be cut.

    The forced prefix is already in the prompt and not in `generated`, so compare on what
    follows it -- callers pass the suffix of the target after that prefix.
    """
    a, t = answer.strip(), str(target).strip()
    if not a:
        return False
    return a == t or (len(a) < len(t) and t.startswith(a))


def targets(pair) -> tuple[str, str]:
    """(Y, X) with the forced prefix stripped.

    The prefix is already in the prompt and so is absent from the generated text; the
    comparison has to be on what follows it. Kept in one place because getting it wrong
    silently scores every prefixed pair as incorrect.
    """
    prefix = str(pair.meta.get("readout", {}).get("forced_prefix") or "")
    Y, X = str(pair.meta.get("Y", "")), str(pair.meta.get("X", ""))

    def strip(s: str) -> str:
        return s[len(prefix):] if prefix and s.startswith(prefix) else s
    return strip(Y), strip(X)


@torch.no_grad()
def baseline_answers(model, pairs, run: str = "clean", max_new: int = MAX_NEW) -> dict[str, str]:
    """What the model writes untouched, keyed by pair id.

    Independent of any intervention, so it is generated once and reused across an alpha
    sweep: half the work, and every alpha is then compared against a byte-identical
    baseline rather than a separately sampled one.
    """
    return {pair.id: parse_answer(generate(model, run_ids(pair, run), pair, None, max_new))
            for pair in dedupe(pairs, run)}


@torch.no_grad()
def answer_flip(model, pairs, hooks_fn, run: str = "clean", max_new: int = MAX_NEW,
                before_by_id: dict[str, str] | None = None) -> list[dict]:
    """Per pair: what the model writes before and after, and whether it is now correct.

    Targets are compared after stripping the forced prefix, which sits in the prompt
    rather than in the generated text.
    """
    rows = []
    for pair in dedupe(pairs, run):
        ids = run_ids(pair, run)
        Y, X = str(pair.meta.get("Y", "")), str(pair.meta.get("X", ""))
        tgt_Y, tgt_X = targets(pair)

        before = (before_by_id[pair.id] if before_by_id is not None
                  else parse_answer(generate(model, ids, pair, None, max_new)))
        after = parse_answer(generate(model, ids, pair, hooks_fn, max_new))
        rows.append(dict(
            id=pair.id, tool_case=pair.meta.get("tool_case"), X=X, Y=Y,
            answer_before=before, answer_after=after,
            correct_before=matches(before, tgt_Y), correct_after=matches(after, tgt_Y),
            own_before=matches(before, tgt_X), own_after=matches(after, tgt_X),
            changed=before != after,
        ))
    return rows


def summarise(rows: list[dict]) -> dict:
    """Tool-Ignored rate and accuracy, before and after.

    An override is any case that does not end up writing the tool's (correct) value, so
    the Tool-Ignored rate is 1 - accuracy on this population. On validation pairs
    accuracy_before is 0 by construction; if it is not, those pairs are not the failures
    they are labelled as and the relative reduction is measured against the wrong floor.
    """
    n = len(rows) or 1
    acc_b = sum(r["correct_before"] for r in rows) / n
    acc_a = sum(r["correct_after"] for r in rows) / n
    ti_b, ti_a = 1 - acc_b, 1 - acc_a
    return dict(
        n=len(rows),
        accuracy_before=acc_b, accuracy_after=acc_a,
        tool_ignored_before=ti_b, tool_ignored_after=ti_a,
        tool_ignored_abs_drop=ti_b - ti_a,
        tool_ignored_rel_drop=(ti_b - ti_a) / ti_b if ti_b else float("nan"),
        frac_changed=sum(r["changed"] for r in rows) / n,
        frac_still_own_after=sum(r["own_after"] for r in rows) / n,
    )


# --------------------------------------------------------------------------- tier 3

@torch.no_grad()
def delta_shift_for(model, pair, hooks_fn, run: str = "clean") -> float:
    """Delta shift on a conflict pair -- the reference Tier 3 compares against.

    Recomputed here rather than reused from the sweep so both sides of the specificity
    ratio come from the same hook path and the same positions convention.
    """
    ids = run_ids(pair, run)
    n = int(ids.shape[-1])
    before = run_delta(model, ids, pair)
    after = run_delta(model, ids, pair, hooks_fn(pair, positions_from(pair, n)))
    return after - before


def agreement_text(pair) -> str | None:
    """`clean_text` with the tool reporting X instead of Y: agreement, nothing to decide.

    Rebuilt from the stored text rather than the ids because X rarely tokenises to Y's
    length -- which is fine here, since Tier 3 only reads Delta and never patches across
    the two runs.
    """
    text = pair.clean_text
    X = str(pair.meta.get("X", ""))
    if "```output\n" not in text or not X:
        return None
    head, rest = text.rsplit("```output\n", 1)
    if "\n```" not in rest:
        return None
    _, tail = rest.split("\n```", 1)
    return f"{head}```output\n{X}\n```{tail}"


@torch.no_grad()
def agreement_delta(model, pairs, hooks_fn) -> list[dict]:
    """Delta before and after the intervention when the tool reports the model's own answer.

    The prompt still ends with the probe plus forced prefix, so the readout is the last
    position and `own_tok` / `tool_tok` are unchanged -- the same ruler, a different
    injected value.
    """
    rows = []
    for pair in dedupe(pairs, "clean"):
        text = agreement_text(pair)
        if text is None:
            continue
        ids = torch.tensor(model.tokenizer(text, add_special_tokens=False)["input_ids"],
                           device=model.cfg.device)[None]
        mp = ids.shape[1] - 1
        positions = positions_from(pair, ids.shape[1])

        before = delta(model(ids), pair.own_tok, pair.tool_tok, mp).item()
        after = delta(model.run_with_hooks(ids, fwd_hooks=list(hooks_fn(pair, positions))),
                      pair.own_tok, pair.tool_tok, mp).item()
        rows.append(dict(id=pair.id, delta_before=before, delta_after=after,
                         shift=after - before))
    return rows


def specificity(agree_rows: list[dict], conflict_shift: float, tol: float = 0.25) -> dict:
    """Is the intervention specific to conflicts?

    Compares the mean Delta shift on agreement cases against the shift it produces on
    real conflicts. A specific intervention moves conflicts and leaves agreement alone;
    one that moves both by the same amount is pushing toward Y unconditionally, which
    on this data means "be correct" rather than "trust the tool".
    """
    n = len(agree_rows) or 1
    shift = sum(r["shift"] for r in agree_rows) / n
    ratio = abs(shift) / abs(conflict_shift) if conflict_shift else float("nan")
    return dict(n=len(agree_rows), agreement_delta_shift=shift,
                conflict_delta_shift=conflict_shift, ratio=ratio,
                specific=bool(ratio < tol), tol=tol)
