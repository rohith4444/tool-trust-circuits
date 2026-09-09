"""Matched clean/corrupt pairs: the data contract between the dataset side and this pipeline.

The dataset side hands over *text segments*; everything token-level is computed here
with the model's own tokenizer, so per-model tokenization differences never leak into
the dataset files.

    text_clean   = prefix + tool_clean   + suffix + shared_prefix
    text_corrupt = prefix + tool_corrupt + suffix + shared_prefix

`prefix` ends with the tool-output opening delimiter, `suffix` starts with the closing
one and ends in `\\boxed{`. The metric is read at the last position:

    Δ = logit(tool_tok) − logit(own_tok)

where `own_tok` / `tool_tok` are the first tokens at which the model's own answer and
the *ruler* answer diverge. The ruler is the answer the positive logit belongs to. For the
Layer 2 dataset (docs/DATASET.md) that is the ground truth Y, so Δ = logit(Y) − logit(X)
with clean = tool says Y and corrupt = tool says Z; for the older prefix/suffix format it
is `tool_corrupt` itself. `shared_prefix` is the token-level common prefix of the own
answer and the ruler (Qwen splits digits: "18" vs "17" share "1"), so the last position's
logits predict the first divergent token.

A pair is only usable when the two injected values actually differ at that token:
see `build_pair` and docs/CIRCUIT.md for the drop rules.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import torch

log = logging.getLogger(__name__)

OUTPUT_OPEN = "```output\n"   # from configs.models.DELIMITERS / build_pairs.py:OUTPUT_BLOCK


@dataclass
class Pair:
    clean_ids: torch.Tensor          # [T]
    corrupt_ids: torch.Tensor        # [T], differs from clean_ids only inside output_span
    own_tok: int                     # first divergent token of the reasoning answer
    tool_tok: int                    # first divergent token of the ruler answer
    measure_pos: int                 # logits[measure_pos] predict the divergent token (= T-1)
    post_output: list[int]           # range(output_span[1], T): closing fence onward
    output_span: tuple[int, int]     # token span of the injected value, half-open
    own_spans: list[tuple[int, int]] # every occurrence of the reasoning answer in the prefix
    clean_text: str                  # the exact strings the ids came from (EAP-IG re-tokenizes these)
    corrupt_text: str
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.clean_ids.shape[0])

    @property
    def id(self) -> str:
        return str(self.meta.get("id", "?"))


def first_divergence(a: list[int], b: list[int]) -> int | None:
    """Index where two token lists first differ; None if one is a prefix of the other."""
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None


def _encode(tokenizer, text: str):
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]


def _token_span(offsets, c0: int, c1: int) -> tuple[int, int] | None:
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and e > c0 and s < c1]
    return (idx[0], idx[-1] + 1) if idx else None


def _find_all(hay: str, needle: str) -> list[tuple[int, int]]:
    # not inside a longer number or a decimal ("124." at sentence end is fine, "124.5" / "0124" are not)
    pat = re.compile(r"(?<!\d)(?<!\d\.)" + re.escape(needle) + r"(?!\d)(?!\.\d)")
    return [(m.start(), m.end()) for m in pat.finditer(hay)]


def build_pair(tokenizer, prefix: str, own_answer: str, tool_clean: str, tool_corrupt: str,
               suffix: str, meta: dict | None = None, max_tokens: int | None = None,
               ruler: str | None = None, expect_readout: dict | None = None) -> Pair | None:
    """Build a Pair from text segments. Returns None (with a debug log) when the pair is unusable.

    `ruler` is the answer whose first divergent token (vs `own_answer`) carries the positive
    logit; defaults to `tool_corrupt`. `expect_readout` = {tok_Y, tok_X, divergence_index}
    from the dataset file; when given, the recomputed readout must match it exactly (a
    mismatch means the wrong tokenizer is loaded for this file, so it raises).

    Skips: own/ruler token-prefix clash; the two injected values sharing the readout token
    (then Δ cannot differ between runs); the corrupt value backing the own answer at the
    readout; clean and corrupt values tokenizing to different lengths (keeps T equal;
    biases toward same-digit-count pairs, documented); tokens differing outside the value
    span; a `max_tokens` left-truncation that would cut the value span or every own-answer
    occurrence.
    """
    meta = dict(meta or {})
    ruler = tool_corrupt if ruler is None else ruler
    own_ids = tokenizer.encode(own_answer, add_special_tokens=False)
    ruler_ids = tokenizer.encode(ruler, add_special_tokens=False)
    k = first_divergence(own_ids, ruler_ids)
    if k is None:
        log.debug("%s: prefix clash %r vs %r", meta.get("id"), own_answer, ruler)
        return None
    own_tok, tool_tok = int(own_ids[k]), int(ruler_ids[k])
    if expect_readout is not None:
        got = dict(tok_Y=tool_tok, tok_X=own_tok, divergence_index=k)
        want = {f: int(expect_readout[f]) for f in got}
        if got != want:
            raise RuntimeError(f"{meta.get('id')}: readout recomputed with this tokenizer {got} != "
                               f"stored {want}; is the loaded model the one the pair file was built for?")

    # The readout only moves between runs if the two injected values differ at that token.
    val_c = tokenizer.encode(tool_clean, add_special_tokens=False)
    val_x = tokenizer.encode(tool_corrupt, add_special_tokens=False)
    if len(val_c) <= k or len(val_x) <= k or val_c[k] == val_x[k]:
        log.debug("%s: injected values %r / %r share the readout token at index %d",
                  meta.get("id"), tool_clean, tool_corrupt, k)
        return None
    if val_x[k] == own_tok:
        log.debug("%s: corrupt value %r backs the own answer %r at the readout", meta.get("id"), tool_corrupt, own_answer)
        return None

    shared = tokenizer.decode(own_ids[:k]) if k else ""
    text_c = prefix + tool_clean + suffix + shared
    text_x = prefix + tool_corrupt + suffix + shared
    ids_c, off_c = _encode(tokenizer, text_c)
    ids_x, off_x = _encode(tokenizer, text_x)
    if len(ids_c) != len(ids_x):
        log.debug("%s: clean/corrupt lengths differ (%d vs %d)", meta.get("id"), len(ids_c), len(ids_x))
        return None
    span_c = _token_span(off_c, len(prefix), len(prefix) + len(tool_clean))
    span_x = _token_span(off_x, len(prefix), len(prefix) + len(tool_corrupt))
    if span_c is None or span_x is None or span_c != span_x:
        log.debug("%s: value spans differ %s vs %s", meta.get("id"), span_c, span_x)
        return None
    t0, t1 = span_x
    if ids_c[:t0] != ids_x[:t0] or ids_c[t1:] != ids_x[t1:]:
        log.debug("%s: ids differ outside the value span", meta.get("id"))
        return None
    if k and ids_x[-k:] != own_ids[:k]:
        log.debug("%s: shared prefix does not survive tokenization", meta.get("id"))
        return None
    own_spans = [s for s in (_token_span(off_x, a, b) for a, b in _find_all(prefix, own_answer)) if s]

    cut = 0
    T = len(ids_x)
    if max_tokens and T > max_tokens:
        want = T - max_tokens
        if want >= t0:
            return None
        found = None
        for c in range(want, min(want + 16, t0)):
            c0 = off_x[c][0]
            nc, nx = text_c[c0:], text_x[c0:]
            if (tokenizer.encode(nx, add_special_tokens=False) == ids_x[c:]
                    and tokenizer.encode(nc, add_special_tokens=False) == ids_c[c:]):
                found = (c, nc, nx)
                break
        if found is None:
            log.debug("%s: no clean truncation point", meta.get("id"))
            return None
        cut, text_c, text_x = found
        ids_c, ids_x = ids_c[cut:], ids_x[cut:]
        t0, t1 = t0 - cut, t1 - cut
        kept = [(a - cut, b - cut) for a, b in own_spans if a >= cut]
        if own_spans and not kept:
            log.debug("%s: truncation cut every own-answer occurrence", meta.get("id"))
            return None
        own_spans = kept
        T = len(ids_x)

    meta.update(own_answer=own_answer, tool_clean=tool_clean, tool_corrupt=tool_corrupt, ruler=ruler,
                shared_prefix=shared, divergence_index=k, truncated_tokens=cut)
    return Pair(
        clean_ids=torch.tensor(ids_c, dtype=torch.long),
        corrupt_ids=torch.tensor(ids_x, dtype=torch.long),
        own_tok=own_tok, tool_tok=tool_tok,
        measure_pos=T - 1,
        post_output=list(range(t1, T)),
        output_span=(t0, t1),
        own_spans=own_spans,
        clean_text=text_c, corrupt_text=text_x,
        meta=meta,
    )


def build_pairs(tokenizer, examples: list[dict], max_tokens: int | None = None,
                model_name: str | None = None) -> list[Pair]:
    """examples: dicts with prefix, own_answer, tool_clean, tool_corrupt, suffix, meta, and optionally
    ruler and expect_readout. The stored readout is only checked when the example was built for
    `model_name` (its `meta["model"]`); other models re-derive it with their own tokenizer."""
    pairs = []
    for ex in examples:
        expect = ex.get("expect_readout") if model_name and ex.get("meta", {}).get("model") == model_name else None
        p = build_pair(tokenizer, ex["prefix"], ex["own_answer"], ex["tool_clean"], ex["tool_corrupt"],
                       ex["suffix"], meta=ex.get("meta"), max_tokens=max_tokens,
                       ruler=ex.get("ruler"), expect_readout=expect)
        if p is not None:
            pairs.append(p)
    log.info("built %d/%d pairs", len(pairs), len(examples))
    return pairs


PAIR_FILE_SCHEMA = """\
Expected pair file: jsonl, one matched pair per line, in one of two formats.

(1) Layer 2 dataset (scripts/build_pairs.py, docs/DATASET.md) -- detected by `clean_text` + `readout`:
  pair_id        str   unique row key
  model          str   torl | simpletir | demyagent (whose trace and tokenizer this is)
  X, Y, Z        str   own answer, ground truth, fabricated wrong tool value
  clean_text     str   full prompt with the last ```output block = Y, ending in "\\boxed{" + forced_prefix
  corrupted_text str   identical except the block = Z
  readout        obj   {tok_Y, tok_X, divergence_index, forced_prefix}
  tool_case      str   tool_right | tool_echoed | tool_other | no_answer  (-> label override | accept)
  The loader splits clean_text into prefix / suffix itself; the ruler is Y (Δ = logit(Y) − logit(X)).

(2) Segment format:
  problem_id    str   benchmark + index
  model         str   torl | simpletir | demyagent (whose trace this is)
  prefix        str   trace text up to and including the tool-output opening delimiter
                      (for the ATTC harness: through "\\n```output\\n")
  own_answer    str   the reasoning answer as it appears in the prefix
  tool_clean    str   correct tool output (the value inside the output block)
  tool_corrupt  str   plausible wrong tool output; also the ruler
  suffix        str   closing delimiter ("\\n```\\n") plus the continuation ending in "\\boxed{"
  ground_truth  str
  label         str   accept | override (Layer-1 label of the natural trace, if any)
Everything token-level (spans, measure position, divergent tokens) is derived here.
Full description: docs/CIRCUIT.md"""


SEGMENT_FIELDS = ("problem_id", "model", "prefix", "own_answer", "tool_clean", "tool_corrupt", "suffix")
DATASET_FIELDS = ("pair_id", "model", "X", "Y", "Z", "clean_text", "corrupted_text", "readout")
REQUIRED_FIELDS = SEGMENT_FIELDS  # kept for callers that import it

# The circuit code talks about accept (model repeated the tool) vs override (model went its own
# way). The dataset records what the *original* tool said instead; this is the translation.
TOOL_CASE_LABEL = {"tool_echoed": "accept", "tool_right": "override", "tool_other": "override", "no_answer": None}


def _example_from_dataset_row(row: dict, where: str) -> dict:
    """Split a docs/DATASET.md record into the segments `build_pair` wants.

    clean_text = prefix + Y + suffix + forced_prefix, with prefix ending at the last
    ```output opening fence. forced_prefix is stripped because build_pair re-derives and
    re-appends the shared prefix itself.
    """
    ct, xt = row["clean_text"], row["corrupted_text"]
    X, Y, Z = str(row["X"]), str(row["Y"]), str(row["Z"])
    fp = str(row["readout"].get("forced_prefix") or "")
    i = ct.rfind(OUTPUT_OPEN)
    if i < 0:
        raise ValueError(f"{where}: no {OUTPUT_OPEN!r} in clean_text")
    cut = i + len(OUTPUT_OPEN)
    prefix, rest = ct[:cut], ct[cut:]
    if not rest.startswith(Y) or not rest.endswith(fp):
        raise ValueError(f"{where}: clean_text does not continue with Y={Y!r} and end with forced_prefix={fp!r}")
    suffix = rest[len(Y): len(rest) - len(fp)]
    if prefix + Y + suffix + fp != ct or prefix + Z + suffix + fp != xt:
        raise ValueError(f"{where}: clean/corrupted texts do not decompose as prefix + value + suffix + forced_prefix")
    meta = {k: v for k, v in row.items() if k not in ("clean_text", "corrupted_text")}
    meta["id"] = str(row["pair_id"])
    meta["label"] = TOOL_CASE_LABEL.get(row.get("tool_case"))
    return dict(prefix=prefix, own_answer=X, tool_clean=Y, tool_corrupt=Z, suffix=suffix,
                ruler=Y, expect_readout=row["readout"], meta=meta)


def _example_from_segment_row(row: dict) -> dict:
    meta = {k: v for k, v in row.items() if k not in SEGMENT_FIELDS}
    meta["id"] = str(row["problem_id"])
    meta["model"] = row["model"]
    return dict(prefix=row["prefix"], own_answer=str(row["own_answer"]), tool_clean=str(row["tool_clean"]),
                tool_corrupt=str(row["tool_corrupt"]), suffix=row["suffix"], meta=meta)


def load_examples(path: str, label: str | None = None) -> list[dict]:
    """Read a pair file (jsonl, either format in PAIR_FILE_SCHEMA) into example dicts for `build_pairs`."""
    import json
    rows = []
    n_dataset = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            where = f"{path} line {i + 1}"
            if "clean_text" in row and "readout" in row:
                missing = [k for k in DATASET_FIELDS if k not in row]
                if missing:
                    raise ValueError(f"{where}: missing fields {missing}\n{PAIR_FILE_SCHEMA}")
                ex = _example_from_dataset_row(row, where)
                n_dataset += 1
            else:
                missing = [k for k in SEGMENT_FIELDS if k not in row]
                if missing:
                    raise ValueError(f"{where}: missing fields {missing}\n{PAIR_FILE_SCHEMA}")
                ex = _example_from_segment_row(row)
            if label is not None and ex["meta"].get("label") != label:
                continue
            rows.append(ex)
    log.info("read %d examples from %s (%s format)%s", len(rows), path,
             "dataset" if n_dataset else "segment", f" (label={label})" if label else "")
    return rows


def load_pairs(path: str, tokenizer, label: str | None = None, max_tokens: int | None = None,
               model_name: str | None = None) -> list[Pair]:
    return build_pairs(tokenizer, load_examples(path, label), max_tokens=max_tokens, model_name=model_name)