"""Matched clean/corrupt pairs: the data contract between the dataset side and this pipeline.

The dataset side hands over *text segments*; everything token-level is computed here
with the model's own tokenizer, so per-model tokenization differences never leak into
the dataset files.

    text_clean   = prefix + tool_clean   + suffix + shared_prefix
    text_corrupt = prefix + tool_corrupt + suffix + shared_prefix

`prefix` ends with the tool-output opening delimiter, `suffix` starts with the closing
one and ends in `\\boxed{`. `shared_prefix` is the token-level common prefix of the
own answer and the corrupt tool answer (Qwen splits digits: "18" vs "17" share "1"),
so the last position's logits predict the first divergent token.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import torch

log = logging.getLogger(__name__)


@dataclass
class Pair:
    clean_ids: torch.Tensor          # [T]
    corrupt_ids: torch.Tensor        # [T], differs from clean_ids only inside output_span
    own_tok: int                     # first divergent token of the reasoning answer
    tool_tok: int                    # first divergent token of the corrupt tool answer
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
               suffix: str, meta: dict | None = None, max_tokens: int | None = None) -> Pair | None:
    """Build a Pair from text segments. Returns None (with a debug log) when the pair is unusable.

    Skips: own/tool token-prefix clash; clean and corrupt values tokenizing to different
    lengths (keeps T equal; biases toward same-digit-count pairs, documented); tokens
    differing outside the value span; a `max_tokens` left-truncation that would cut the
    value span or every own-answer occurrence.
    """
    meta = dict(meta or {})
    own_ids = tokenizer.encode(own_answer, add_special_tokens=False)
    tool_ids = tokenizer.encode(tool_corrupt, add_special_tokens=False)
    k = first_divergence(own_ids, tool_ids)
    if k is None:
        log.debug("%s: prefix clash %r vs %r", meta.get("id"), own_answer, tool_corrupt)
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

    meta.update(own_answer=own_answer, tool_clean=tool_clean, tool_corrupt=tool_corrupt,
                shared_prefix=shared, divergence_index=k, truncated_tokens=cut)
    return Pair(
        clean_ids=torch.tensor(ids_c, dtype=torch.long),
        corrupt_ids=torch.tensor(ids_x, dtype=torch.long),
        own_tok=int(own_ids[k]), tool_tok=int(tool_ids[k]),
        measure_pos=T - 1,
        post_output=list(range(t1, T)),
        output_span=(t0, t1),
        own_spans=own_spans,
        clean_text=text_c, corrupt_text=text_x,
        meta=meta,
    )


def build_pairs(tokenizer, examples: list[dict], max_tokens: int | None = None) -> list[Pair]:
    """examples: dicts with prefix, own_answer, tool_clean, tool_corrupt, suffix, meta."""
    pairs = []
    for ex in examples:
        p = build_pair(tokenizer, ex["prefix"], ex["own_answer"], ex["tool_clean"], ex["tool_corrupt"],
                       ex["suffix"], meta=ex.get("meta"), max_tokens=max_tokens)
        if p is not None:
            pairs.append(p)
    log.info("built %d/%d pairs", len(pairs), len(examples))
    return pairs


PAIR_FILE_SCHEMA = """\
Expected pair file: jsonl, one matched pair per line, fields
  problem_id    str   benchmark + index
  model         str   torl | simpletir | demyagent (whose trace this is)
  prefix        str   trace text up to and including the tool-output opening delimiter
                      (for the ATTC harness: through "\\n```output\\n")
  own_answer    str   the reasoning answer as it appears in the prefix
  tool_clean    str   correct tool output (the value inside the output block)
  tool_corrupt  str   plausible wrong tool output
  suffix        str   closing delimiter ("\\n```\\n") plus the continuation ending in "\\boxed{"
                      (canonical, or the corrupt trace's real continuation; same for both runs)
  ground_truth  str
  label         str   accept | override (Layer-1 label of the natural trace, if any)
Everything token-level (spans, measure position, divergent tokens) is derived here.
Full description: docs/pair_dataset.md"""


REQUIRED_FIELDS = ("problem_id", "model", "prefix", "own_answer", "tool_clean", "tool_corrupt", "suffix")


def load_examples(path: str, label: str | None = None) -> list[dict]:
    """Read the pair file (jsonl, PAIR_FILE_SCHEMA) into example dicts for `build_pairs`."""
    import json
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED_FIELDS if k not in row]
            if missing:
                raise ValueError(f"{path} line {i + 1}: missing fields {missing}\n{PAIR_FILE_SCHEMA}")
            if label is not None and row.get("label") != label:
                continue
            meta = {k: v for k, v in row.items() if k not in REQUIRED_FIELDS}
            meta["id"] = str(row["problem_id"])
            rows.append(dict(prefix=row["prefix"], own_answer=str(row["own_answer"]), tool_clean=str(row["tool_clean"]),
                             tool_corrupt=str(row["tool_corrupt"]), suffix=row["suffix"], meta=meta))
    log.info("read %d examples from %s%s", len(rows), path, f" (label={label})" if label else "")
    return rows


def load_pairs(path: str, tokenizer, label: str | None = None, max_tokens: int | None = None) -> list[Pair]:
    return build_pairs(tokenizer, load_examples(path, label), max_tokens=max_tokens)
