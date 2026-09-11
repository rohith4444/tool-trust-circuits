"""Re-run the TIR loop with an intervention active, producing trace-shaped output.

This is the end-task evaluation. ATTC's own fix (`EffectiveCIR`) is a reworded prompt, so
they re-ran `math_eval.py` with one flag changed and vLLM did the rest. Ours edits
activations, which vLLM cannot host -- so the loop has to be rebuilt around
HookedTransformer. Everything else is kept identical to `math_eval.py` on purpose: same
prompt, same stop rule, same sandbox, same injection format, same call limit, temperature
0. The before/after difference should be the intervention and nothing else.

WHERE THE INTERVENTION ENTERS

Only at post-output positions. Before the first ```output block exists no position
qualifies, so the model writes its first reasoning and its first code exactly as the
baseline did -- at temperature 0 that is deterministic, so the sandbox returns the same
result and the trace is identical up to that point. From the first closed output block
onward the hook applies to every position, including tokens generated afterwards.

That is also why resuming from an existing trace is not a shortcut: for a single-tool-call
trace it produces what full regeneration would.

STOP RULE, COPIED EXACTLY

    output = output.rstrip()                                  # math_eval.py:262
    elif ("boxed" not in output and output.endswith("```")):  # line 291 -> tool call

vLLM excludes a stop string from its output, so the text ends just *before* "```output",
i.e. at the code block's closing fence. Generating token by token we overshoot and have to
truncate back, or the check never fires and every trace runs to the token cap.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger(__name__)

# math_eval.py:183 -- the TIR stop words. "```output" is what ends a tool-calling turn.
STOP_STRINGS = ("</s>", "<|im_end|>", "<|endoftext|>", "\n\n---", "```output")

MAX_FUNC_CALL = 4
MAX_TOKENS_PER_CALL = 3072


def build_prompt(question: str, prompt_type: str) -> str:
    """Inlined from generation/utils.py::construct_prompt.

    Copied rather than imported because math_eval's module graph pulls in vLLM, which the
    analysis environment deliberately does not have -- the two pin conflicting torch
    versions, which is why `gen` and `analysis` are separate.
    """
    if prompt_type == "torl":
        return ("A conversation between User and Assistant. The user asks a question, and "
                "the Assistant solves it.\nUser: Please integrate natural language reasoning "
                "with programs to solve the problem above, and put your final answer within "
                f"\\boxed{{}}.\n{question}\nAssistant:")
    if prompt_type == "simpleTIR":
        return ("Solve the following problem step by step. You now have the ability to "
                "selectively write executable Python code to enhance your reasoning process. "
                "The Python code will be executed by an external sandbox, and the output "
                "(after '\\n```output\\n') is returned to aid your reasoning and help you "
                "arrive at the final answer. The Python code should be complete scripts, "
                "including necessary imports.\nCode Format:\n Each code snippet is wrapped "
                "between \\n```python \\n```.\\n You need to use `print()` to output "
                "intermediate results.\nAnswer Format:\nYou should use \\boxed to return your "
                "final answer. The last part of your response should be:\n\\boxed{'The final "
                f"answer goes here.'}}\nUser Question:\n{question}")
    if prompt_type == "DemyAgent":
        return ("Please solve the following problem step by step. During your reasoning "
                "process, if needed, you can choose to write python code to enhance your "
                "reasoning.You need to use `print()` to output intermediate results.The code "
                "executor will run your code and provide the execution results back to you to "
                "support your reasoning process.\nA Python code block and its execution result "
                "strictly follow the following format:\n```python\n#your code here\n```\n"
                "```output\nexecution result here\n```\n Please put the final answer within "
                f"\\boxed{{}}.\nQuestion:\n{question}")
    raise NotImplementedError(f"prompt_type {prompt_type!r}")


def truncate_at_stop(text: str) -> tuple[str, str | None]:
    """Cut `text` at the earliest stop string. Returns (text, which_stop_or_None)."""
    best, which = len(text), None
    for s in STOP_STRINGS:
        i = text.find(s)
        if 0 <= i < best:
            best, which = i, s
    return (text[:best], which) if which else (text, None)


@torch.no_grad()
def generate_turn(model, ids: torch.Tensor, hooks_fn, max_new: int,
                  use_cache: bool = True) -> tuple[str, str | None]:
    """One turn: greedy tokens until a stop string, EOS, or `max_new`.

    `hooks_fn(offset, n_tokens)` returns hooks for a forward covering absolute positions
    [offset, offset + n_tokens). Returns the turn's text with the stop string removed, and
    which stop fired.

    WHY THE CACHE, AND WHAT IT COSTS TO GET RIGHT

    Without one, every step re-runs the whole sequence: generating 500 tokens after a
    1,500-token prompt is ~875,000 token-passes instead of ~2,000, and each step allocates
    logits for EVERY position ([1, T, 152064] is ~600 MB at T=2000) only to read the last
    row. That fragments an 80 GB card into an OOM, which is exactly what it did.

    The catch is that with a cache a forward contains only the tokens not yet cached, so
    index 0 of the tensor is not position 0 of the sequence. A hook keyed to absolute
    positions would quietly stop firing -- the intervention would shape the context and
    then vanish while the model writes its answer, with no error to notice.

    So the offset is tracked explicitly and passed to `hooks_fn`, which maps absolute
    positions to local ones. Inferring it from the tensor's shape instead ("one token means
    a generation step") is nearly right and fails silently when an injected block happens
    to tokenise to a single token.

    `use_cache=False` runs the naive path, for checking the two agree.

    The decoded text is re-scanned every step rather than matching token ids: a stop string
    can straddle a token boundary, and "```output" does on Qwen.
    """
    cur = ids if ids.dim() == 2 else ids[None]
    start = cur.shape[1]
    eos = model.tokenizer.eos_token_id
    device = cur.device
    n_ctx = model.cfg.n_ctx

    if not use_cache:
        for _ in range(max_new):
            if cur.shape[1] >= n_ctx:
                break
            logits = model.run_with_hooks(cur, fwd_hooks=list(hooks_fn(0, cur.shape[1])))
            nxt = int(logits[0, -1].argmax())
            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            if nxt == eos:
                break
            text, which = truncate_at_stop(model.tokenizer.decode(cur[0, start:]))
            if which is not None:
                return text, which
        return model.tokenizer.decode(cur[0, start:]), None

    from transformer_lens import HookedTransformerKeyValueCache
    cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, cur.shape[0])

    fed = 0                                   # absolute position of the next unfed token
    step_in = cur                             # first forward is the whole prompt
    out_ids: list[int] = []
    for _ in range(max_new):
        if fed + step_in.shape[1] >= n_ctx:
            break
        logits = model.run_with_hooks(
            step_in, fwd_hooks=list(hooks_fn(fed, step_in.shape[1])),
            past_kv_cache=cache)
        fed += step_in.shape[1]
        nxt = int(logits[0, -1].argmax())
        out_ids.append(nxt)
        if nxt == eos:
            break
        text, which = truncate_at_stop(model.tokenizer.decode(out_ids))
        if which is not None:
            return text, which
        step_in = torch.tensor([[nxt]], device=device)     # only the new token from here
    return model.tokenizer.decode(out_ids), None


@torch.no_grad()
def run_one(model, executor, extract_program, question: str, prompt_type: str, hooks_at,
            max_func_call: int = MAX_FUNC_CALL,
            max_tokens_per_call: int = MAX_TOKENS_PER_CALL,
            use_cache: bool = True) -> dict:
    """The TIR loop for one problem. Returns the generated trace and some bookkeeping.

    `hooks_at(cut, offset, n_tokens)` builds the intervention's hooks, or nothing while
    `cut` is None. The returned `code` is the generated text only -- not the prompt --
    which is what the trace files store and what the scorers parse.

    Each turn re-tokenises `prompt + query` from scratch and starts a fresh cache, so a
    turn's own indexing and absolute sequence positions coincide: `cut` needs no rebasing.
    """
    prompt = build_prompt(question, prompt_type)
    query = ""                                   # generated text, mirrors math_eval's query
    cut: int | None = None                       # token index where the intervention starts
    n_calls, stopped = 0, "boxed"

    for epoch in range(max_func_call):
        ids = model.tokenizer(prompt + query, add_special_tokens=False,
                              return_tensors="pt")["input_ids"].to(model.cfg.device)
        out, _ = generate_turn(
            model, ids, lambda off, n: hooks_at(cut, off, n),
            max_tokens_per_call, use_cache)
        out = out.rstrip()                       # math_eval.py:262
        query += out

        # math_eval.py:291. Not "ends with ```output" -- vLLM strips the stop string, so
        # a tool-calling turn ends at the code block's own closing fence.
        if not ("boxed" not in out and out.endswith("```")):
            stopped = "boxed" if "boxed" in out else "no_tool_call"
            break

        program = extract_program(query)
        res, report = executor.apply(program)
        query += f"\n```output\n{res if res else report}\n```\n"
        n_calls += 1
        if cut is None:
            # First tool output is now in context: the intervention switches on here and
            # stays on. Re-tokenised because the injected block is not model output.
            cut = int(model.tokenizer(prompt + query, add_special_tokens=False,
                                      return_tensors="pt")["input_ids"].shape[1])
        if epoch == max_func_call - 1:
            query += "\nReach max function call limit."      # math_eval.py:357
            stopped = "max_func_call"

    return dict(code=query, n_tool_calls=n_calls, stopped_because=stopped,
                intervened=cut is not None)


def hooks_factory(method: str, **kw):
    """(cut, offset, n_tokens) -> hooks. One shape for both methods, off until `cut` exists.

    A forward covers absolute positions [offset, offset + n_tokens). The hooks index into
    that forward's tensor, so an absolute position p is local index p - offset. Without the
    cache offset is always 0 and the two coincide; with it they do not, and using absolute
    indices directly would steer the wrong tokens or none at all.

    Every position from `cut` onward is covered, so tokens generated after the tool output
    are steered as well -- that is when the model writes its answer, which is the moment the
    intervention exists to act on.
    """
    from ..hooks import scale_hooks, steer_hooks

    def build(cut, offset, n_tokens):
        if cut is None:
            return []
        local = [i for i in range(n_tokens) if offset + i >= cut]
        if not local:
            return []
        if method == "steer":
            return steer_hooks(kw["layer"], kw["vec"], local, kw["alpha"])
        return scale_hooks(kw["scales"], local)
    return build
