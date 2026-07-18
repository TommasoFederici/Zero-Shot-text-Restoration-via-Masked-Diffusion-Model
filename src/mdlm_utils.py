"""Custom absorbing-state denoising loop for real MDLM inference.

Kept separate from `src/models.py`: this module operates on plain token
ids/tensors rather than the `CorruptionResult`/`RestorationResult`
dataclasses, so the tokenization-alignment and denoising logic can be
unit-tested independently of the `Restorer` interface.

`kuleshov-group/mdlm-owt` is loadable via
`transformers.AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)`
using GPT-2's tokenizer/vocab, but its own reference repository
(`kuleshov-group/mdlm`) only ships an *unconditional* sampler -- there is no
infilling/conditioning entry point to call directly. `denoise` below ports
the reverse-diffusion update rule from that repo's `diffusion.py`
(`_ddpm_update`, log-linear noise schedule from `noise_schedule.py`) rather
than approximating it: at each step from noise level `t` down to `s < t`,
every currently-masked position's next value is drawn from the analytic
posterior

    q_xs        = p_x0 * (move_chance_t - move_chance_s)
    q_xs[MASK]  = move_chance_s   (overwritten, not added)
    x_s        ~ Categorical(q_xs)   (Gumbel-max sampling, not argmax)

where `p_x0 = softmax(model(x_t))` and `move_chance_t = (1 - eps) * t` is the
log-linear schedule's masking probability at time `t`. This is genuinely
stochastic -- which positions unmask at a given step, and which token they
resolve to, are both sampled, not chosen by confidence ranking.

Conditioning/infilling needs no special-casing to bolt onto that formula:
the official update already guarantees any position that starts off
*unmasked* is copied forward unchanged forever (it only ever resamples
positions still equal to `mask_token_id`). Since context tokens here are
simply never set to `mask_token_id` in the first place, running the exact
official reverse step over the whole sequence already is correct
conditional infilling.

One deliberate deviation from the official code: the paper's schedule is
tuned for O(1000)-step unconditional sampling, where the residual
probability of a position staying masked after the final step (a few
tenths of a percent, from `move_chance_s` at `t=eps`) is negligible. Our
ablation runs as few as 4-16 steps, where that residual would leak literal
`mask_token_id`s into decoded text. So `denoise` finalizes any position
still masked after the reverse loop with one extra forward pass and an
argmax over non-mask logits -- a practical cleanup step, not part of the
ported sampling algorithm.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import torch


def _word_char_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) character span of each whitespace-delimited word.

    Matches the segmentation used by `text.split()` (see
    `src/data.py`'s `_split_words`), so word indices from
    `CorruptionResult.mask_positions` line up with the spans returned here.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def compute_word_window(
    num_words: int,
    mask_positions: list[tuple[int, int]],
    context_window_words: int,
) -> tuple[int, int]:
    """Compute a word-index window covering all mask positions plus margin.

    Used to bound the amount of surrounding text tokenized/denoised for a
    single restoration, instead of always using the entire document --
    latency/memory scale with sequence length, and most of a long
    document is irrelevant to filling a local gap.

    Args:
        num_words: total number of words in the source text.
        mask_positions: word-index (start, end) ranges to be restored.
        context_window_words: number of words of margin to keep on each
            side of the outermost mask positions.

    Returns:
        A `(window_start, window_end)` word-index pair, clamped to
        `[0, num_words]`, spanning from `context_window_words` words
        before the earliest mask start to `context_window_words` words
        after the latest mask end.
    """
    window_start = max(0, min(s for s, _ in mask_positions) - context_window_words)
    window_end = min(num_words, max(e for _, e in mask_positions) + context_window_words)
    return window_start, window_end


def align_mask_positions_to_tokens(
    original_text: str,
    mask_positions: list[tuple[int, int]],
    tokenizer: Any,
) -> list[tuple[int, int]]:
    """Map word-index mask ranges to subword token-index ranges.

    `original_text` (not the corrupted text) is tokenized with an offset
    mapping, since GPT-2's tokenizer has no native mask token and
    re-tokenizing a string containing a literal placeholder such as
    "<mask>" would fragment unpredictably.

    Args:
        original_text: the uncorrupted source text.
        mask_positions: word-index (start, end) ranges, as produced by
            `Corruptor.corrupt` (`src/data.py`).
        tokenizer: a HuggingFace tokenizer exposing
            `__call__(text, return_offsets_mapping=True)`.

    Returns:
        A list of (token_start, token_end) ranges, same length/order as
        `mask_positions`, giving the token indices covering each masked
        word range.

    Raises:
        ValueError: if a mask range has no corresponding tokens.
    """
    word_spans = _word_char_spans(original_text)
    encoding = tokenizer(original_text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    token_ranges: list[tuple[int, int]] = []
    for start, end in mask_positions:
        char_start = word_spans[start][0]
        char_end = word_spans[end - 1][1]

        covered = [
            i
            for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > char_start and tok_start < char_end
        ]
        if not covered:
            raise ValueError(f"no tokens found covering word range ({start}, {end})")
        token_ranges.append((min(covered), max(covered) + 1))
    return token_ranges


def resolve_mask_token_id(model: Any, tokenizer: Any) -> int:
    """Determine the token id representing MDLM's absorbing/mask state.

    `kuleshov-group/mdlm-owt`'s `config.json` reports `vocab_size=50258`
    against GPT-2's own vocab of 50257 -- one extra slot, matching the
    common absorbing-state-diffusion convention of appending a single
    mask/absorbing token at the final vocab index. That's used as a
    last-resort fallback here, since the model's `modeling_mdlm.py` has
    no explicit `mask_token_id` concept of its own.

    Args:
        model: the loaded MDLM model (`trust_remote_code=True`).
        tokenizer: the associated tokenizer.

    Returns:
        The mask token id to use when constructing masked `input_ids`.

    Raises:
        RuntimeError: if no mask token id can be determined from the
            tokenizer, the model's config, or the vocab-size convention.
            This must not be guessed silently -- verify interactively (in
            Colab) which id the checkpoint actually expects before
            relying on this function.
    """
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is not None:
        return mask_token_id

    model_vocab_size = getattr(model.config, "vocab_size", None)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    if (
        model_vocab_size is not None
        and tokenizer_vocab_size is not None
        and model_vocab_size == tokenizer_vocab_size + 1
    ):
        return model_vocab_size - 1

    raise RuntimeError(
        "could not resolve a mask token id from the tokenizer, model "
        "config, or vocab-size convention; inspect the model's remote "
        "code/config to find the correct id before running real MDLM "
        "inference"
    )


def _move_chance(t: torch.Tensor, eps: float) -> torch.Tensor:
    """Log-linear noise schedule's masking probability at time `t`.

    Closed form of `1 - exp(-sigma(t))` for `kuleshov-group/mdlm`'s
    `LogLinearNoise` schedule (`sigma(t) = -log1p(-(1 - eps) * t)`), ported
    from that repo's `noise_schedule.py`. `t` ranges over `[eps, 1]`, so the
    returned masking probability ranges over `[eps * (1 - eps), 1 - eps]`.

    Args:
        t: noise level(s), any shape.
        eps: schedule floor (see `denoise`).

    Returns:
        Masking probability at `t`, same shape as `t`.
    """
    return (1 - eps) * t


def _sample_categorical(
    probs: torch.Tensor, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Sample from unnormalized categorical weights via the Gumbel-max trick.

    Ported verbatim (modulo an added `generator` argument) from
    `kuleshov-group/mdlm`'s `diffusion.py::_sample_categorical`. Dividing
    `probs` by i.i.d. `Exponential(1)`-like noise and taking the argmax
    samples exactly from the categorical distribution proportional to
    `probs`, regardless of `probs`'s overall normalization -- so `probs`
    need not sum to 1 along the last dimension.

    Args:
        probs: nonnegative unnormalized categorical weights, shape
            `[..., vocab_size]`.
        generator: optional `torch.Generator` for reproducible sampling.

    Returns:
        Sampled indices, shape `probs.shape[:-1]`.
    """
    uniform = torch.rand(probs.shape, device=probs.device, generator=generator)
    gumbel_norm = 1e-10 - (uniform + 1e-10).log()
    return (probs / gumbel_norm).argmax(dim=-1)


def denoise(
    model: Any,
    input_ids: torch.Tensor,
    masked_token_ranges: list[tuple[int, int]],
    mask_token_id: int,
    num_timesteps: int,
    device: torch.device,
    eps: float = 1e-3,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Resolve masked positions via MDLM's absorbing-state reverse process.

    Ports the reverse-diffusion update from `kuleshov-group/mdlm`'s
    `diffusion.py::_ddpm_update` (see module docstring for the full
    derivation): at each of `num_timesteps` steps from noise level `t` down
    to `s < t`, every still-masked position's next value is sampled from
    the analytic posterior `q_xs = p_x0 * (move_chance_t - move_chance_s)`
    (with the mask-index entry overwritten to `move_chance_s`), via
    Gumbel-max categorical sampling -- not argmax, and not a fixed number
    of positions per step. Positions outside `masked_token_ranges` (never
    set to `mask_token_id`) are never touched, which is exactly the
    official update's own `copy_flag` behavior for already-unmasked
    positions -- no separate conditioning/clamping logic is needed.

    Any position still equal to `mask_token_id` after the reverse loop
    (possible at low `num_timesteps`, where the schedule's residual
    stay-masked probability at the final step is non-negligible) is
    resolved with one extra forward pass and an argmax over non-mask
    logits, so the returned tensor never contains `mask_token_id`.

    Args:
        model: the loaded MDLM model, callable as
            `model(input_ids, timesteps=..., return_dict=True).logits`.
            `kuleshov-group/mdlm-owt` has `time_conditioning=False` in its
            config, so its forward pass ignores `timesteps` internally --
            the real per-step noise level is still passed here so this
            keeps working correctly for a time-conditional checkpoint.
        input_ids: token ids, shape `[1, seq_len]`, with every position
            inside `masked_token_ranges` already set to `mask_token_id`.
        masked_token_ranges: token-index (start, end) ranges to resolve,
            as produced by `align_mask_positions_to_tokens`.
        mask_token_id: the absorbing-state token id (see
            `resolve_mask_token_id`).
        num_timesteps: number of denoising steps (`ablation axis:
            "inference efficiency"`); more steps means finer-grained noise
            increments, closer to the paper's continuous-time reverse
            process.
        device: device to run the model on.
        eps: log-linear noise schedule floor -- `t` ranges over
            `[eps, 1]` across the reverse loop. Matches the official
            repo's default.
        generator: optional `torch.Generator` for reproducible sampling.
            `None` (default) samples non-reproducibly, matching the
            official sampler's default stochastic behavior.

    Returns:
        The resolved `input_ids` tensor, shape `[1, seq_len]`, with every
        previously-masked position replaced by a non-mask token id.
    """
    input_ids = input_ids.clone().to(device)
    remaining: list[int] = [
        i for start, end in masked_token_ranges for i in range(start, end)
    ]
    timesteps = torch.linspace(1.0, eps, num_timesteps + 1, device=device)

    for step in range(num_timesteps):
        if not remaining:
            break
        t, s = timesteps[step], timesteps[step + 1]
        move_chance_t = _move_chance(t, eps)
        move_chance_s = _move_chance(s, eps)

        with torch.no_grad():
            logits = model(
                input_ids,
                timesteps=t.expand(input_ids.shape[0]),
                return_dict=True,
            ).logits
        position_logits = logits[0, remaining, :]
        p_x0 = torch.softmax(position_logits, dim=-1)

        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, mask_token_id] = move_chance_s
        sampled_ids = _sample_categorical(q_xs, generator=generator)

        for local_idx, position in enumerate(remaining):
            input_ids[0, position] = sampled_ids[local_idx]
        remaining = [pos for pos in remaining if input_ids[0, pos] == mask_token_id]

    if remaining:
        with torch.no_grad():
            logits = model(
                input_ids,
                timesteps=timesteps[-1].expand(input_ids.shape[0]),
                return_dict=True,
            ).logits
        position_logits = logits[0, remaining, :].clone()
        position_logits[:, mask_token_id] = float("-inf")
        predicted_ids = position_logits.argmax(dim=-1)
        for local_idx, position in enumerate(remaining):
            input_ids[0, position] = predicted_ids[local_idx]

    return input_ids
