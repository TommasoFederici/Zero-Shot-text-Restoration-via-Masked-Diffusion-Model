"""Custom absorbing-state denoising loop for real MDLM inference.

Kept separate from `src/models.py`: this module operates on plain token
ids/tensors rather than the `CorruptionResult`/`RestorationResult`
dataclasses, so the tokenization-alignment and denoising logic can be
unit-tested independently of the `Restorer` interface.

`kuleshov-group/mdlm-owt` is loadable via
`transformers.AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)`
using GPT-2's tokenizer/vocab, but its own reference repository only
documents unconditional sampling, not conditional/masked infilling.
`denoise` below implements a simplified conditional (absorbing-state)
reverse process: context tokens are clamped and never resampled, while
masked tokens are progressively committed over `num_timesteps` steps in
order of model confidence (highest-confidence predictions first), similar
to MaskGIT-style non-autoregressive decoding.
"""

from __future__ import annotations

import re
from typing import Any

import torch


def _word_char_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) character span of each whitespace-delimited word.

    Matches the segmentation used by `text.split()` (see
    `src/data.py`'s `_split_words`), so word indices from
    `CorruptionResult.mask_positions` line up with the spans returned here.
    """
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


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

    Args:
        model: the loaded MDLM model (`trust_remote_code=True`).
        tokenizer: the associated tokenizer.

    Returns:
        The mask token id to use when constructing masked `input_ids`.

    Raises:
        RuntimeError: if no mask token id can be determined from either
            the tokenizer or the model's config. This must not be guessed
            silently -- verify interactively (in Colab) which id the
            checkpoint actually expects before relying on this function.
    """
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    mask_token_id = getattr(model.config, "mask_token_id", None)
    if mask_token_id is not None:
        return mask_token_id
    raise RuntimeError(
        "could not resolve a mask token id from the tokenizer or model "
        "config; inspect the model's remote code/config to find the "
        "correct id before running real MDLM inference"
    )


def denoise(
    model: Any,
    input_ids: torch.Tensor,
    masked_token_ranges: list[tuple[int, int]],
    mask_token_id: int,
    num_timesteps: int,
    device: torch.device,
) -> torch.Tensor:
    """Iteratively resolve masked positions via confidence-based unmasking.

    Implements a simplified absorbing-state discrete-diffusion reverse
    process: at each of `num_timesteps` steps, the model predicts a token
    for every still-masked position, and the highest-confidence
    predictions are committed (clamped) first, leaving harder positions to
    benefit from more resolved context in later steps. Positions outside
    `masked_token_ranges` are never touched.

    Args:
        model: the loaded MDLM model, callable as `model(input_ids).logits`.
        input_ids: token ids, shape `[1, seq_len]`, with every position
            inside `masked_token_ranges` already set to `mask_token_id`.
        masked_token_ranges: token-index (start, end) ranges to resolve,
            as produced by `align_mask_positions_to_tokens`.
        mask_token_id: the absorbing-state token id (see
            `resolve_mask_token_id`).
        num_timesteps: number of denoising steps; fewer steps commit more
            positions per step (coarser, closer to a single-shot fill),
            more steps commit fewer per step (finer, more sequentially
            conditioned).
        device: device to run the model on.

    Returns:
        The resolved `input_ids` tensor, shape `[1, seq_len]`, with every
        previously-masked position replaced by its committed token id.
    """
    input_ids = input_ids.clone().to(device)
    remaining: list[int] = [
        i for start, end in masked_token_ranges for i in range(start, end)
    ]

    for step in range(num_timesteps):
        if not remaining:
            break
        steps_left = num_timesteps - step

        with torch.no_grad():
            logits = model(input_ids).logits
        position_logits = logits[0, remaining, :]
        probs = torch.softmax(position_logits, dim=-1)
        confidences, predicted_ids = probs.max(dim=-1)

        if step == num_timesteps - 1:
            num_to_commit = len(remaining)
        else:
            num_to_commit = min(
                len(remaining), max(1, round(len(remaining) / steps_left))
            )

        commit_local_indices = torch.argsort(confidences, descending=True)[
            :num_to_commit
        ].tolist()

        for local_idx in commit_local_indices:
            position = remaining[local_idx]
            input_ids[0, position] = predicted_ids[local_idx]

        committed_positions = {remaining[i] for i in commit_local_indices}
        remaining = [pos for pos in remaining if pos not in committed_positions]

    return input_ids
