"""Local sanity checks for src/mdlm_utils.py's reverse-diffusion `denoise`.

Uses a small dummy `torch.nn.Module` standing in for a real MDLM
checkpoint -- never loads real HuggingFace weights, per the project's
local-testing rules.

Run: python -m pytest tests/test_mdlm_utils.py
"""

from __future__ import annotations

from typing import Optional

import torch

from src.mdlm_utils import denoise

VOCAB_SIZE = 50
MASK_TOKEN_ID = VOCAB_SIZE - 1


class _DummyOutput:
    """Mimics the `.logits`-bearing object real HF model calls return."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _DummyMDLMModel(torch.nn.Module):
    """Predicts the same fixed logit vector at every position.

    Not a real diffusion model -- just enough of a stand-in to drive
    `denoise`'s reverse loop with a controllable predicted x0 distribution,
    without loading real HuggingFace weights.
    """

    def __init__(self, logits_row: torch.Tensor) -> None:
        super().__init__()
        self.logits_row = logits_row

    def forward(
        self,
        input_ids: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> _DummyOutput:
        batch, seq_len = input_ids.shape
        logits = self.logits_row.expand(batch, seq_len, -1).clone()
        return _DummyOutput(logits)


def _concentrated_logits(target_id: int) -> torch.Tensor:
    """Near-all predicted probability mass on a single target token."""
    logits = torch.zeros(VOCAB_SIZE)
    logits[target_id] = 20.0
    return logits


def _uniform_nonmask_logits() -> torch.Tensor:
    """Roughly uniform predicted probability across all non-mask tokens."""
    logits = torch.zeros(VOCAB_SIZE)
    logits[MASK_TOKEN_ID] = -20.0
    return logits


def test_denoise_never_touches_context_positions() -> None:
    context = torch.tensor([1, 2, 3, 6, 7])
    input_ids = torch.tensor([[1, 2, 3, MASK_TOKEN_ID, MASK_TOKEN_ID, 6, 7]])
    model = _DummyMDLMModel(_concentrated_logits(target_id=5))

    resolved = denoise(
        model=model,
        input_ids=input_ids,
        masked_token_ranges=[(3, 5)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=4,
        device=torch.device("cpu"),
    )

    assert torch.equal(resolved[0, [0, 1, 2, 5, 6]], context)


def test_denoise_never_leaves_mask_tokens_even_at_one_step() -> None:
    input_ids = torch.tensor([[1, MASK_TOKEN_ID, MASK_TOKEN_ID, MASK_TOKEN_ID, 2]])
    model = _DummyMDLMModel(_uniform_nonmask_logits())

    resolved = denoise(
        model=model,
        input_ids=input_ids,
        masked_token_ranges=[(1, 4)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=1,
        device=torch.device("cpu"),
    )

    assert not bool((resolved == MASK_TOKEN_ID).any())


def test_denoise_resolves_to_high_confidence_prediction() -> None:
    target_id = 5
    input_ids = torch.tensor([[1, MASK_TOKEN_ID, 2]])
    model = _DummyMDLMModel(_concentrated_logits(target_id))

    resolved = denoise(
        model=model,
        input_ids=input_ids,
        masked_token_ranges=[(1, 2)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=8,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )

    assert resolved[0, 1].item() == target_id


def test_denoise_same_seed_is_reproducible() -> None:
    input_ids = torch.tensor([[1] + [MASK_TOKEN_ID] * 6 + [2]])
    model = _DummyMDLMModel(_uniform_nonmask_logits())

    resolved_a = denoise(
        model=model,
        input_ids=input_ids.clone(),
        masked_token_ranges=[(1, 7)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=8,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(123),
    )
    resolved_b = denoise(
        model=model,
        input_ids=input_ids.clone(),
        masked_token_ranges=[(1, 7)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=8,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(123),
    )

    assert torch.equal(resolved_a, resolved_b)


def test_denoise_is_stochastic_without_a_fixed_seed() -> None:
    input_ids = torch.tensor([[1] + [MASK_TOKEN_ID] * 6 + [2]])
    model = _DummyMDLMModel(_uniform_nonmask_logits())

    resolved_a = denoise(
        model=model,
        input_ids=input_ids.clone(),
        masked_token_ranges=[(1, 7)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=8,
        device=torch.device("cpu"),
    )
    resolved_b = denoise(
        model=model,
        input_ids=input_ids.clone(),
        masked_token_ranges=[(1, 7)],
        mask_token_id=MASK_TOKEN_ID,
        num_timesteps=8,
        device=torch.device("cpu"),
    )

    assert not torch.equal(resolved_a, resolved_b)
