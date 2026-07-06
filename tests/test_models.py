"""Local sanity checks for src/models.py restoration pipeline.

Uses ONLY the mock/dummy inference path (use_mock=True everywhere) --
never loads or downloads a real HuggingFace/MDLM model, per the project's
local-testing rules.

Run: python -m pytest tests/test_models.py
"""

from __future__ import annotations

import os
import re

import pytest

from src.data import RandomTokenMasking, SpanMasking, load_toy_dataset
from src.mdlm_utils import align_mask_positions_to_tokens
from src.models import (
    MDLMRestorer,
    QwenFIMRestorer,
    RestorationResult,
    build_restorer,
)


def test_qwen_fim_restorer_span_masking_produces_result() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    result = QwenFIMRestorer(use_mock=True).restore(corruption)

    assert isinstance(result, RestorationResult)
    assert result.is_mock is True
    assert len(result.restored_spans) == 1
    assert result.model_name
    assert result.prompt_used is not None
    assert "<|fim_prefix|>" in result.prompt_used


def test_qwen_fim_restorer_random_token_masking_multiple_positions() -> None:
    text = load_toy_dataset()[2]
    corruption = RandomTokenMasking(seed=42).corrupt(text, 0.3)
    result = QwenFIMRestorer(use_mock=True).restore(corruption)

    assert len(result.restored_spans) == len(corruption.mask_positions)
    for ground_truth, restored in zip(
        corruption.ground_truth_spans, result.restored_spans
    ):
        assert ground_truth in restored


def test_mdlm_restorer_mock_produces_result() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    result = MDLMRestorer(use_mock=True, num_timesteps=8).restore(corruption)

    assert isinstance(result, RestorationResult)
    assert result.inference_config["num_timesteps"] == 8


def test_mdlm_restorer_num_timesteps_reflected_in_config() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    result_a = MDLMRestorer(use_mock=True, num_timesteps=4).restore(corruption)
    result_b = MDLMRestorer(use_mock=True, num_timesteps=16).restore(corruption)

    assert result_a.inference_config["num_timesteps"] == 4
    assert result_b.inference_config["num_timesteps"] == 16
    assert result_a.restored_spans != result_b.restored_spans


def test_restoration_result_alignment_with_corruption() -> None:
    text = load_toy_dataset()[2]
    for corruptor in (SpanMasking(seed=42), RandomTokenMasking(seed=42)):
        corruption = corruptor.corrupt(text, 0.3)
        for restorer in (QwenFIMRestorer(use_mock=True), MDLMRestorer(use_mock=True)):
            result = restorer.restore(corruption)
            assert (
                len(result.restored_spans)
                == len(result.corruption.ground_truth_spans)
                == len(result.corruption.mask_positions)
            )


def test_build_restorer_dispatches_correctly() -> None:
    assert isinstance(build_restorer("qwen_fim"), QwenFIMRestorer)
    assert isinstance(build_restorer("mdlm"), MDLMRestorer)
    with pytest.raises(ValueError):
        build_restorer("unknown")


def test_build_restorer_defaults_to_mock() -> None:
    assert build_restorer("qwen_fim").use_mock is True
    assert build_restorer("mdlm").use_mock is True


def test_align_mask_positions_to_tokens_word_per_token() -> None:
    class _StubTokenizer:
        """Minimal offset-mapping tokenizer stub (one token per word),
        avoiding any real HF tokenizer load in local tests."""

        def __call__(self, text: str, return_offsets_mapping: bool = False):
            offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            return {"offset_mapping": offsets}

    text = "The quick brown fox jumps"
    token_ranges = align_mask_positions_to_tokens(
        original_text=text,
        mask_positions=[(1, 3), (4, 5)],
        tokenizer=_StubTokenizer(),
    )
    assert token_ranges == [(1, 3), (4, 5)]


@pytest.mark.skipif(
    not os.environ.get("RUN_GPU_TESTS"),
    reason="requires real MDLM weights + GPU/network, run only in Colab",
)
def test_mdlm_restore_real_smoke() -> None:
    text = load_toy_dataset()[0]
    corruption = SpanMasking(seed=1).corrupt(text, 0.3)
    result = MDLMRestorer(use_mock=False, num_timesteps=4).restore(corruption)

    assert isinstance(result, RestorationResult)
    assert result.is_mock is False
    assert len(result.restored_spans) == len(corruption.mask_positions)


def test_unload_is_safe_before_any_real_restore() -> None:
    qwen = QwenFIMRestorer(use_mock=True)
    mdlm = MDLMRestorer(use_mock=True)

    qwen.unload()
    mdlm.unload()

    assert qwen._model is None
    assert mdlm._model is None


def test_latency_is_recorded() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    qwen_result = QwenFIMRestorer(use_mock=True).restore(corruption)
    mdlm_result = MDLMRestorer(use_mock=True).restore(corruption)

    assert qwen_result.latency_seconds >= 0.0
    assert mdlm_result.latency_seconds >= 0.0
