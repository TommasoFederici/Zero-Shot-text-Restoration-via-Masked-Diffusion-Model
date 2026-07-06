"""Local sanity checks for src/data.py corruption pipeline.

Uses ONLY load_toy_dataset() -- never triggers a real WikiText-103
download, per the project's local-testing rules.

Run: python -m pytest tests/test_data.py
"""

from __future__ import annotations

import pytest

from src.data import (
    RandomTokenMasking,
    SpanMasking,
    load_toy_dataset,
)


def test_load_toy_dataset_non_empty() -> None:
    texts = load_toy_dataset()
    assert isinstance(texts, list)
    assert len(texts) > 0
    assert all(isinstance(t, str) and t.strip() for t in texts)


def test_random_token_masking_masks_expected_count() -> None:
    text = load_toy_dataset()[2]
    num_words = len(text.split())
    result = RandomTokenMasking(seed=42).corrupt(text, 0.3)

    expected_count = max(1, min(num_words, round(0.3 * num_words)))
    assert len(result.ground_truth_spans) == expected_count
    assert len(result.mask_positions) == expected_count
    assert result.corrupted_text.count(result.mask_token) == expected_count
    assert abs(result.masking_ratio_actual - expected_count / num_words) < 1e-9


def test_span_masking_single_contiguous_placeholder() -> None:
    text = load_toy_dataset()[2]
    result = SpanMasking(seed=42).corrupt(text, 0.3)

    assert result.corrupted_text.count(result.mask_token) == 1
    assert len(result.ground_truth_spans) == 1
    assert len(result.mask_positions) == 1

    start, end = result.mask_positions[0]
    words = text.split()
    reconstructed = " ".join(
        words[:start] + result.ground_truth_spans[0].split() + words[end:]
    )
    assert reconstructed == result.original_text


def test_determinism_same_seed_same_result() -> None:
    text = load_toy_dataset()[2]
    result_a = RandomTokenMasking(seed=7).corrupt(text, 0.3)
    result_b = RandomTokenMasking(seed=7).corrupt(text, 0.3)
    assert result_a.corrupted_text == result_b.corrupted_text
    assert result_a.mask_positions == result_b.mask_positions

    span_a = SpanMasking(seed=7).corrupt(text, 0.3)
    span_b = SpanMasking(seed=7).corrupt(text, 0.3)
    assert span_a.corrupted_text == span_b.corrupted_text
    assert span_a.mask_positions == span_b.mask_positions


def test_short_text_masks_at_least_one_word() -> None:
    short_text = "Short text example."  # 3 words
    result = RandomTokenMasking(seed=1).corrupt(short_text, 0.1)
    assert len(result.ground_truth_spans) >= 1
    assert result.corrupted_text.count(result.mask_token) >= 1


def test_span_masking_full_ratio_masks_entire_text() -> None:
    text = "One two three four five."
    result = SpanMasking(seed=1).corrupt(text, 1.0)
    assert result.corrupted_text == result.mask_token
    assert result.ground_truth_spans[0] == " ".join(text.split())


def test_empty_text_raises_value_error() -> None:
    with pytest.raises(ValueError):
        RandomTokenMasking().corrupt("   ", 0.3)
    with pytest.raises(ValueError):
        SpanMasking().corrupt("", 0.3)
