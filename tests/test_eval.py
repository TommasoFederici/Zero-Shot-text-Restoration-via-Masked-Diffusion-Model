"""Local sanity checks for src/eval.py evaluation pipeline.

Uses ONLY the mock/dummy inference path (use_mock=True everywhere) --
never loads or downloads a real HuggingFace model (BERTScore's embedding
model, GPT-2 for perplexity), per the project's local-testing rules.
ROUGE-L is pure Python and is always computed for real.

Run: python -m pytest tests/test_eval.py
"""

from __future__ import annotations

from src.data import RandomTokenMasking, SpanMasking, load_toy_dataset
from src.eval import EvaluationResult, evaluate_batch, evaluate_restoration
from src.models import MDLMRestorer, QwenFIMRestorer, RestorationResult


def _build_restoration_result(
    corruption, restored_spans: list[str], restored_text: str
) -> RestorationResult:
    return RestorationResult(
        corruption=corruption,
        restored_text=restored_text,
        restored_spans=restored_spans,
        model_name="hand-built",
        is_mock=True,
        inference_config={},
        prompt_used=None,
        latency_seconds=0.0,
    )


def test_evaluate_restoration_returns_evaluation_result() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    restoration = MDLMRestorer(use_mock=True).restore(corruption)
    result = evaluate_restoration(restoration, use_mock=True)

    assert isinstance(result, EvaluationResult)
    assert 0.0 <= result.rouge_l_f1 <= 1.0
    assert all(0.0 <= score <= 1.0 for score in result.rouge_l_f1_per_span)
    assert 0.0 <= result.bertscore_f1 <= 1.0
    assert result.perplexity > 0.0


def test_evaluate_restoration_metadata_passthrough() -> None:
    text = load_toy_dataset()[2]
    corruption = RandomTokenMasking(seed=42).corrupt(text, 0.3)
    restoration = QwenFIMRestorer(use_mock=True).restore(corruption)
    result = evaluate_restoration(restoration, use_mock=True)

    assert result.model_name == restoration.model_name
    assert result.corruption_type == corruption.corruption_type
    assert result.masking_ratio_requested == corruption.masking_ratio_requested
    assert result.masking_ratio_actual == corruption.masking_ratio_actual
    assert result.inference_config == restoration.inference_config


def test_rouge_l_perfect_match_scores_high() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    restoration = _build_restoration_result(
        corruption, corruption.ground_truth_spans, corruption.original_text
    )
    result = evaluate_restoration(restoration, use_mock=True)

    assert result.rouge_l_f1 == 1.0
    assert all(score == 1.0 for score in result.rouge_l_f1_per_span)


def test_rouge_l_no_overlap_scores_low() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    mismatched_text = "Zzyzx qwerty plonk gribble squonk wobbulate."
    restoration = _build_restoration_result(
        corruption, ["gribble squonk"], mismatched_text
    )
    perfect_restoration = _build_restoration_result(
        corruption, corruption.ground_truth_spans, corruption.original_text
    )

    mismatch_result = evaluate_restoration(restoration, use_mock=True)
    perfect_result = evaluate_restoration(perfect_restoration, use_mock=True)

    assert mismatch_result.rouge_l_f1 < 0.2
    assert mismatch_result.rouge_l_f1 < perfect_result.rouge_l_f1


def test_mock_bertscore_perfect_match_vs_mismatch() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    mismatched_text = "Zzyzx qwerty plonk gribble squonk wobbulate."

    perfect_restoration = _build_restoration_result(
        corruption, corruption.ground_truth_spans, corruption.original_text
    )
    mismatch_restoration = _build_restoration_result(
        corruption, ["gribble squonk"], mismatched_text
    )

    perfect_result = evaluate_restoration(perfect_restoration, use_mock=True)
    mismatch_result = evaluate_restoration(mismatch_restoration, use_mock=True)

    assert perfect_result.bertscore_f1 > mismatch_result.bertscore_f1


def test_mock_perplexity_perfect_match_lower_than_mismatch() -> None:
    text = load_toy_dataset()[2]
    corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    mismatched_text = "Zzyzx qwerty plonk gribble squonk wobbulate."

    perfect_restoration = _build_restoration_result(
        corruption, corruption.ground_truth_spans, corruption.original_text
    )
    mismatch_restoration = _build_restoration_result(
        corruption, ["gribble squonk"], mismatched_text
    )

    perfect_result = evaluate_restoration(perfect_restoration, use_mock=True)
    mismatch_result = evaluate_restoration(mismatch_restoration, use_mock=True)

    assert perfect_result.perplexity < mismatch_result.perplexity


def test_evaluate_batch_length_matches_input() -> None:
    text = load_toy_dataset()[2]
    span_corruption = SpanMasking(seed=42).corrupt(text, 0.3)
    random_corruption = RandomTokenMasking(seed=42).corrupt(text, 0.3)

    restorations = [
        QwenFIMRestorer(use_mock=True).restore(span_corruption),
        MDLMRestorer(use_mock=True).restore(random_corruption),
    ]
    results = evaluate_batch(restorations, use_mock=True)

    assert len(results) == len(restorations)
    for result, restoration in zip(results, restorations):
        assert result.model_name == restoration.model_name
        assert result.corruption_type == restoration.corruption.corruption_type
