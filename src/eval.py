"""Evaluation pipeline for restored text.

Provides:
    - `EvaluationResult`: ROUGE-L, BERTScore, and Perplexity scores for a
      single `RestorationResult` (src/models.py), plus flattened metadata
      pass-through fields (from the nested `CorruptionResult`, src/data.py)
      so downstream CSV/plot export in the Colab ablation loop (Phase 4)
      doesn't need to re-traverse nested objects.
    - `evaluate_restoration` / `evaluate_batch`: orchestration functions,
      mocking BERTScore/Perplexity by default since both load real HF
      models internally (ROUGE-L is pure-Python and always computed for
      real).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.models import RestorationResult


@dataclass
class EvaluationResult:
    """Evaluation scores for a single `RestorationResult`.

    `model_name`, `corruption_type`, `masking_ratio_requested`,
    `masking_ratio_actual`, and `inference_config` are flattened
    pass-through copies of fields nested inside `restoration` and
    `restoration.corruption`, so the Colab notebook's ablation loop can
    build a DataFrame/CSV directly from a list of `EvaluationResult`
    without traversing nested objects. The full `restoration` is kept for
    traceability/debuggability.
    """

    restoration: RestorationResult
    rouge_l_f1: float
    rouge_l_f1_per_span: list[float]
    bertscore_precision: float
    bertscore_recall: float
    bertscore_f1: float
    perplexity: float
    is_mock: bool
    model_name: str
    corruption_type: str
    masking_ratio_requested: float
    masking_ratio_actual: float
    inference_config: dict[str, object]


def _rouge_l_f1(candidate: str, reference: str) -> float:
    """Compute the ROUGE-L F1 score of `candidate` against `reference`.

    Uses the `rouge-score` package directly (not `evaluate`'s wrapper),
    since it is pure Python with no network/model-loading indirection and
    is safe to always run for real, even in local dev and tests.

    Args:
        candidate: generated/restored text.
        reference: ground-truth text to compare against.

    Returns:
        The ROUGE-L F1 score, in [0.0, 1.0].
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, candidate)["rougeL"].fmeasure


def _jaccard_overlap(candidate: str, reference: str) -> float:
    """Return the Jaccard overlap between the whitespace-token sets of
    `candidate` and `reference`, used as the deterministic basis for mock
    BERTScore/Perplexity values.

    Args:
        candidate: generated/restored text.
        reference: ground-truth text to compare against.

    Returns:
        A value in [0.0, 1.0]; 1.0 for identical token sets, 0.0 for
        disjoint token sets. Two empty token sets are treated as fully
        overlapping (1.0).
    """
    candidate_tokens = set(candidate.split())
    reference_tokens = set(reference.split())
    if not candidate_tokens and not reference_tokens:
        return 1.0
    union = candidate_tokens | reference_tokens
    if not union:
        return 1.0
    intersection = candidate_tokens & reference_tokens
    return len(intersection) / len(union)


def _bertscore_mock(candidate: str, reference: str) -> tuple[float, float, float]:
    """Produce deterministic fake BERTScore precision/recall/F1 without
    loading any model.

    Maps Jaccard token overlap into `[0.5, 1.0]`, mimicking real
    BERTScore's non-zero floor for unrelated English text (contextual
    embeddings rarely score fully dissimilar sentences near 0).

    Args:
        candidate: generated/restored text.
        reference: ground-truth text to compare against.

    Returns:
        A `(precision, recall, f1)` tuple, all equal under this heuristic.
    """
    score = 0.5 + 0.5 * _jaccard_overlap(candidate, reference)
    return score, score, score


def _bertscore_real(candidate: str, reference: str) -> tuple[float, float, float]:
    """Compute real BERTScore precision/recall/F1 via the `bert-score` package.

    Args:
        candidate: generated/restored text.
        reference: ground-truth text to compare against.

    Returns:
        A `(precision, recall, f1)` tuple.

    Note:
        Lazily imports `bert_score` and loads a real transformer model
        (RoBERTa-large by default) to compute contextual embeddings. Do
        NOT call this in local dev or tests. Intended only for the Colab
        notebook (Phase 4) or explicit opt-in scripts.
    """
    from bert_score import score

    precision, recall, f1 = score([candidate], [reference], lang="en")
    return precision.item(), recall.item(), f1.item()


def _perplexity_mock(candidate: str, reference: str) -> float:
    """Produce a deterministic fake perplexity value without loading any model.

    Args:
        candidate: generated/restored text.
        reference: ground-truth text to compare against.

    Returns:
        A low pseudo-perplexity (10.0) for an exact match, otherwise a
        value in `(50.0, 60.0]` that increases as Jaccard overlap
        decreases -- deterministic and monotonic, with no network/model.
    """
    if candidate == reference:
        return 10.0
    return 50.0 + 10.0 * (1.0 - _jaccard_overlap(candidate, reference))


def _perplexity_real(text: str) -> float:
    """Compute real perplexity of `text` via HF `evaluate`'s perplexity metric.

    Args:
        text: text to score (typically the full restored text).

    Returns:
        The perplexity score (GPT-2 default model).

    Note:
        Lazily imports `evaluate` and loads a real causal LM (GPT-2 by
        default) to compute token log-likelihoods. Do NOT call this in
        local dev or tests. Intended only for the Colab notebook
        (Phase 4) or explicit opt-in scripts.

        `text` is the fully reconstructed document (see `_reinsert_spans`
        in `src/models.py`), not the windowed slice `context_window_words`
        bounds for restoration -- so it can still exceed GPT-2's 1024-token
        context on long WikiText-103 documents. `evaluate`'s perplexity
        metric does not truncate unless `max_length` is passed explicitly;
        without it, an over-long sequence overflows GPT-2's position
        embeddings, which surfaces as an opaque CUDA device-side assert
        (asynchronous GPU errors get reported at a later, unrelated call)
        rather than a clean Python error.
    """
    from evaluate import load

    perplexity_metric = load("perplexity", module_type="metric")
    results = perplexity_metric.compute(
        predictions=[text], model_id="gpt2", max_length=1024
    )
    return results["mean_perplexity"]


def evaluate_restoration(
    result: RestorationResult, use_mock: bool = True
) -> EvaluationResult:
    """Compute ROUGE-L, BERTScore, and Perplexity for a `RestorationResult`.

    ROUGE-L is always computed for real (no model/network dependency).
    BERTScore and Perplexity are mocked by default, since both load real
    HF models internally, per the project's local-testing rules.

    Args:
        result: the `RestorationResult` to evaluate.
        use_mock: if True (default), BERTScore and Perplexity use
            deterministic mock implementations that load no model. Set
            False only in the Colab notebook (Phase 4) or explicit
            opt-in scripts.

    Returns:
        An `EvaluationResult` with all scores and pass-through metadata.
    """
    corruption = result.corruption

    rouge_l_f1 = _rouge_l_f1(result.restored_text, corruption.original_text)
    rouge_l_f1_per_span = [
        _rouge_l_f1(restored, ground_truth)
        for restored, ground_truth in zip(
            result.restored_spans, corruption.ground_truth_spans
        )
    ]

    if use_mock:
        bertscore_precision, bertscore_recall, bertscore_f1 = _bertscore_mock(
            result.restored_text, corruption.original_text
        )
        perplexity = _perplexity_mock(result.restored_text, corruption.original_text)
    else:
        bertscore_precision, bertscore_recall, bertscore_f1 = _bertscore_real(
            result.restored_text, corruption.original_text
        )
        perplexity = _perplexity_real(result.restored_text)

    return EvaluationResult(
        restoration=result,
        rouge_l_f1=rouge_l_f1,
        rouge_l_f1_per_span=rouge_l_f1_per_span,
        bertscore_precision=bertscore_precision,
        bertscore_recall=bertscore_recall,
        bertscore_f1=bertscore_f1,
        perplexity=perplexity,
        is_mock=use_mock,
        model_name=result.model_name,
        corruption_type=corruption.corruption_type,
        masking_ratio_requested=corruption.masking_ratio_requested,
        masking_ratio_actual=corruption.masking_ratio_actual,
        inference_config=result.inference_config,
    )


def evaluate_batch(
    results: list[RestorationResult], use_mock: bool = True
) -> list[EvaluationResult]:
    """Evaluate a list of `RestorationResult`s.

    Args:
        results: the `RestorationResult`s to evaluate.
        use_mock: see `evaluate_restoration`.

    Returns:
        A list of `EvaluationResult`, one per input, in the same order.
    """
    return [evaluate_restoration(result, use_mock=use_mock) for result in results]
