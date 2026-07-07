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
from typing import Any, Optional

from src.models import RestorationResult

# Lazily-populated module-level caches for the real BERTScore/perplexity
# backends. Both `bert_score.score(...)` and `evaluate`'s perplexity metric
# reload their underlying model from scratch on every call if not cached
# this way -- observed directly in a real Colab run as
# "Some weights of RobertaModel were not initialized..." printed once per
# restoration instead of once per notebook run. Caching mirrors the lazy
# `self._model` pattern already used by `Restorer` subclasses in
# `src/models.py`.
_bertscore_scorer: Optional[Any] = None
_perplexity_model: Optional[Any] = None
_perplexity_tokenizer: Optional[Any] = None
_perplexity_device: Optional[Any] = None


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
        Lazily loads a real transformer model (RoBERTa-large by default,
        cached module-wide in `_bertscore_scorer` after first use) to
        compute contextual embeddings. Do NOT call this in local dev or
        tests. Intended only for the Colab notebook (Phase 4) or explicit
        opt-in scripts.

        Uses the `BERTScorer` class rather than `bert_score.score(...)`:
        the latter reconstructs the underlying model from scratch on
        every call, which is fine once but reloads RoBERTa-large onto the
        GPU for every single restoration when called from `evaluate_batch`.
    """
    global _bertscore_scorer
    if _bertscore_scorer is None:
        from bert_score import BERTScorer

        _bertscore_scorer = BERTScorer(lang="en")

    precision, recall, f1 = _bertscore_scorer.score([candidate], [reference])
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
    """Compute real perplexity of `text` via a cached GPT-2 causal LM.

    Args:
        text: text to score (typically the full restored text).

    Returns:
        The perplexity score (GPT-2 default model).

    Note:
        Lazily loads GPT-2 (cached module-wide in `_perplexity_model`/
        `_perplexity_tokenizer` after first use) to compute token
        log-likelihoods. Do NOT call this in local dev or tests. Intended
        only for the Colab notebook (Phase 4) or explicit opt-in scripts.

        Implemented directly against `transformers` rather than via HF
        `evaluate`'s perplexity metric: that metric's `compute()` calls
        `AutoModelForCausalLM.from_pretrained(model_id)` internally on
        every invocation regardless of caching the `Metric` object
        returned by `evaluate.load(...)`, reloading GPT-2 onto the GPU
        for every single restoration when called from `evaluate_batch`.

        `text` is the fully reconstructed document (see `_reinsert_spans`
        in `src/models.py`), not the windowed slice `context_window_words`
        bounds for restoration -- so it can still exceed GPT-2's 1024-token
        context on long WikiText-103 documents. Truncating to 1024 tokens
        avoids overflowing GPT-2's position embeddings, which otherwise
        surfaces as an opaque CUDA device-side assert (asynchronous GPU
        errors get reported at a later, unrelated call) rather than a
        clean Python error. The tokenizer's `eos_token_id` is prepended as
        a BOS surrogate (GPT-2 has no dedicated BOS token) so the first
        real token is scored with some context, matching the convention
        used by standard perplexity implementations (e.g. HF `evaluate`'s
        `add_start_token=True` default).
    """
    import torch

    global _perplexity_model, _perplexity_tokenizer, _perplexity_device
    if _perplexity_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _perplexity_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _perplexity_model = AutoModelForCausalLM.from_pretrained("gpt2")
        _perplexity_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _perplexity_model.to(_perplexity_device)
        _perplexity_model.eval()

    token_ids = _perplexity_tokenizer(text, add_special_tokens=False)["input_ids"]
    token_ids = [_perplexity_tokenizer.eos_token_id] + token_ids[:1023]
    input_ids = torch.tensor([token_ids], device=_perplexity_device)

    with torch.no_grad():
        loss = _perplexity_model(input_ids, labels=input_ids).loss

    return torch.exp(loss).item()


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
