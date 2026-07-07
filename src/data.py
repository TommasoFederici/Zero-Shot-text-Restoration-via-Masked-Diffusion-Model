"""Data pipeline for the zero-shot text restoration ablation study.

Provides:
    - WikiText-103 loading (`load_wikitext103`) and a toy in-memory dataset
      (`load_toy_dataset`) for local development, plus a dispatcher
      (`load_dataset_texts`).
    - Word-boundary-preserving text corruption strategies
      (`RandomTokenMasking`, `SpanMasking`) implementing the `Corruptor`
      interface, producing `CorruptionResult` objects consumable uniformly
      by model wrappers (src/models.py) and evaluation code (src/eval.py).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CorruptionResult:
    """Result of applying a corruption strategy to a piece of text.

    Positions in `mask_positions` are word indices (from whitespace
    splitting) over the word list of `original_text`, not character
    offsets or subword token indices. This keeps the result tokenizer
    agnostic: downstream model wrappers can re-tokenize `original_text`
    themselves and map word indices to their own subword indices.
    """

    original_text: str
    corrupted_text: str
    ground_truth_spans: list[str]
    mask_positions: list[tuple[int, int]]
    masking_ratio_requested: float
    masking_ratio_actual: float
    corruption_type: str
    mask_token: str
    num_words_total: int
    seed: Optional[int]


def _split_words(text: str) -> list[str]:
    """Split text into whitespace-delimited word tokens.

    Each token retains its exact surface form (attached punctuation,
    casing), so `" ".join(_split_words(text))` reconstructs `text` up to
    whitespace normalization (multiple spaces/newlines collapse to single
    spaces). This whitespace-level granularity is what guarantees "exact
    boundary preservation": masking never cuts inside a word.
    """
    return text.split()


class Corruptor(ABC):
    """Common interface for text corruption strategies used in the ablation study."""

    def __init__(self, mask_token: str = "<mask>", seed: Optional[int] = None) -> None:
        """Initialize the corruptor.

        Args:
            mask_token: placeholder string inserted in place of masked words.
            seed: seed for reproducible masking. If None, each call to
                `corrupt` uses non-deterministic randomness.
        """
        self.mask_token = mask_token
        self.seed = seed

    @abstractmethod
    def corrupt(self, text: str, ratio: float) -> CorruptionResult:
        """Corrupt `text` by masking approximately `ratio` fraction of words.

        Args:
            text: input text, assumed non-empty after stripping whitespace.
            ratio: fraction in (0, 1] of words to mask.

        Returns:
            A `CorruptionResult` describing the corrupted text and the
            ground-truth content that was removed.

        Raises:
            ValueError: if `text` is empty/whitespace-only or `ratio` is
                not in (0, 1].
        """
        raise NotImplementedError

    def _make_rng(self) -> random.Random:
        """Return a fresh `random.Random` instance seeded with `self.seed`."""
        return random.Random(self.seed)

    def _validate_and_prepare(self, text: str, ratio: float) -> tuple[list[str], int]:
        """Validate inputs and compute the word list and mask count.

        Args:
            text: input text.
            ratio: requested masking ratio.

        Returns:
            A tuple of (words, num_words_to_mask), where `num_words_to_mask`
            is clamped to be at least 1 and at most `len(words)`.

        Raises:
            ValueError: if `text` is empty/whitespace-only or `ratio` is
                not in (0, 1].
        """
        if not text.strip():
            raise ValueError("text is empty or whitespace-only")
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")
        words = _split_words(text)
        num_to_mask = max(1, min(len(words), round(ratio * len(words))))
        return words, num_to_mask


class RandomTokenMasking(Corruptor):
    """Masks a random subset of individual words, independently selected.

    Words are chosen uniformly at random without replacement; each masked
    word is replaced by its own `mask_token` occurrence (masked words are
    not collapsed even if two selected indices happen to be adjacent).
    This distinguishes random token masking from `SpanMasking`, where a
    single contiguous run is replaced by exactly one placeholder.
    """

    def corrupt(self, text: str, ratio: float) -> CorruptionResult:
        words, num_to_mask = self._validate_and_prepare(text, ratio)
        rng = self._make_rng()
        indices = sorted(rng.sample(range(len(words)), num_to_mask))

        ground_truth_spans = [words[i] for i in indices]
        mask_positions = [(i, i + 1) for i in indices]

        masked_index_set = set(indices)
        corrupted_words = [
            self.mask_token if i in masked_index_set else word
            for i, word in enumerate(words)
        ]

        return CorruptionResult(
            original_text=text,
            corrupted_text=" ".join(corrupted_words),
            ground_truth_spans=ground_truth_spans,
            mask_positions=mask_positions,
            masking_ratio_requested=ratio,
            masking_ratio_actual=num_to_mask / len(words),
            corruption_type="random_token",
            mask_token=self.mask_token,
            num_words_total=len(words),
            seed=self.seed,
        )


class SpanMasking(Corruptor):
    """Masks a single contiguous word span whose length approximates `ratio`.

    The entire span is replaced by exactly one `mask_token` occurrence,
    simulating a structured/contiguous corruption (e.g. a missing sentence
    fragment), as opposed to `RandomTokenMasking`'s independently masked
    words.
    """

    def __init__(
        self,
        mask_token: str = "<mask>",
        seed: Optional[int] = None,
        num_spans: int = 1,
    ) -> None:
        """Initialize the span masking corruptor.

        Args:
            mask_token: placeholder string inserted in place of the masked span.
            seed: seed for reproducible masking.
            num_spans: number of separate contiguous spans to mask. Only
                `num_spans=1` (a single contiguous corrupted region) is
                currently supported, matching the ablation study's
                "Random vs Span" corruption topology axis.

        Raises:
            NotImplementedError: if `num_spans` is not 1.
        """
        super().__init__(mask_token=mask_token, seed=seed)
        if num_spans != 1:
            raise NotImplementedError(
                "SpanMasking currently only supports num_spans=1"
            )
        self.num_spans = num_spans

    def corrupt(self, text: str, ratio: float) -> CorruptionResult:
        words, num_to_mask = self._validate_and_prepare(text, ratio)
        rng = self._make_rng()

        max_start = len(words) - num_to_mask
        start = rng.randint(0, max_start)
        end = start + num_to_mask

        ground_truth_spans = [" ".join(words[start:end])]
        mask_positions = [(start, end)]

        corrupted_words = words[:start] + [self.mask_token] + words[end:]

        return CorruptionResult(
            original_text=text,
            corrupted_text=" ".join(corrupted_words),
            ground_truth_spans=ground_truth_spans,
            mask_positions=mask_positions,
            masking_ratio_requested=ratio,
            masking_ratio_actual=num_to_mask / len(words),
            corruption_type="span",
            mask_token=self.mask_token,
            num_words_total=len(words),
            seed=self.seed,
        )


def load_toy_dataset() -> list[str]:
    """Return a small, fixed, in-memory list of example texts for local
    development and unit testing.

    Contains no external I/O or network calls, so it is safe to use in
    any test/dev context without downloading real data, per the project's
    local-testing rules.

    Returns:
        A short list of hand-written multi-sentence strings of varying
        length, including one very short string to exercise edge-case
        handling in the corruptors.
    """
    return [
        "The quick brown fox jumps over the lazy dog near the river.",
        "Short text example.",
        (
            "Machine learning models can generate coherent text given a prompt, "
            "but controlled infilling remains a challenging task for "
            "autoregressive systems."
        ),
        (
            "Diffusion models reformulate text generation as a discrete "
            "denoising process, which allows them to integrate hard "
            "constraints without additional training."
        ),
        "Zero-shot restoration is hard.",
        (
            "Researchers evaluate restoration quality using ROUGE-L, "
            "BERTScore, and perplexity to compare autoregressive and "
            "diffusion-based approaches across several masking ratios."
        ),
    ]


def _passes_length_filter(text: str, min_words: int, max_words: Optional[int]) -> bool:
    """Check whether a candidate WikiText-103 line should be kept.

    Extracted as a pure function so the filtering rules can be unit-tested
    without a real dataset download (see `tests/test_data.py`).

    Args:
        text: a stripped candidate line.
        min_words: minimum whitespace-token count to keep the line (drops
            blank lines and short fragments).
        max_words: maximum whitespace-token count to keep the line, or
            None for no upper bound. Lines are dropped rather than
            truncated, so every retained sample stays a complete
            document -- truncating could cut a paragraph mid-sentence and
            feed a corrupted/incomplete document into the corruption
            pipeline. Bounding document length also keeps per-sample
            latency comparable across the ablation grid, and bounds
            `MDLMRestorer`'s worst case: for `RandomTokenMasking`,
            `compute_word_window` (src/mdlm_utils.py) spans from the
            first to the last masked word without shrinking the
            interior, so on an unbounded document that window -- and the
            cost of each denoising step -- can approach the full
            document length regardless of `context_window_words`.

    Returns:
        True if the line should be kept.
    """
    word_count = len(text.split())
    if word_count < min_words:
        return False
    if max_words is not None and word_count > max_words:
        return False
    return not (text.startswith("=") and text.endswith("="))


def load_wikitext103(
    split: str = "train",
    max_samples: Optional[int] = 200,
    min_words: int = 10,
    max_words: Optional[int] = 500,
    cache_dir: Optional[Path] = None,
) -> list[str]:
    """Load a subset of WikiText-103 (raw) via HuggingFace `datasets`.

    Filters out empty/whitespace-only lines, short lines, markdown-style
    section headers (e.g. "= Title ="), and (by default) overly long
    lines -- see `_passes_length_filter`.

    Args:
        split: HuggingFace split name, e.g. "train", "validation", "test".
        max_samples: cap on the number of returned lines. None returns all
            matching lines; keep this small in local dev to avoid
            excessive memory/download.
        min_words: minimum whitespace-token count for a line to be kept.
        max_words: maximum whitespace-token count for a line to be kept,
            or None for no upper bound. Over-long lines are dropped, not
            truncated (see `_passes_length_filter` for the rationale).
        cache_dir: optional cache directory passed to `load_dataset`;
            defaults to HuggingFace's own cache location if None.

    Returns:
        A list of cleaned text strings, one per retained WikiText-103
        line/paragraph.

    Note:
        This function performs a real network call / download the first
        time it runs. Do not call it in local dev or test code — use
        `load_toy_dataset` instead. It is intended to be called only from
        the Colab notebook (Phase 4) or explicit opt-in scripts.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    texts = [line.strip() for line in dataset["text"]]
    texts = [t for t in texts if _passes_length_filter(t, min_words, max_words)]
    if max_samples is not None:
        texts = texts[:max_samples]
    return texts


def load_dataset_texts(use_toy: bool = True, **wikitext_kwargs) -> list[str]:
    """Dispatch to either the toy dataset or real WikiText-103.

    Defaults to the toy dataset so that calling this function without
    thought never triggers an accidental real download, per the project's
    local-testing rules. Real data must be requested explicitly.

    Args:
        use_toy: if True (default), return `load_toy_dataset()`. If False,
            return `load_wikitext103(**wikitext_kwargs)`.
        **wikitext_kwargs: forwarded to `load_wikitext103` when
            `use_toy=False`.

    Returns:
        A list of text strings.
    """
    return load_toy_dataset() if use_toy else load_wikitext103(**wikitext_kwargs)
