"""Model wrappers for zero-shot text restoration.

Provides:
    - A common `Restorer` interface (mirroring `Corruptor` in src/data.py)
      that consumes a `CorruptionResult` (src/data.py) and produces a
      `RestorationResult`.
    - `QwenFIMRestorer`: Qwen2.5-Coder via Fill-in-the-Middle prompting
      (autoregressive baseline).
    - `MDLMRestorer`: zero-shot Masked Diffusion Language Model restoration
      (non-autoregressive baseline).
    - A dispatcher (`build_restorer`) that defaults to mock/dummy inference,
      matching `load_dataset_texts`'s "safe by default" philosophy.

Restoration outputs are consumed by evaluation code (src/eval.py, Phase 3)
for ROUGE-L, BERTScore, and perplexity computation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from src.data import CorruptionResult


@dataclass
class RestorationResult:
    """Result of restoring masked content with a text-restoration model.

    `restored_spans` holds only the fill-in content for each mask, aligned
    positionally with `corruption.ground_truth_spans` and
    `corruption.mask_positions` (same list length and order):
    `len(restored_spans) == len(corruption.ground_truth_spans) ==
    len(corruption.mask_positions)`. This lets evaluation code (src/eval.py)
    zip the three lists directly for per-span metric computation.
    """

    corruption: CorruptionResult
    restored_text: str
    restored_spans: list[str]
    model_name: str
    is_mock: bool
    inference_config: dict[str, object]
    prompt_used: Optional[str]
    latency_seconds: float


class Restorer(ABC):
    """Common interface for text-restoration model wrappers."""

    def __init__(self, model_name: str, use_mock: bool = True) -> None:
        """Initialize the restorer.

        Args:
            model_name: identifier for the underlying model/checkpoint,
                recorded in `RestorationResult.model_name`.
            use_mock: if True (default), `restore` never loads a real
                model and instead produces deterministic dummy output.
                Set False only in the Colab notebook (Phase 4) or explicit
                opt-in scripts, per the project's local-testing rules.
        """
        self.model_name = model_name
        self.use_mock = use_mock

    @abstractmethod
    def restore(self, corruption: CorruptionResult) -> RestorationResult:
        """Restore masked content in `corruption.corrupted_text`.

        Args:
            corruption: a `CorruptionResult` produced by a `Corruptor`.

        Returns:
            A `RestorationResult` with restored text/spans and metadata.
        """
        raise NotImplementedError

    def unload(self) -> None:
        """Release any cached real-model resources held by this restorer.

        Real restorers cache their loaded model/tokenizer across
        `restore()` calls (see `QwenFIMRestorer`/`MDLMRestorer`) so a
        multi-text ablation loop doesn't reload a multi-GB checkpoint per
        text. Call this before switching to a different model type in
        the same Colab session, so GPU memory is actually freed rather
        than held by PyTorch's caching allocator. No-op by default (and
        for restorers still in mock mode, which never load anything).
        """


def _reinsert_spans(corruption: CorruptionResult, restored_spans: list[str]) -> str:
    """Rebuild full restored text by reinserting `restored_spans` into
    `corruption.original_text`'s word list at `corruption.mask_positions`.

    Args:
        corruption: the source `CorruptionResult`.
        restored_spans: fill-in content, one per mask position, in order.

    Returns:
        The reconstructed text with each masked word range replaced by its
        corresponding restored span.
    """
    words = corruption.original_text.split()
    result_words: list[str] = []
    prev_end = 0
    for (start, end), span in zip(corruption.mask_positions, restored_spans):
        result_words.extend(words[prev_end:start])
        result_words.append(span)
        prev_end = end
    result_words.extend(words[prev_end:])
    return " ".join(result_words)


class QwenFIMRestorer(Restorer):
    """Autoregressive restoration via Qwen2.5-Coder Fill-in-the-Middle prompting.

    For `SpanMasking` corruptions (a single contiguous mask position), a
    single standard FIM prompt is built and one fill is generated. For
    `RandomTokenMasking` corruptions (multiple independent single-word mask
    positions), each mask position is restored via its own FIM call, built
    against the true surrounding context from `corruption.original_text`
    with only that one position blanked -- not against `corrupted_text`,
    which has every position masked simultaneously. This keeps every FIM
    call a standard single-hole prompt rather than inventing a nonstandard
    multi-hole FIM variant, at the cost of one model call per masked word.
    """

    FIM_PREFIX_TOKEN = "<|fim_prefix|>"
    FIM_SUFFIX_TOKEN = "<|fim_suffix|>"
    FIM_MIDDLE_TOKEN = "<|fim_middle|>"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B",
        use_mock: bool = True,
        max_new_tokens: int = 64,
        temperature: float = 0.2,
        context_window_words: Optional[int] = 75,
        qwen_batch_size: int = 16,
    ) -> None:
        """Initialize the Qwen FIM restorer.

        Args:
            model_name: HuggingFace checkpoint identifier.
            use_mock: see `Restorer.__init__`.
            max_new_tokens: generation cap used by the real inference path.
            temperature: sampling temperature used by the real inference path.
            context_window_words: max number of words kept on each side of a
                mask position when building its FIM prompt (see
                `_build_fim_prompts`). Bounds prompt length -- and thus
                latency/memory -- regardless of document length, instead of
                feeding the entire prefix/suffix. `None` disables windowing
                (uses the full prefix/suffix, the original behavior).
            qwen_batch_size: max number of FIM prompts generated in a single
                batched `model.generate()` call in the real inference path
                (see `_restore_real`). `RandomTokenMasking` can produce many
                independent single-word mask positions per document, each
                needing its own FIM prompt -- batching multiple prompts per
                `generate()` call collapses those into far fewer forward
                passes, while chunking at this size bounds peak GPU memory
                instead of batching everything in one call.
        """
        super().__init__(model_name=model_name, use_mock=use_mock)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.context_window_words = context_window_words
        self.qwen_batch_size = qwen_batch_size
        self._model = None
        self._tokenizer = None

    def unload(self) -> None:
        """Release the cached Qwen model/tokenizer and free GPU memory."""
        self._model = None
        self._tokenizer = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def restore(self, corruption: CorruptionResult) -> RestorationResult:
        start_time = time.perf_counter()
        prompts = self._build_fim_prompts(corruption)

        if self.use_mock:
            restored_spans = self._restore_mock(corruption, prompts)
        else:
            restored_spans = self._restore_real(corruption, prompts)

        latency = time.perf_counter() - start_time
        return RestorationResult(
            corruption=corruption,
            restored_text=_reinsert_spans(corruption, restored_spans),
            restored_spans=restored_spans,
            model_name=self.model_name,
            is_mock=self.use_mock,
            inference_config={
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "context_window_words": self.context_window_words,
                "qwen_batch_size": self.qwen_batch_size,
            },
            prompt_used=prompts[0] if prompts else None,
            latency_seconds=latency,
        )

    def _build_fim_prompts(self, corruption: CorruptionResult) -> list[str]:
        """Build one FIM prompt per mask position.

        For each `(start, end)` in `corruption.mask_positions`, the prefix
        is the joined words of `corruption.original_text` before `start`
        and the suffix is the joined words after `end`, so the model only
        ever sees one blanked region per call -- the multi-mask case
        (`RandomTokenMasking`) is handled as N independent single-hole FIM
        prompts rather than one combined multi-hole prompt. Prefix/suffix
        are each capped to `self.context_window_words` words (nearest to
        the mask) when set, rather than using the entire document -- this
        bounds prompt length/latency/memory independent of document
        length, matching standard FIM practice (Codex/StarCoder-style
        local context windows around the gap, not whole-document context).

        Args:
            corruption: the source `CorruptionResult`.

        Returns:
            A list of FIM-formatted prompt strings, one per mask position,
            in the same order as `corruption.mask_positions`.
        """
        words = corruption.original_text.split()
        prompts: list[str] = []
        for start, end in corruption.mask_positions:
            prefix_words = words[:start]
            suffix_words = words[end:]
            if self.context_window_words is not None:
                prefix_words = prefix_words[-self.context_window_words :]
                suffix_words = suffix_words[: self.context_window_words]
            prefix = " ".join(prefix_words)
            suffix = " ".join(suffix_words)
            prompts.append(
                f"{self.FIM_PREFIX_TOKEN}{prefix}"
                f"{self.FIM_SUFFIX_TOKEN}{suffix}"
                f"{self.FIM_MIDDLE_TOKEN}"
            )
        return prompts

    def _restore_mock(
        self, corruption: CorruptionResult, prompts: list[str]
    ) -> list[str]:
        """Produce deterministic fake fills without loading any model.

        Args:
            corruption: the source `CorruptionResult`.
            prompts: FIM prompts built by `_build_fim_prompts` (unused by
                the mock path beyond matching its length/order).

        Returns:
            A list of strings of the form `"<mock:{ground_truth}>"`, one
            per mask position, so tests can assert exact expected content
            without loading a real model.
        """
        return [f"<mock:{span}>" for span in corruption.ground_truth_spans]

    def _restore_real(
        self, corruption: CorruptionResult, prompts: list[str]
    ) -> list[str]:
        """Run real Qwen2.5-Coder generation via `transformers`.

        Args:
            corruption: the source `CorruptionResult`.
            prompts: FIM prompts built by `_build_fim_prompts`.

        Returns:
            A list of generated fill strings, one per prompt, in the same
            order as `prompts`.

        Note:
            Lazily imports `transformers` and loads real model weights on
            the first call, caching them on `self` for reuse by
            subsequent calls -- do NOT call this in local dev or tests,
            it will attempt to download and load a multi-GB checkpoint.
            Intended only for the Colab notebook (Phase 4) or explicit
            opt-in scripts. Call `unload()` to free the cached model.

            Prompts are processed in batches of `self.qwen_batch_size`,
            left-padded so every row in a batch shares the same prompt
            length -- required for `RandomTokenMasking`, which can produce
            many independent single-word mask positions (and thus many FIM
            prompts) per document; batching collapses those into far fewer
            `generate()` calls instead of one sequential call per mask.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.padding_side = "left"
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype="auto", device_map="auto"
            )
        tokenizer = self._tokenizer
        model = self._model

        restored_spans: list[str] = []
        for batch_start in range(0, len(prompts), self.qwen_batch_size):
            batch = prompts[batch_start : batch_start + self.qwen_batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(
                model.device
            )
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0.0,
            )
            prompt_length = inputs["input_ids"].shape[1]
            for row in output_ids:
                generated = tokenizer.decode(
                    row[prompt_length:], skip_special_tokens=True
                )
                restored_spans.append(generated)
        return restored_spans


class MDLMRestorer(Restorer):
    """Zero-shot restoration via a Masked Diffusion Language Model.

    Conceptually performs iterative denoising over `num_timesteps` steps,
    resolving `corrupted_text`'s mask tokens progressively. `num_timesteps`
    is an ablation axis: inference efficiency (fewer steps = faster,
    typically lower quality).
    """

    def __init__(
        self,
        model_name: str = "kuleshov-group/mdlm-owt",
        use_mock: bool = True,
        num_timesteps: int = 16,
        context_window_words: Optional[int] = 75,
        eps: float = 1e-3,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize the MDLM restorer.

        Args:
            model_name: checkpoint identifier for the MDLM backend.
            use_mock: see `Restorer.__init__`.
            num_timesteps: number of denoising steps (ablation axis:
                "inference efficiency"). Higher = more steps = typically
                higher quality, slower inference.
            context_window_words: max number of words kept on each side of
                the outermost mask positions when building the sequence to
                denoise (see `_restore_real`). Bounds sequence length/
                latency/memory independent of document length, and keeps
                context comparable to `QwenFIMRestorer`'s own windowing for
                a fair ablation comparison. `None` disables windowing (uses
                the entire document, the original behavior).
            eps: log-linear noise schedule floor forwarded to
                `src.mdlm_utils.denoise`. See that function's docstring.
            seed: if set, seeds a `torch.Generator` used for `denoise`'s
                reverse-diffusion sampling, making an otherwise-stochastic
                restoration reproducible across runs. `None` (default)
                samples non-reproducibly.
        """
        super().__init__(model_name=model_name, use_mock=use_mock)
        self.num_timesteps = num_timesteps
        self.context_window_words = context_window_words
        self.eps = eps
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self._device = None

    def unload(self) -> None:
        """Release the cached MDLM model/tokenizer and free GPU memory."""
        self._model = None
        self._tokenizer = None
        self._device = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def restore(self, corruption: CorruptionResult) -> RestorationResult:
        start_time = time.perf_counter()

        if self.use_mock:
            restored_spans = self._restore_mock(corruption)
        else:
            restored_spans = self._restore_real(corruption)

        latency = time.perf_counter() - start_time
        return RestorationResult(
            corruption=corruption,
            restored_text=_reinsert_spans(corruption, restored_spans),
            restored_spans=restored_spans,
            model_name=self.model_name,
            is_mock=self.use_mock,
            inference_config={
                "num_timesteps": self.num_timesteps,
                "context_window_words": self.context_window_words,
                "eps": self.eps,
                "seed": self.seed,
            },
            prompt_used=None,
            latency_seconds=latency,
        )

    def _restore_mock(self, corruption: CorruptionResult) -> list[str]:
        """Produce deterministic fake fills without loading any model.

        The mock output encodes `self.num_timesteps` into the fake fill
        string so that ablation pipeline/plotting code exercising the
        "inference efficiency" axis can be tested end-to-end locally
        without a real diffusion model.

        Args:
            corruption: the source `CorruptionResult`.

        Returns:
            A list of strings of the form
            `"<mock:{ground_truth}:steps={num_timesteps}>"`, one per mask
            position.
        """
        return [
            f"<mock:{span}:steps={self.num_timesteps}>"
            for span in corruption.ground_truth_spans
        ]

    def _restore_real(self, corruption: CorruptionResult) -> list[str]:
        """Run real MDLM absorbing-state denoising via a custom sampling loop.

        Loads the real `kuleshov-group/mdlm-owt` checkpoint via
        `transformers.AutoModelForMaskedLM` with `trust_remote_code=True`
        (its own reference repository only ships an unconditional
        sampler), then resolves masked token ranges with
        `src.mdlm_utils.denoise`, which runs MDLM's actual stochastic
        reverse-diffusion update (ported from that repo's `diffusion.py`)
        and, as a byproduct of that update rule, keeps all non-masked
        context tokens unchanged throughout the reverse process.

        The checkpoint's custom modeling code hard-imports `flash_attn`
        with no eager/sdpa fallback, but real `flash-attn` only supports
        Ampere-or-newer GPUs (not Colab's free-tier T4). `src.flash_attn_shim`
        installs a pure-PyTorch stand-in under the same module names
        before loading, so this runs on any GPU.

        When `self.context_window_words` is set, only a window of words
        around the outermost mask positions (see `corruption.mask_positions`)
        is tokenized/denoised, rather than the entire `original_text` --
        this bounds sequence length/latency independent of document length.
        For `SpanMasking` (a single contiguous mask) this always shrinks
        the sequence to roughly `2 * context_window_words` words. For
        `RandomTokenMasking`, since mask positions are scattered across the
        whole document by construction, the window still spans from the
        first to the last masked word plus margin -- it bounds the
        irrelevant head/tail of the document, but doesn't shrink the
        interior, since all masks are denoised jointly in one pass (unlike
        `QwenFIMRestorer`, which treats each mask position independently).

        Args:
            corruption: the source `CorruptionResult`.

        Returns:
            A list of restored fill strings, one per mask position, in
            the same order as `corruption.mask_positions`.

        Note:
            Lazily imports `torch`/`transformers` and loads real model
            weights on the first call, caching them on `self` for reuse
            by subsequent calls -- do NOT call this in local dev or
            tests. Intended only for the Colab notebook (Phase 4) or
            explicit opt-in scripts. Call `unload()` to free the cached
            model.
        """
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        from src.flash_attn_shim import install as install_flash_attn_shim
        from src.mdlm_utils import (
            align_mask_positions_to_tokens,
            compute_word_window,
            denoise,
            resolve_mask_token_id,
        )

        if self._model is None:
            install_flash_attn_shim()
            self._tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._model = AutoModelForMaskedLM.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()
        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        words = corruption.original_text.split()
        if self.context_window_words is not None:
            window_start, window_end = compute_word_window(
                len(words), corruption.mask_positions, self.context_window_words
            )
        else:
            window_start, window_end = 0, len(words)
        windowed_text = " ".join(words[window_start:window_end])
        windowed_mask_positions = [
            (s - window_start, e - window_start) for s, e in corruption.mask_positions
        ]

        token_ranges = align_mask_positions_to_tokens(
            windowed_text, windowed_mask_positions, tokenizer
        )
        input_ids = tokenizer(windowed_text, return_tensors="pt")["input_ids"]
        mask_token_id = resolve_mask_token_id(model, tokenizer)
        for start, end in token_ranges:
            input_ids[0, start:end] = mask_token_id

        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=device).manual_seed(self.seed)

        resolved_ids = denoise(
            model=model,
            input_ids=input_ids,
            masked_token_ranges=token_ranges,
            mask_token_id=mask_token_id,
            num_timesteps=self.num_timesteps,
            device=device,
            eps=self.eps,
            generator=generator,
        )

        return [
            tokenizer.decode(resolved_ids[0, start:end], skip_special_tokens=True).strip()
            for start, end in token_ranges
        ]


def build_restorer(model_type: str, use_mock: bool = True, **kwargs: object) -> Restorer:
    """Construct a `Restorer` by name, defaulting to the mock inference path.

    Defaults to `use_mock=True` so that calling this function without
    thought never triggers accidental real model loading, per the
    project's local-testing rules.

    Args:
        model_type: one of "qwen_fim", "mdlm".
        use_mock: if True (default), the returned restorer's `restore()`
            never loads a real model. Real model loading requires
            explicitly passing `use_mock=False`.
        **kwargs: forwarded to the chosen restorer's constructor (e.g.
            `num_timesteps` for MDLM, `max_new_tokens` for Qwen).

    Returns:
        A `Restorer` instance.

    Raises:
        ValueError: if `model_type` is not recognized.
    """
    if model_type == "qwen_fim":
        return QwenFIMRestorer(use_mock=use_mock, **kwargs)
    if model_type == "mdlm":
        return MDLMRestorer(use_mock=use_mock, **kwargs)
    raise ValueError(f"Unknown model_type: {model_type!r}")
