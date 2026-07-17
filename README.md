# Zero-Shot Text Restoration via Masked Diffusion Models

An ablation study comparing an **autoregressive** language model and a **masked diffusion language model (MDLM)** on the task of zero-shot text restoration: filling in masked/corrupted spans of a document without any task-specific fine-tuning.

## Motivation

Autoregressive (AR) language models generate text left-to-right, which makes controlled infilling awkward: they rely on Fill-in-the-Middle (FIM) prompting tricks to condition on both a prefix and a suffix. Masked diffusion language models instead generate all tokens jointly through iterative denoising, making bidirectional infilling a native operation rather than a prompting workaround.

This project asks: **does that structural advantage translate into better zero-shot restoration quality**, and how do the two paradigms trade off against corruption type, corruption severity, and inference cost?

## Method

Text is corrupted with two strategies, then restored by each model and scored against the original:

- **Corruption strategies** ([src/data.py](src/data.py))
  - `RandomTokenMasking` — independently masks a random subset of individual words.
  - `SpanMasking` — masks a single contiguous run of words (a structured gap).
- **Restoration models** ([src/models.py](src/models.py))
  - `QwenFIMRestorer` — [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B) via Fill-in-the-Middle prompting (AR baseline).
  - `MDLMRestorer` — [kuleshov-group/mdlm-owt](https://huggingface.co/kuleshov-group/mdlm-owt), a masked diffusion LM restored via iterative denoising ([src/mdlm_utils.py](src/mdlm_utils.py)), with the number of denoising steps as an ablation axis.
- **Evaluation** ([src/eval.py](src/eval.py)) — ROUGE-L F1, BERTScore (precision/recall/F1), and perplexity (GPT-2), plus wall-clock latency.

The full ablation grid varies masking ratio (0.1 / 0.3 / 0.5), corruption type (span / random token), and — for MDLM — the number of denoising timesteps, over a sample of [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext).

## Repository structure

```
src/
  data.py             # dataset loading + corruption strategies
  models.py           # Restorer wrappers (Qwen FIM, MDLM)
  mdlm_utils.py        # MDLM denoising loop, mask-token/token-alignment helpers
  flash_attn_shim.py  # pure-PyTorch stand-in for flash-attn (T4-compatible)
  eval.py             # ROUGE-L / BERTScore / perplexity computation
tests/                # unit tests, run entirely against mock/toy data
run_experiment.ipynb  # Colab entrypoint: runs the full ablation study with real models
data/experiment_results/  # exported results (CSV, JSON) and plots from the ablation run
```

## Getting started

```bash
pip install -r requirements.txt
pytest
```

The test suite and all local development run exclusively against dummy/mock inference and a small in-memory toy dataset — no model weights or datasets are downloaded. Real inference (Qwen2.5-Coder, MDLM, real WikiText-103, real BERTScore/perplexity) only happens inside [run_experiment.ipynb](run_experiment.ipynb), intended to run on Colab with a GPU (`use_mock=False`).

## Results

Aggregated results from the full ablation run (2,400 restoration/evaluation records, spanning both models, both corruption types, and three masking ratios) are in [data/experiment_results/](data/experiment_results/):

- `ablation_results.csv` / `ablation_results.json` — per-record scores (ROUGE-L, BERTScore, perplexity, latency) and configuration.
- `model_headline_summary.png` — overall AR vs. MDLM comparison.
- `corruption_type_comparison.png` — span vs. random-token masking.
- `rouge_l_f1_vs_ratio.png`, `bertscore_f1_vs_ratio.png` — quality degradation as masking ratio increases.
- `mdlm_timesteps_tradeoff.png` — MDLM quality vs. number of denoising steps (inference-cost trade-off).
- `perplexity_distribution.png` — fluency of restored text.

## License

Released under the [MIT License](LICENSE).
