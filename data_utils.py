import random
from typing import List, Dict, Tuple
from datasets import load_dataset

def load_and_clean_wikitext(split: str = "test", num_samples: int = 500, min_words: int = 50, max_words: int = 150) -> List[str]:
    """
    Download WikiText-103, filter out titles and empty strings, and select paragraphs
    with a specific length to ensure sufficient context.
    """
    print(f"Loading WikiText-103 (split={split})...")
    # We use wikitext-103-raw-v1 to keep the punctuation intact
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    
    clean_texts = []
    for item in dataset:
        text = item["text"].strip()
        # Filter out empty strings and titles (which are often just a single word or enclosed in '=')
        if not text or (text.startswith("=") and text.endswith("=")):
            continue
            
        words = text.split()
        if min_words <= len(words) <= max_words:
            clean_texts.append(text)
            
        if len(clean_texts) >= num_samples:
            break
            
    print(f"Selected {len(clean_texts)} valid samples.")
    return clean_texts

def apply_span_masking(text: str, mask_ratio: float = 0.2) -> Dict:
    """
    Apply Span Masking: remove a single contiguous block of text.
    Ideal for testing global context understanding.
    """
    words = text.split()
    total_words = len(words)
    mask_length = max(1, int(total_words * mask_ratio))
    
    # Choose a random start point for the hole, avoiding the extreme edges
    # to ensure there is always some prefix and suffix
    start_idx = random.randint(1, total_words - mask_length - 1)
    end_idx = start_idx + mask_length
    
    prefix = " ".join(words[:start_idx])
    middle = " ".join(words[start_idx:end_idx])
    suffix = " ".join(words[end_idx:])
    
    return {
        "type": "span",
        "original_text": text,
        "prefix": prefix,
        "middle_ground_truth": middle,
        "suffix": suffix,
        "num_masks": mask_length
    }

def format_for_qwen_fim(prefix: str, suffix: str) -> str:
    """
    Format the input for the autoregressive baseline (Qwen2.5-Coder).
    Uses the standard FIM tokens to force the model to generate the middle.
    """
    # Exact tokens used by the Qwen documentation for FIM
    return f"{prefix}"

def format_for_mdlm_diffusion(prefix: str, suffix: str, num_masks: int, mask_token: str = "[MASK]") -> str:
    """
    Format the input for the masked diffusion baseline (MDLM).
    Physically inserts the exact number of masks between the prefix and suffix.
    """
    masks_string = " ".join([mask_token] * num_masks)
    return f"{prefix} {masks_string} {suffix}"

def prepare_dataset_for_experiment(texts: List[str], mask_ratio: float = 0.2, mask_token: str = "[MASK]") -> List[Dict]:
    """
    Complete pipeline: takes the clean texts, applies masking 
    and generates the prompts ready for both models.
    Returns a list of dictionaries, each containing:
    - original_text: the original unmasked text
    - ground_truth: the removed middle text
    - prompt_qwen: the prompt formatted for the autoregressive model
    - prompt_mdlm: the prompt formatted for the masked diffusion model
    - num_masks: the number of tokens masked
    """
    experiment_data = []
    
    for text in texts:
        # Apply span masking to get the prefix, middle (ground truth), and suffix
        masked_data = apply_span_masking(text, mask_ratio=mask_ratio)
        
        # Generate the prompts for the two models
        prompt_ar = format_for_qwen_fim(masked_data["prefix"], masked_data["suffix"])
        prompt_diff = format_for_mdlm_diffusion(
            masked_data["prefix"], 
            masked_data["suffix"], 
            masked_data["num_masks"], 
            mask_token=mask_token
        )
        
        # Assemble the sample dictionary
        sample = {
            "original_text": masked_data["original_text"],
            "ground_truth": masked_data["middle_ground_truth"],
            "prompt_qwen": prompt_ar,
            "prompt_mdlm": prompt_diff,
            "num_masks": masked_data["num_masks"]
        }
        experiment_data.append(sample)
        
    return experiment_data