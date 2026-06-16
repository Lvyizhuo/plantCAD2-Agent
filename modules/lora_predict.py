"""LoRA-based functional prediction.

Adapted from src/lora_fine_tune.py.
Loads a base classification model with a LoRA adapter overlay,
tokenizes the input sequence directly, and runs inference.
"""

import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

# Token IDs for ACGT nucleotides in the tokenizer
NUCLEOTIDES = ["A", "C", "G", "T"]


def predict_with_lora(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    sequence: str,
    task_type: str,
    device: str = "cuda:0",
    max_length: int = 8192,
) -> dict:
    """Run prediction using a LoRA-adapted model.

    Args:
        model: PeftModel with LoRA adapter loaded.
        tokenizer: The loaded tokenizer.
        sequence: DNA sequence string.
        task_type: "classification" or "regression".
        device: Device for inference.
        max_length: Maximum sequence length for tokenization.

    Returns:
        For classification:
            {"prediction": "POSITIVE"/"NEGATIVE", "probability": float}
        For regression:
            {"prediction": float}
    """
    # Tokenize the sequence directly (not from parquet)
    encoding = tokenizer.encode_plus(
        sequence,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = encoding["input_ids"].to(device)

    # Forward pass
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    if task_type == "classification":
        probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()[0]
        # Binary classification: index 0 = NEGATIVE, index 1 = POSITIVE
        positive_prob = float(probs[1])
        prediction = "POSITIVE" if positive_prob >= 0.5 else "NEGATIVE"
        return {
            "prediction": prediction,
            "probability": positive_prob,
        }
    elif task_type == "regression":
        predicted_value = float(logits.cpu().numpy().squeeze())
        return {
            "prediction": predicted_value,
        }
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
