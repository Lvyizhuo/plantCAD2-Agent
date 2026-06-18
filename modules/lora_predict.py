"""LoRA-based functional prediction.

Adapted from src/lora_fine_tune.py.
Loads a base classification model with a LoRA adapter overlay,
tokenizes the input sequence directly, and runs inference.
"""

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel


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
        For binary classification:
            {"prediction": "POSITIVE"/"NEGATIVE", "probability": float}
        For multi-label classification:
            {"prediction": "MULTI_LABEL", "probabilities": list[float], "num_labels": int}
        For regression:
            {"prediction": float}

    Raises:
        ValueError: If task_type is unsupported.
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

    # Forward pass (attention_mask filtered by the wrapped forward in model_loader)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    if task_type == "classification":
        num_labels = logits.shape[-1]
        if num_labels > 2:
            # Multi-label classification (e.g. cell_type with 92 labels)
            probs = torch.sigmoid(logits.cpu()).numpy()[0]
            logger.debug(f"Multi-label prediction: {num_labels} labels")
            return {
                "prediction": "MULTI_LABEL",
                "probabilities": [float(p) for p in probs],
                "num_labels": num_labels,
            }
        else:
            # Binary classification
            probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()[0]
            positive_prob = float(probs[1])
            prediction = "POSITIVE" if positive_prob >= 0.5 else "NEGATIVE"
            logger.debug(f"Binary prediction: {prediction} (prob={positive_prob:.4f})")
            return {
                "prediction": prediction,
                "probability": positive_prob,
            }
    elif task_type == "regression":
        predicted_value = float(logits.cpu().numpy().squeeze())
        logger.debug(f"Regression prediction: {predicted_value:.4f}")
        return {
            "prediction": predicted_value,
        }
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
