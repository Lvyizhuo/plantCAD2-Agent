"""Masked token prediction at specified positions.

Extracted and adapted from notebooks/examples.ipynb.
Masks each target position independently and predicts
the ACGT probability distribution.
"""

import logging

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 8192
NUCLEOTIDES = ["A", "C", "G", "T"]


def masked_predict(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sequence: str,
    positions: list[int],
    device: str = "cuda:0",
) -> dict:
    """Predict nucleotide probabilities at specified positions.

    For each position, the nucleotide is replaced with [MASK],
    and the model predicts the probability of each ACGT nucleotide.

    Args:
        model: The loaded AutoModelForMaskedLM.
        tokenizer: The loaded tokenizer.
        sequence: DNA sequence string.
        positions: List of 0-based positions to mask and predict.
        device: Device for inference.

    Returns:
        Dict with 'predictions' mapping each position to {A, C, G, T} probs.
    """
    if len(sequence) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds maximum {MAX_SEQ_LENGTH}bp"
        )

    # Validate positions
    for pos in positions:
        if pos < 0 or pos >= len(sequence):
            raise ValueError(
                f"Position {pos} out of range [0, {len(sequence)})"
            )

    # Tokenize once
    encoding = tokenizer.encode_plus(
        sequence,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoding["input_ids"].to(device)

    acgt_indices = [tokenizer.get_vocab()[nc.lower()] for nc in NUCLEOTIDES]
    predictions = {}

    # Process each position: mask and predict
    # Note: for efficiency with many positions, we could batch them,
    # but per the PRD we handle single sequences.
    for pos in positions:
        # Clone to avoid modifying the original
        masked_ids = input_ids.clone()
        masked_ids[0, pos] = tokenizer.mask_token_id

        with torch.inference_mode():
            outputs = model(input_ids=masked_ids)

        logits = outputs.logits[0, pos, acgt_indices]
        probs = torch.nn.functional.softmax(logits.cpu(), dim=0).numpy()

        predictions[str(pos)] = {
            nuc: float(probs[i]) for i, nuc in enumerate(NUCLEOTIDES)
        }

    return {"predictions": predictions}
