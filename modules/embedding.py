"""Embedding extraction for DNA sequences.

Extracted and adapted from notebooks/examples.ipynb.
Handles RCPS (reverse-complement parameter sharing) by splitting
the 2x-dim output into forward/reverse halves and averaging.
"""

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForMaskedLM, AutoTokenizer

# PlantCAD2 max context length
MAX_SEQ_LENGTH = 8192


def extract_embeddings(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sequence: str,
    device: str = "cuda:0",
    normalize: bool = True,
) -> dict:
    """Extract per-position embeddings from a DNA sequence.

    The model outputs hidden states of dimension 2 * d_model due to RCPS
    (reverse-complement parameter sharing). This function splits the output
    into forward and reverse-complement halves, flips the reverse half,
    and averages them to produce d_model-dimensional embeddings.

    Args:
        model: The loaded AutoModelForMaskedLM.
        tokenizer: The loaded tokenizer.
        sequence: DNA sequence string (ACGT characters).
        device: Device for inference.
        normalize: Whether to L2-normalize the output embeddings.

    Returns:
        Dict with:
            - embeddings: numpy array of shape (seq_len, d_model)
            - shape: tuple (seq_len, d_model)
            - sequence_length: int

    Raises:
        ValueError: If sequence length exceeds maximum.
    """
    if len(sequence) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds maximum {MAX_SEQ_LENGTH}bp"
        )

    # Tokenize
    encoding = tokenizer.encode_plus(
        sequence,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoding["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    # Forward pass with hidden states
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    # Last hidden layer: shape (1, seq_len, 2 * d_model) due to RCPS
    embeddings = outputs.hidden_states[-1]

    # Move to CPU and convert to float32
    embeddings = embeddings.to(torch.float32).cpu().numpy()

    # Split RCPS output: first half = forward, second half = reverse complement
    hidden_size = embeddings.shape[-1] // 2
    forward = embeddings[..., 0:hidden_size]
    reverse = embeddings[..., hidden_size:]
    # Flip the reverse complement along the sequence dimension
    reverse = reverse[..., ::-1]
    # Average forward and reverse-complement embeddings
    averaged = (forward + reverse) / 2

    # Remove batch dimension: (1, seq_len, d_model) -> (seq_len, d_model)
    averaged = averaged.squeeze(0)

    if normalize:
        norms = np.linalg.norm(averaged, axis=-1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        averaged = averaged / norms

    logger.debug(f"Extracted embeddings: shape={averaged.shape}, normalized={normalize}")

    return {
        "embeddings": averaged,
        "shape": averaged.shape,
        "sequence_length": seq_len,
    }
