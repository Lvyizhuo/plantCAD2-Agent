"""Zero-shot variant effect scoring.

Extracted and adapted from src/zero_shot_score.py.
Computes log-likelihood ratios (LLR) for variant alleles by masking
the target position and comparing predicted probabilities.
"""

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForMaskedLM, AutoTokenizer

MAX_SEQ_LENGTH = 8192
NUCLEOTIDES = ["A", "C", "G", "T"]


def score_variant(
    model: AutoModelForMaskedLM,
    tokenizer: AutoTokenizer,
    sequence: str,
    position: int,
    ref_allele: str,
    alt_alleles: list[str],
    device: str = "cuda:0",
) -> dict:
    """Score variant effect using zero-shot masked prediction.

    Masks the specified position, runs inference, and computes
    log(P_alt / P_ref) for each alternative allele.

    Args:
        model: The loaded AutoModelForMaskedLM.
        tokenizer: The loaded tokenizer.
        sequence: Full DNA context sequence containing the variant.
        position: 0-based position of the variant in the sequence.
        ref_allele: Reference allele (A/C/G/T).
        alt_alleles: List of alternative alleles to score.
        device: Device for inference.

    Returns:
        Dict with:
            - scores: dict mapping alt_allele -> LLR score
            - ref_prob: probability of the reference allele
            - alt_probs: dict mapping alt_allele -> probability

    Interpretation:
        LLR > 0: variant is more likely than reference (possibly deleterious)
        LLR < 0: variant is less likely than reference (possibly neutral/benign)

    Raises:
        ValueError: If sequence length exceeds maximum or position is out of range.
    """
    if len(sequence) > MAX_SEQ_LENGTH:
        raise ValueError(
            f"Sequence length {len(sequence)} exceeds maximum {MAX_SEQ_LENGTH}bp"
        )
    if position < 0 or position >= len(sequence):
        raise ValueError(
            f"Position {position} out of range [0, {len(sequence)})"
        )

    ref_allele = ref_allele.upper()
    alt_alleles = [a.upper() for a in alt_alleles]

    # Tokenize
    encoding = tokenizer.encode_plus(
        sequence,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    input_ids = encoding["input_ids"].to(device)

    # Mask the target position
    input_ids[0, position] = tokenizer.mask_token_id

    # Forward pass
    with torch.inference_mode():
        outputs = model(input_ids=input_ids)

    # Extract logits at the masked position for ACGT tokens
    acgt_indices = [tokenizer.get_vocab()[nc.lower()] for nc in NUCLEOTIDES]
    logits = outputs.logits[0, position, acgt_indices]
    probs = torch.nn.functional.softmax(logits.cpu(), dim=0).numpy()

    # Build probability maps
    nuc_prob = {nuc: float(probs[i]) for i, nuc in enumerate(NUCLEOTIDES)}

    ref_prob = nuc_prob[ref_allele]
    scores = {}
    alt_probs = {}
    for alt in alt_alleles:
        alt_p = nuc_prob[alt]
        alt_probs[alt] = alt_p
        if ref_prob > 0 and alt_p > 0:
            scores[alt] = float(np.log(alt_p / ref_prob))
        else:
            scores[alt] = float("inf") if alt_p > 0 else float("-inf")

    logger.debug(
        f"Variant scoring at pos {position}: "
        f"ref={ref_allele}({ref_prob:.4f}), "
        f"scores={scores}"
    )

    return {
        "scores": scores,
        "ref_prob": ref_prob,
        "alt_probs": alt_probs,
    }
