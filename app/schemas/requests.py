"""Pydantic request/response models for PlantCAD2 API.

Includes comprehensive input validation for DNA sequences and other parameters.
"""

from typing import Optional, Union
from pydantic import BaseModel, Field, field_validator

from app.config import settings


# --- Constants ---
VALID_NUCLEOTIDES = set("ACGTacgt")
MAX_SEQUENCE_LENGTH = settings.max_sequence_length  # 8192


def validate_dna_sequence(v: str) -> str:
    """Validate DNA sequence: non-empty, correct length, valid characters."""
    if not v or not v.strip():
        raise ValueError("Sequence must not be empty")

    v = v.strip().upper()

    if len(v) == 0:
        raise ValueError("Sequence must not be empty")

    if len(v) > MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Sequence length {len(v)} exceeds maximum {MAX_SEQUENCE_LENGTH}bp"
        )

    invalid_chars = set(v) - {"A", "C", "G", "T"}
    if invalid_chars:
        raise ValueError(
            f"Sequence contains invalid characters: {invalid_chars}. "
            f"Only A, C, G, T (case-insensitive) are allowed"
        )

    return v


# --- Embeddings ---

class EmbeddingRequest(BaseModel):
    sequence: str = Field(
        ...,
        description=f"DNA sequence (ACGT, max {MAX_SEQUENCE_LENGTH}bp)",
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    normalize: bool = Field(True, description="L2-normalize embeddings")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        return validate_dna_sequence(v)


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    shape: list[int]
    sequence_length: int


# --- Variant Score ---

class VariantScoreRequest(BaseModel):
    sequence: str = Field(
        ...,
        description=f"DNA context sequence containing the variant (max {MAX_SEQUENCE_LENGTH}bp)",
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    position: int = Field(..., ge=0, description="0-based variant position in the sequence")
    ref_allele: str = Field(
        ...,
        pattern="^[ACGTacgt]$",
        description="Reference allele (A/C/G/T)",
    )
    alt_alleles: list[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="List of alternative alleles to score (A/C/G/T, max 3)",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        return validate_dna_sequence(v)

    @field_validator("ref_allele")
    @classmethod
    def validate_ref_allele(cls, v: str) -> str:
        return v.upper()

    @field_validator("alt_alleles")
    @classmethod
    def validate_alt_alleles(cls, v: list[str]) -> list[str]:
        validated = []
        for alt in v:
            alt = alt.upper()
            if alt not in {"A", "C", "G", "T"}:
                raise ValueError(
                    f"Invalid alt allele '{alt}'. Must be one of A, C, G, T"
                )
            validated.append(alt)
        return validated

    @field_validator("position")
    @classmethod
    def validate_position_range(cls, v: int, info) -> int:
        # Note: Full position vs sequence length validation happens in the endpoint
        # since we need access to the sequence value
        if v < 0:
            raise ValueError("Position must be non-negative")
        return v


class VariantScoreResponse(BaseModel):
    scores: dict[str, float]
    ref_prob: float
    alt_probs: dict[str, float]


# --- LoRA Predict ---

VALID_TASKS = [
    "acr_arabidopsis",
    "acr_nine_species",
    "acr_cell_type",
    "expression_on_off",
    "expression_absolute",
    "translation_on_off",
    "translation_absolute",
]


class PredictRequest(BaseModel):
    sequence: str = Field(
        ...,
        description=f"DNA sequence (ACGT, max {MAX_SEQUENCE_LENGTH}bp)",
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    task: str = Field(
        ...,
        description=f"Task name: {', '.join(VALID_TASKS)}",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        return validate_dna_sequence(v)

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in VALID_TASKS:
            raise ValueError(
                f"Unknown task '{v}'. Must be one of: {', '.join(VALID_TASKS)}"
            )
        return v


class PredictResponse(BaseModel):
    task: str
    prediction: Union[str, float]
    probability: Optional[float] = None
    probabilities: Optional[list[float]] = None
    num_labels: Optional[int] = None


# --- Masked Predict ---

class MaskedPredictRequest(BaseModel):
    sequence: str = Field(
        ...,
        description=f"DNA sequence (ACGT, max {MAX_SEQUENCE_LENGTH}bp)",
        min_length=1,
        max_length=MAX_SEQUENCE_LENGTH,
    )
    positions: list[int] = Field(
        ...,
        min_length=1,
        max_length=100,  # Limit to prevent excessive computation
        description="List of 0-based positions to mask and predict (max 100)",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        return validate_dna_sequence(v)

    @field_validator("positions")
    @classmethod
    def validate_positions(cls, v: list[int]) -> list[int]:
        validated = []
        seen = set()
        for pos in v:
            if pos < 0:
                raise ValueError(f"Position must be non-negative, got {pos}")
            if pos in seen:
                raise ValueError(f"Duplicate position: {pos}")
            seen.add(pos)
            validated.append(pos)
        return validated


class MaskedPredictResponse(BaseModel):
    predictions: dict[str, dict[str, float]]


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
