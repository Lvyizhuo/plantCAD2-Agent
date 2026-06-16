"""Pydantic request/response models for PlantCAD2 API."""

from typing import Optional, Union
from pydantic import BaseModel, Field


# --- Embeddings ---

class EmbeddingRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence (ACGT, max 8192bp)")
    normalize: bool = Field(True, description="L2-normalize embeddings")


class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
    shape: list[int]
    sequence_length: int


# --- Variant Score ---

class VariantScoreRequest(BaseModel):
    sequence: str = Field(..., description="DNA context sequence containing the variant")
    position: int = Field(..., ge=0, description="0-based variant position in the sequence")
    ref_allele: str = Field(..., pattern="^[ACGTacgt]$", description="Reference allele")
    alt_alleles: list[str] = Field(
        ..., min_length=1,
        description="List of alternative alleles to score (A/C/G/T)"
    )


class VariantScoreResponse(BaseModel):
    scores: dict[str, float]
    ref_prob: float
    alt_probs: dict[str, float]


# --- LoRA Predict ---

class PredictRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence")
    task: str = Field(
        ...,
        description="Task name: acr_arabidopsis, acr_nine_species, acr_cell_type, "
                    "expression_on_off, expression_absolute, translation_on_off, "
                    "translation_absolute",
    )


class PredictResponse(BaseModel):
    task: str
    prediction: Union[str, float]
    probability: Optional[float] = None
    probabilities: Optional[list[float]] = None
    num_labels: Optional[int] = None


# --- Masked Predict ---

class MaskedPredictRequest(BaseModel):
    sequence: str = Field(..., description="DNA sequence")
    positions: list[int] = Field(
        ..., min_length=1,
        description="List of 0-based positions to mask and predict"
    )


class MaskedPredictResponse(BaseModel):
    predictions: dict[str, dict[str, float]]


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
