"""Router for POST /embeddings endpoint."""

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from app.schemas.requests import EmbeddingRequest, EmbeddingResponse

router = APIRouter()


@router.post("", response_model=EmbeddingResponse)
async def get_embeddings(body: EmbeddingRequest, request: Request):
    """Extract per-position embeddings for a DNA sequence."""
    engine = request.app.state.engine
    try:
        result = engine.get_embeddings(
            sequence=body.sequence,
            normalize=body.normalize,
        )
        return EmbeddingResponse(
            embeddings=result["embeddings"].tolist(),
            shape=list(result["shape"]),
            sequence_length=result["sequence_length"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
