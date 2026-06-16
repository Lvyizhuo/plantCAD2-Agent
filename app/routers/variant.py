"""Router for POST /variant-score endpoint."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.requests import VariantScoreRequest, VariantScoreResponse

router = APIRouter()


@router.post("", response_model=VariantScoreResponse)
async def score_variant(body: VariantScoreRequest, request: Request):
    """Score variant effect using zero-shot masked prediction."""
    engine = request.app.state.engine
    try:
        result = engine.score_variant(
            sequence=body.sequence,
            position=body.position,
            ref_allele=body.ref_allele,
            alt_alleles=body.alt_alleles,
        )
        return VariantScoreResponse(
            scores=result["scores"],
            ref_prob=result["ref_prob"],
            alt_probs=result["alt_probs"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
