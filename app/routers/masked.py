"""Router for POST /masked-predict endpoint."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.requests import MaskedPredictRequest, MaskedPredictResponse

router = APIRouter()


@router.post("", response_model=MaskedPredictResponse)
async def masked_predict(body: MaskedPredictRequest, request: Request):
    """Predict nucleotide probabilities at specified positions."""
    engine = request.app.state.engine
    try:
        result = engine.masked_predict(
            sequence=body.sequence,
            positions=body.positions,
        )
        return MaskedPredictResponse(predictions=result["predictions"])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
