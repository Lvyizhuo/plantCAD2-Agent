"""Router for POST /predict endpoint."""

from fastapi import APIRouter, HTTPException, Request

from app.schemas.requests import PredictRequest, PredictResponse

router = APIRouter()


@router.post("", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    """Run a LoRA fine-tuned task prediction."""
    engine = request.app.state.engine
    try:
        result = engine.predict_function(
            sequence=body.sequence,
            task=body.task,
        )
        return PredictResponse(
            task=result["task"],
            prediction=result["prediction"],
            probability=result.get("probability"),
            probabilities=result.get("probabilities"),
            num_labels=result.get("num_labels"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
