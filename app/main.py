"""PlantCAD2 Inference API — FastAPI entry point."""

import logging

from fastapi import FastAPI

from app.config import settings
from app.routers import embeddings, variant, predict, masked
from app.schemas.requests import HealthResponse
from modules.engine import PlantCAD2Engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PlantCAD2 Inference Service",
    description="REST API for PlantCAD2 DNA language model inference",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    logger.info("Starting PlantCAD2 engine...")
    app.state.engine = PlantCAD2Engine(
        base_model_path=settings.base_model_path,
        lora_models_path=settings.lora_models_path,
        device=settings.device,
        preload_lora=settings.preload_lora,
    )
    logger.info("PlantCAD2 engine ready")


@app.get("/health", response_model=HealthResponse)
async def health():
    engine = getattr(app.state, "engine", None)
    return HealthResponse(
        status="ok" if engine is not None else "not_ready",
        model_loaded=engine is not None,
        device=settings.device,
    )


app.include_router(embeddings.router, prefix="/embeddings", tags=["embeddings"])
app.include_router(variant.router, prefix="/variant-score", tags=["variant-score"])
app.include_router(predict.router, prefix="/predict", tags=["predict"])
app.include_router(masked.router, prefix="/masked-predict", tags=["masked-predict"])
