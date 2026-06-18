"""PlantCAD2 Inference API — FastAPI entry point."""

import sys
import traceback
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.routers import embeddings, variant, predict, masked
from app.schemas.requests import HealthResponse
from modules.engine import PlantCAD2Engine

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)
# Also log to file with rotation
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logger.add(
    str(log_dir / "plantcad2_{time:YYYYMMDD}.log"),
    rotation="00:00",  # New file at midnight
    retention="30 days",
    compression="gz",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
)

app = FastAPI(
    title="PlantCAD2 Inference Service",
    description="REST API for PlantCAD2 DNA language model inference",
    version="0.1.0",
)


# --- Exception handlers with logging ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log Pydantic validation errors (422) with detailed context."""
    errors = exc.errors()
    error_details = []
    for err in errors:
        loc = " -> ".join(str(l) for l in err.get("loc", []))
        msg = err.get("msg", "")
        err_type = err.get("type", "")
        input_val = err.get("input", "")
        error_details.append({
            "field": loc,
            "message": msg,
            "type": err_type,
            "input": str(input_val)[:100] if input_val else "",
        })

    # Format for log
    log_lines = []
    for detail in error_details:
        line = f"  - 字段: {detail['field']}, 错误: {detail['message']}"
        if detail["input"]:
            line += f", 输入值: {detail['input']}"
        log_lines.append(line)

    logger.warning(
        f"请求参数校验失败 [{request.method} {request.url.path}]\n"
        + "\n".join(log_lines)
    )

    # Format for response (user-friendly)
    response_errors = []
    for detail in error_details:
        response_errors.append(f"{detail['field']}: {detail['message']}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": "请求参数校验失败",
            "errors": response_errors,
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Log HTTP exceptions (400, 404, 500, etc.)."""
    if exc.status_code >= 500:
        logger.error(
            f"服务器错误 [{request.method} {request.url.path}] "
            f"状态码={exc.status_code}: {exc.detail}"
        )
    elif exc.status_code >= 400:
        logger.warning(
            f"客户端错误 [{request.method} {request.url.path}] "
            f"状态码={exc.status_code}: {exc.detail}"
        )

    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Log unexpected errors (500) with full traceback."""
    logger.error(
        f"未捕获异常 [{request.method} {request.url.path}]\n"
        f"异常类型: {type(exc).__name__}\n"
        f"异常信息: {exc}\n"
        f"堆栈跟踪:\n{traceback.format_exc()}"
    )

    return JSONResponse(
        status_code=500,
        content={"detail": "服务器内部错误"},
    )


# --- Startup ---

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


# --- Routes ---

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
