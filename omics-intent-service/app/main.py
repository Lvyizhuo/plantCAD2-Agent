"""
组学智能体意图识别服务入口

基于FastAPI的意图识别服务，通过大模型识别用户意图并调用对应的下游任务接口。
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from loguru import logger

from app.config import settings
from app.routers.intent import router as intent_router
from app.schemas.requests import HealthResponse


# 配置日志
def setup_logging():
    """配置loguru日志"""
    # 移除默认handler
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # 文件输出
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "intent_service_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        encoding="utf-8",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    setup_logging()
    logger.info("=" * 60)
    logger.info("组学智能体意图识别服务启动中...")
    logger.info(f"LLM模型: {settings.llm_model}")
    logger.info(f"LLM服务地址: {settings.llm_base_url}")
    logger.info(f"PlantCAD2服务地址: {settings.plantcad2_base_url}")
    logger.info(f"EVO2服务地址: {settings.evo2_base_url}")
    logger.info("=" * 60)

    yield

    # 关闭时
    logger.info("组学智能体意图识别服务关闭")


# 创建FastAPI应用
app = FastAPI(
    title="组学智能体意图识别服务",
    description="通过大模型识别用户意图，返回任务推荐或直接调用下游接口获取结果。",
    version="1.0.0",
    lifespan=lifespan,
)


# 注册路由
app.include_router(intent_router)


# 健康检查
@app.get("/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    return HealthResponse(status="ok", service="omics-intent-service")


# 异常处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求参数验证错误处理"""
    logger.warning(f"请求参数验证失败 | path={request.url.path} errors={exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "confidence": "low",
            "guide_message": "请求参数格式错误，请检查后重试。",
            "error": {
                "code": 1005,
                "message": "请求参数格式错误",
                "detail": str(exc.errors()),
            },
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理"""
    logger.error(f"未处理的异常 | path={request.url.path} error={str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "confidence": "low",
            "guide_message": "服务器内部错误，请稍后重试。",
            "error": {
                "code": 1004,
                "message": "服务异常",
                "detail": str(exc),
            },
        },
    )
