# src/__init__.py
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.models import Base, engine
from src.routes import api_router

logger = logging.getLogger("uvicorn")


# 1. 定义 lifespan (生命周期) 管理器
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 仅在开发环境自动建表，避免生产环境启动时隐式改库
    if settings.ENV.lower() in {"dev", "develop", "development", "local"}:
        Base.metadata.create_all(bind=engine)
        logger.info("Auto create tables enabled (ENV=%s).", settings.ENV)
    else:
        logger.info("Skip auto create tables (ENV=%s).", settings.ENV)
    logger.info("Server starting on port %s (ENV=%s).", settings.PORT, settings.ENV)
    logger.info(
        "Database: %s:%s/%s", settings.DB_HOST, settings.DB_PORT, settings.DB_NAME
    )
    yield
    logger.info("Server shutting down...")


# 2. 实例化 FastAPI，并传入 lifespan 参数
app = FastAPI(
    title="储能电池寿命分析及算法测试平台 API",
    version="1.0.0",
    description="集成数据可视化、深度学习算法训练及电池寿命预测功能。",
    lifespan=lifespan,  # <--- 在这里注册
)

# 3. 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 注册所有路由
app.include_router(api_router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """基础健康检查"""
    return {"status": "ok", "env": settings.ENV, "database": "connected"}
