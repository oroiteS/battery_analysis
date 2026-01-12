# /src/routes/__init__.py
from fastapi import APIRouter

from src.routes import auth, data, models, training

api_router = APIRouter()
# 聚合各个业务模块
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(data.router, prefix="/data", tags=["Data Management"])
api_router.include_router(
    training.router, prefix="/training", tags=["Training Platform"]
)
api_router.include_router(models.router, prefix="/models", tags=["Model Versions"])
