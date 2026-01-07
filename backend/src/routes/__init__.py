from fastapi import APIRouter
from src.routes import auth

api_router = APIRouter()

# 聚合各个业务模块
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
# api_router.include_router(batteries.router, prefix="/batteries", tags=["Batteries"])
# api_router.include_router(training.router, prefix="/training-jobs", tags=["Training Jobs"])
# api_router.include_router(models.router, prefix="/models", tags=["Models"])
# api_router.include_router(predictions.router, prefix="/predictions", tags=["Predictions"])
