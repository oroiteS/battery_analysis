# /src/routes/__init__.py
from fastapi import APIRouter
from src.routes import auth

api_router = APIRouter()
# 聚合各个业务模块
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])

