# src/routes/models.py
"""
模型版本管理 API 路由

提供模型版本的查询、下载和删除功能
"""

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.config import settings
from src.models import ModelVersion, TrainingJobRun, User, get_db
from src.routes.auth import get_current_user

router = APIRouter()


# --- Schemas ---
class ModelVersionResponse(BaseModel):
    """模型版本响应"""

    id: int
    user_id: int
    run_id: int
    algorithm: str
    name: str
    version: str
    config: dict
    metrics: dict
    checkpoint_path: str
    created_at: datetime

    class Config:
        from_attributes = True


class ModelVersionDetailResponse(BaseModel):
    """模型版本详情响应"""

    model: ModelVersionResponse
    training_info: Optional[dict] = None


# --- API Endpoints ---


@router.get("/versions", response_model=list[ModelVersionResponse])
async def list_model_versions(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    algorithm: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """
    列出用户的模型版本

    - 支持按算法筛选
    - 分页查询
    - 按创建时间倒序
    """
    query = db.query(ModelVersion).filter(
        ModelVersion.user_id == current_user.id, ModelVersion.deleted_at.is_(None)
    )

    if algorithm:
        query = query.filter(ModelVersion.algorithm == algorithm)

    models = (
        query.order_by(ModelVersion.created_at.desc()).offset(offset).limit(limit).all()
    )

    return [ModelVersionResponse.model_validate(m) for m in models]


@router.get("/versions/{version_id}", response_model=ModelVersionDetailResponse)
async def get_model_version(
    version_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    获取模型版本详情

    - 模型基本信息
    - 关联的训练任务信息
    """
    model = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.id == version_id,
            ModelVersion.user_id == current_user.id,
            ModelVersion.deleted_at.is_(None),
        )
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="模型版本不存在"
        )

    # 查询关联的训练信息
    training_run = (
        db.query(TrainingJobRun).filter(TrainingJobRun.id == model.run_id).first()
    )

    training_info = None
    if training_run:
        training_info = {
            "run_id": training_run.id,
            "job_id": training_run.job_id,
            "algorithm": training_run.algorithm,
            "status": training_run.status,
            "started_at": training_run.started_at,
            "finished_at": training_run.finished_at,
        }

    return ModelVersionDetailResponse(
        model=ModelVersionResponse.model_validate(model), training_info=training_info
    )


@router.get("/versions/{version_id}/download")
async def download_model_checkpoint(
    version_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    下载模型checkpoint文件

    返回 .pth 文件
    """
    model = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.id == version_id,
            ModelVersion.user_id == current_user.id,
            ModelVersion.deleted_at.is_(None),
        )
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="模型版本不存在"
        )

    # 构建完整路径
    checkpoint_path = settings.BASE_DIR / model.checkpoint_path

    if not checkpoint_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="模型文件不存在"
        )

    # 返回文件
    filename = f"{model.name}_{model.version}.pth"
    return FileResponse(
        path=str(checkpoint_path),
        filename=filename,
        media_type="application/octet-stream",
    )


@router.delete("/versions/{version_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_version(
    version_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    删除模型版本（软删除）

    不会物理删除文件，只是标记为已删除
    """
    model = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.id == version_id,
            ModelVersion.user_id == current_user.id,
            ModelVersion.deleted_at.is_(None),
        )
        .first()
    )

    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="模型版本不存在"
        )

    # 软删除
    db.execute(
        update(ModelVersion)
        .where(ModelVersion.id == version_id)
        .values(deleted_at=datetime.now(timezone.utc))
    )
    db.commit()

    return None


@router.get("/algorithms")
async def list_supported_algorithms(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """
    获取支持的算法列表

    返回所有可用的算法及其描述
    """
    return {
        "algorithms": [
            {
                "code": "BASELINE",
                "name": "Baseline Neural Network",
                "description": "基于数据驱动的前馈神经网络，用于电池SoH预测",
                "supported": True,
            },
            {
                "code": "BILSTM",
                "name": "Bidirectional LSTM",
                "description": "双向长短期记忆网络，捕捉电池退化的时间序列特征",
                "supported": True,
            },
            {
                "code": "DEEPHPM",
                "name": "Deep Hybrid Physics Model",
                "description": "深度混合物理模型，结合物理约束和数据驱动方法",
                "supported": True,
            },
        ]
    }
