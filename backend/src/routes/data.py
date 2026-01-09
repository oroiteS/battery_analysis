# src/routes/data.py
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from src.models import (
    BatteryUnit,
    CycleData,
    Dataset,
    DataUpload,
    User,
    get_db,
)
from src.routes.auth import get_current_user
from src.tasks.parse_upload import parse_uploaded_file

router = APIRouter()


# --- Schemas ---
class DataUploadResponse(BaseModel):
    id: int
    original_filename: str
    file_size: int
    status: str
    error_message: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetResponse(BaseModel):
    id: int
    name: str
    source_type: str
    battery_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class BatteryUnitResponse(BaseModel):
    id: int
    battery_code: str
    group_tag: str | None
    total_cycles: int
    nominal_capacity: float | None

    class Config:
        from_attributes = True


class CycleDataResponse(BaseModel):
    cycle_num: int
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    pcl: float | None
    rul: int | None

    class Config:
        from_attributes = True


# --- Endpoints ---
@router.post(
    "/upload", response_model=DataUploadResponse, status_code=status.HTTP_201_CREATED
)
async def upload_data(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    上传电池数据文件（支持 MAT/CSV/XLSX 格式，异步解析）

    支持的文件格式：
    - .mat: 必须包含 Features_mov_Flt, RUL_Flt, PCL_Flt, Cycles_Flt, Num_Cycles_Flt
    - .csv/.xlsx: 必须包含列 battery_code, cycle_num, feature_1~8, pcl, rul

    上传后状态为 PENDING，后台异步解析，可通过 GET /uploads/{upload_id} 查询状态
    """
    from src.config import settings

    # 验证文件格式
    allowed_extensions = {".mat", ".csv", ".xlsx", ".xls"}
    file_ext = Path(file.filename or "").suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: {file_ext}。支持的格式: {', '.join(allowed_extensions)}",
        )

    # 创建用户专属目录
    user_upload_dir = settings.UPLOAD_PATH / str(current_user.id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    # 生成唯一文件名
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    stored_filename = f"{timestamp}_{file.filename}"
    file_path = user_upload_dir / stored_filename

    # 保存文件
    content = await file.read()
    file_path.write_bytes(content)

    # 存储相对路径（相对于 UPLOAD_PATH）
    relative_path = f"{current_user.id}/{stored_filename}"

    # 创建上传记录
    upload_record = DataUpload(
        user_id=current_user.id,
        original_filename=file.filename or "unknown",
        stored_path=relative_path,  # 存储相对路径
        file_size=len(content),
        status="PENDING",
    )
    db.add(upload_record)
    db.commit()
    db.refresh(upload_record)

    # 启动异步解析任务
    thread = threading.Thread(
        target=parse_uploaded_file, args=(upload_record.id,), daemon=True
    )
    thread.start()

    return upload_record


@router.get("/uploads/{upload_id}", response_model=DataUploadResponse)
def get_upload_status(
    upload_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """查询上传记录的状态"""
    upload = (
        db.query(DataUpload)
        .filter(DataUpload.id == upload_id, DataUpload.user_id == current_user.id)
        .first()
    )

    if not upload:
        raise HTTPException(status_code=404, detail="Upload record not found")

    return upload


@router.get("/uploads", response_model=List[DataUploadResponse])
def list_uploads(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 20,
):
    """获取用户的上传记录列表"""
    uploads = (
        db.query(DataUpload)
        .filter(DataUpload.user_id == current_user.id)
        .order_by(DataUpload.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return uploads


@router.get("/datasets", response_model=List[DatasetResponse])
def list_datasets(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    source_type: str | None = None,
):
    """获取数据集列表（内置 + 用户上传）"""
    query = db.query(
        Dataset.id,
        Dataset.name,
        Dataset.source_type,
        Dataset.created_at,
        func.count(BatteryUnit.id).label("battery_count"),
    ).outerjoin(BatteryUnit, Dataset.id == BatteryUnit.dataset_id)

    # 软删除过滤
    query = query.filter(Dataset.deleted_at.is_(None))

    # 内置数据集 + 用户自己的数据集
    query = query.filter(
        or_(Dataset.source_type == "BUILTIN", Dataset.owner_user_id == current_user.id)
    )

    if source_type:
        query = query.filter(Dataset.source_type == source_type)

    query = query.group_by(Dataset.id).order_by(Dataset.created_at.desc())

    results = query.all()
    return [
        {
            "id": r.id,
            "name": r.name,
            "source_type": r.source_type,
            "battery_count": r.battery_count,
            "created_at": r.created_at,
        }
        for r in results
    ]


@router.get(
    "/datasets/{dataset_id}/batteries", response_model=List[BatteryUnitResponse]
)
def list_batteries(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取数据集中的电池列表"""
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.deleted_at.is_(None))
        .first()
    )
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 权限检查：内置数据集所有人可见，用户数据集仅所有者可见
    if (
        str(dataset.source_type) == "UPLOAD"
        and dataset.owner_user_id != current_user.id  # type: ignore[arg-type]
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    batteries = db.query(BatteryUnit).filter(BatteryUnit.dataset_id == dataset_id).all()
    return batteries


@router.get("/batteries/{battery_id}/cycles", response_model=List[CycleDataResponse])
def get_battery_cycles(
    battery_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    """获取电池的循环数据"""
    battery = (
        db.query(BatteryUnit)
        .join(Dataset)
        .filter(BatteryUnit.id == battery_id, Dataset.deleted_at.is_(None))
        .first()
    )
    if not battery:
        raise HTTPException(status_code=404, detail="Battery not found")

    # 权限检查
    if (
        str(battery.dataset.source_type) == "UPLOAD"
        and battery.dataset.owner_user_id != current_user.id  # type: ignore[arg-type]
    ):
        raise HTTPException(status_code=403, detail="Access denied")

    cycles = (
        db.query(CycleData)
        .filter(CycleData.battery_id == battery_id)
        .order_by(CycleData.cycle_num)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return cycles


@router.delete("/datasets/{dataset_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    软删除数据集（标记 deleted_at = 当前时间）

    限制：
    - 仅允许删除用户自己上传的数据集（source_type='UPLOAD'）
    - 内置数据集（source_type='BUILTIN'）禁止删除

    效果：
    - dataset 表标记为删除
    - 所有查询接口自动过滤已删除数据集
    - 关联数据（battery_unit、training_job 等）通过查询过滤自动隐藏
    """
    # 查询数据集（必须未删除）
    dataset = (
        db.query(Dataset)
        .filter(Dataset.id == dataset_id, Dataset.deleted_at.is_(None))
        .first()
    )

    if not dataset:
        raise HTTPException(
            status_code=404, detail="Dataset not found or already deleted"
        )

    # 权限检查：禁止删除内置数据集
    if str(dataset.source_type) == "BUILTIN":
        raise HTTPException(status_code=403, detail="Cannot delete built-in dataset")

    # 权限检查：仅允许所有者删除
    if dataset.owner_user_id != current_user.id:  # type: ignore[arg-type]
        raise HTTPException(status_code=403, detail="Access denied")

    # 软删除：标记 deleted_at
    dataset.deleted_at = datetime.now(timezone.utc)  # type: ignore[assignment]
    db.commit()

    return None
