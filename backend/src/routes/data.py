# src/routes/data.py
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from backend.src.models import (
    BatteryUnit,
    CycleData,
    Dataset,
    DataUpload,
    User,
    get_db,
)
from backend.src.tasks.parse_upload import parse_uploaded_file

from backend.src.routes.auth import get_current_user

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
    from backend.src.config import settings

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
        .filter(
            DataUpload.id == upload_id,
            DataUpload.user_id == current_user.id,
            DataUpload.deleted_at.is_(None),
        )
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
        .filter(DataUpload.user_id == current_user.id, DataUpload.deleted_at.is_(None))
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
    ).outerjoin(
        BatteryUnit,
        (Dataset.id == BatteryUnit.dataset_id) & (BatteryUnit.deleted_at.is_(None)),
    )

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

    batteries = (
        db.query(BatteryUnit)
        .filter(BatteryUnit.dataset_id == dataset_id, BatteryUnit.deleted_at.is_(None))
        .all()
    )
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
        .filter(
            BatteryUnit.id == battery_id,
            BatteryUnit.deleted_at.is_(None),
            Dataset.deleted_at.is_(None),
        )
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
        .filter(CycleData.battery_id == battery_id, CycleData.deleted_at.is_(None))
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
    软删除数据集及其所有关联数据（级联软删除）

    限制：
    - 仅允许删除用户自己上传的数据集（source_type='UPLOAD'）
    - 内置数据集（source_type='BUILTIN'）禁止删除

    级联删除范围：
    - dataset 表标记为删除
    - 关联的 training_job、model_version、test_job、training_job_run 标记为删除
    - 关联的 battery_unit、cycle_data 标记为删除（核心资产，支持恢复）
    - 关联的 data_upload 标记为删除（保留审计记录）
    - 衍生数据（metrics、logs、predictions）物理删除（可重新生成）
    """
    from backend.src.models import (
        TestExport,
        TestJob,
        TestJobBattery,
        TestJobBatteryMetric,
        TestJobLog,
        TestJobMetricOverall,
        TestJobPrediction,
        TrainingJob,
        TrainingJobBattery,
        TrainingJobRunLog,
        TrainingJobRunMetric,
    )

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

    now = datetime.now(timezone.utc)

    # 1. 获取该数据集下的所有电池 ID（仅未删除的）
    battery_ids = [b.id for b in dataset.batteries if b.deleted_at is None]

    if battery_ids:
        # 2. 软删除关联的训练任务
        training_jobs = (
            db.query(TrainingJob)
            .filter(
                TrainingJob.dataset_id == dataset_id,
                TrainingJob.deleted_at.is_(None),
            )
            .all()
        )

        for job in training_jobs:
            job.deleted_at = now  # type: ignore[assignment]

            # 2.1 软删除算法运行记录
            for run in job.runs:
                if run.deleted_at is None:
                    run.deleted_at = now  # type: ignore[assignment]

                # 2.2 软删除模型版本
                if run.model_version and run.model_version.deleted_at is None:
                    run.model_version.deleted_at = now  # type: ignore[assignment]

        # 3. 软删除关联的测试任务
        test_jobs = (
            db.query(TestJob)
            .filter(
                TestJob.dataset_id == dataset_id,
                TestJob.deleted_at.is_(None),
            )
            .all()
        )

        for test_job in test_jobs:
            test_job.deleted_at = now  # type: ignore[assignment]

        # 4. 物理删除关联数据（衍生数据，可重新生成）
        # 4.1 删除训练任务电池关联
        db.query(TrainingJobBattery).filter(
            TrainingJobBattery.battery_id.in_(battery_ids)
        ).delete(synchronize_session=False)

        # 4.2 删除测试任务电池关联
        db.query(TestJobBattery).filter(
            TestJobBattery.battery_id.in_(battery_ids)
        ).delete(synchronize_session=False)

        # 4.3 删除测试任务相关数据（可重新推理）
        test_job_ids = [tj.id for tj in test_jobs]
        if test_job_ids:
            db.query(TestJobMetricOverall).filter(
                TestJobMetricOverall.test_job_id.in_(test_job_ids)
            ).delete(synchronize_session=False)

            db.query(TestJobBatteryMetric).filter(
                TestJobBatteryMetric.test_job_id.in_(test_job_ids)
            ).delete(synchronize_session=False)

            db.query(TestJobPrediction).filter(
                TestJobPrediction.test_job_id.in_(test_job_ids)
            ).delete(synchronize_session=False)

            db.query(TestJobLog).filter(
                TestJobLog.test_job_id.in_(test_job_ids)
            ).delete(synchronize_session=False)

            db.query(TestExport).filter(
                TestExport.test_job_id.in_(test_job_ids)
            ).delete(synchronize_session=False)

        # 4.4 删除训练任务相关数据（可从checkpoint重新计算）
        training_job_ids = [tj.id for tj in training_jobs]
        if training_job_ids:
            run_ids = [run.id for job in training_jobs for run in job.runs]
            if run_ids:
                db.query(TrainingJobRunMetric).filter(
                    TrainingJobRunMetric.run_id.in_(run_ids)
                ).delete(synchronize_session=False)

                db.query(TrainingJobRunLog).filter(
                    TrainingJobRunLog.run_id.in_(run_ids)
                ).delete(synchronize_session=False)

        # 5. 软删除核心资产（支持恢复）
        # 5.1 软删除循环数据
        db.query(CycleData).filter(
            CycleData.battery_id.in_(battery_ids), CycleData.deleted_at.is_(None)
        ).update({"deleted_at": now}, synchronize_session=False)

        # 5.2 软删除电池单元
        db.query(BatteryUnit).filter(
            BatteryUnit.id.in_(battery_ids), BatteryUnit.deleted_at.is_(None)
        ).update({"deleted_at": now}, synchronize_session=False)

    # 6. 软删除数据集本身
    dataset.deleted_at = now  # type: ignore[assignment]

    # 7. 软删除关联的上传记录（保留审计记录）
    if dataset.upload_id is not None:
        upload = dataset.upload
        if upload is not None and upload.deleted_at is None:
            upload.deleted_at = now  # type: ignore[assignment]

    db.commit()

    return None
