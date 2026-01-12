# src/routes/training.py
"""
训练平台 API 路由

提供训练任务的创建、查询、执行和监控功能
目前支持 Baseline 算法
"""

from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.models import (
    BatteryUnit,
    TrainingJob,
    TrainingJobBattery,
    TrainingJobRun,
    TrainingJobRunLog,
    TrainingJobRunMetric,
    User,
    get_db,
)
from src.routes.auth import get_current_user
from src.tasks.training_worker import start_training_job

router = APIRouter()


# --- Schemas ---
class BatterySelection(BaseModel):
    """电池选择"""

    battery_id: int
    split_role: str = Field(..., pattern="^(train|val|test)$")


class CreateTrainingJobRequest(BaseModel):
    """创建训练任务请求"""

    dataset_id: int
    target: str = Field(..., pattern="^(RUL|PCL|BOTH)$")
    algorithms: list[str] = Field(..., min_length=1)  # 例如: ["BASELINE"]
    batteries: list[BatterySelection]

    # 超参数
    seq_len: int = Field(default=1, ge=1)
    perc_val: float = Field(default=0.2, ge=0.0, le=1.0)
    num_layers: list[int] = Field(default=[2, 3])
    num_neurons: list[int] = Field(default=[100, 150])
    num_epoch: int = Field(default=500, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=0.001, gt=0)
    step_size: int = Field(default=100, ge=1)
    gamma: float = Field(default=0.5, gt=0, le=1)
    num_rounds: int = Field(default=1, ge=1)
    random_seed: int = Field(default=1234)


class TrainingJobResponse(BaseModel):
    """训练任务响应"""

    id: int
    user_id: int
    dataset_id: int
    target: str
    hyperparams: dict[str, Any]
    status: str
    progress: float
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class TrainingRunResponse(BaseModel):
    """训练运行响应"""

    id: int
    job_id: int
    algorithm: str
    status: str
    current_epoch: int
    total_epochs: int
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class TrainingJobDetailResponse(BaseModel):
    """训练任务详情响应"""

    job: TrainingJobResponse
    runs: list[TrainingRunResponse]
    batteries: list[dict[str, Any]]


# --- API Endpoints ---


@router.post(
    "/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_201_CREATED
)
async def create_training_job(
    request: CreateTrainingJobRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    创建训练任务

    - 验证数据集访问权限
    - 验证电池ID有效性
    - 创建训练任务和关联的运行记录
    - 启动后台训练Worker
    """
    # 1. 验证算法支持
    supported_algorithms = ["BASELINE", "BILSTM", "DEEPHPM"]
    for algo in request.algorithms:
        if algo not in supported_algorithms:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的算法: {algo}。当前支持: {supported_algorithms}",
            )

    # 2. 验证数据集访问权限（内置数据集或用户自己的数据集）
    from src.models import Dataset

    dataset = db.query(Dataset).filter(Dataset.id == request.dataset_id).first()

    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="数据集不存在"
        )

    if bool(dataset.source_type == "UPLOAD") and bool(
        dataset.owner_user_id != current_user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="无权访问此数据集"
        )

    # 3. 验证所有电池ID有效性
    battery_ids = [b.battery_id for b in request.batteries]
    batteries = (
        db.query(BatteryUnit)
        .filter(
            BatteryUnit.id.in_(battery_ids),
            BatteryUnit.dataset_id == request.dataset_id,
        )
        .all()
    )

    if len(batteries) != len(battery_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="部分电池ID无效"
        )

    # 4. 创建训练任务
    hyperparams = {
        "seq_len": request.seq_len,
        "perc_val": request.perc_val,
        "num_layers": request.num_layers,
        "num_neurons": request.num_neurons,
        "num_epoch": request.num_epoch,
        "batch_size": request.batch_size,
        "lr": request.lr,
        "step_size": request.step_size,
        "gamma": request.gamma,
        "num_rounds": request.num_rounds,
        "random_seed": request.random_seed,
    }

    training_job = TrainingJob(
        user_id=current_user.id,
        dataset_id=request.dataset_id,
        target=request.target,
        hyperparams=hyperparams,
        status="PENDING",
        progress=0.0,
        created_at=datetime.now(timezone.utc),
    )
    db.add(training_job)
    db.flush()

    # 获取生成的 job_id (类型断言为 int)
    job_id: int = training_job.id  # type: ignore[assignment]

    # 5. 创建电池关联
    for battery_selection in request.batteries:
        job_battery = TrainingJobBattery(
            job_id=job_id,
            battery_id=battery_selection.battery_id,
            split_role=battery_selection.split_role,
        )
        db.add(job_battery)

    # 6. 为每个算法创建运行记录
    for algorithm in request.algorithms:
        training_run = TrainingJobRun(
            job_id=job_id,
            algorithm=algorithm,
            status="PENDING",
            current_epoch=0,
            total_epochs=request.num_epoch,
        )
        db.add(training_run)

    db.commit()
    db.refresh(training_job)

    # 7. 启动后台训练任务
    start_training_job(job_id=job_id)

    return training_job


@router.get("/jobs/{job_id}", response_model=TrainingJobDetailResponse)
async def get_training_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    获取训练任务详情

    - 任务基本信息
    - 所有算法运行状态
    - 关联的电池列表
    """
    # 查询训练任务
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.id == job_id, TrainingJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="训练任务不存在"
        )

    # 查询运行记录
    runs = db.query(TrainingJobRun).filter(TrainingJobRun.job_id == job_id).all()

    # 查询电池信息
    job_batteries = (
        db.query(TrainingJobBattery, BatteryUnit)
        .join(BatteryUnit, TrainingJobBattery.battery_id == BatteryUnit.id)
        .filter(TrainingJobBattery.job_id == job_id)
        .all()
    )

    batteries = [
        {
            "battery_id": battery.id,
            "battery_code": battery.battery_code,
            "split_role": job_battery.split_role,
            "total_cycles": battery.total_cycles,
        }
        for job_battery, battery in job_batteries
    ]

    return TrainingJobDetailResponse(
        job=TrainingJobResponse.model_validate(job),
        runs=[TrainingRunResponse.model_validate(run) for run in runs],
        batteries=batteries,
    )


@router.get("/jobs", response_model=list[TrainingJobResponse])
async def list_training_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    status_filter: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """
    列出用户的训练任务

    - 支持按状态筛选
    - 分页查询
    """
    query = db.query(TrainingJob).filter(TrainingJob.user_id == current_user.id)

    if status_filter:
        query = query.filter(TrainingJob.status == status_filter)

    jobs = (
        query.order_by(TrainingJob.created_at.desc()).offset(offset).limit(limit).all()
    )

    return [TrainingJobResponse.model_validate(job) for job in jobs]


@router.get("/jobs/{job_id}/runs/{run_id}/metrics")
async def get_training_metrics(
    job_id: int,
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    获取训练指标

    返回指定运行的所有epoch指标
    """
    # 验证任务所有权
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.id == job_id, TrainingJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="训练任务不存在"
        )

    # 验证运行记录
    run = (
        db.query(TrainingJobRun)
        .filter(TrainingJobRun.id == run_id, TrainingJobRun.job_id == job_id)
        .first()
    )

    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="运行记录不存在"
        )

    # 查询指标
    metrics = (
        db.query(TrainingJobRunMetric)
        .filter(TrainingJobRunMetric.run_id == run_id)
        .order_by(TrainingJobRunMetric.epoch)
        .all()
    )

    return {
        "run_id": run_id,
        "algorithm": run.algorithm,
        "metrics": [
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_loss": m.val_loss,
                "metrics": m.metrics,
            }
            for m in metrics
        ],
    }


@router.get("/jobs/{job_id}/runs/{run_id}/logs")
async def get_training_logs(
    job_id: int,
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    level: Optional[str] = None,
    limit: int = 100,
):
    """
    获取训练日志

    支持按日志级别筛选
    """
    # 验证任务所有权
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.id == job_id, TrainingJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="训练任务不存在"
        )

    # 查询日志
    query = db.query(TrainingJobRunLog).filter(TrainingJobRunLog.run_id == run_id)

    if level:
        query = query.filter(TrainingJobRunLog.level == level)

    logs = query.order_by(TrainingJobRunLog.created_at.desc()).limit(limit).all()

    if level:
        query = query.filter(TrainingJobRunLog.level == level)

    logs = query.order_by(TrainingJobRunLog.created_at.desc()).limit(limit).all()

    return {
        "run_id": run_id,
        "logs": [
            {
                "id": log.id,
                "level": log.level,
                "message": log.message,
                "created_at": log.created_at,
            }
            for log in logs
        ],
    }


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    删除训练任务（软删除）

    只能删除 PENDING、FAILED 或 SUCCEEDED 状态的任务
    """
    job = (
        db.query(TrainingJob)
        .filter(TrainingJob.id == job_id, TrainingJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="训练任务不存在"
        )

    if bool(job.status == "RUNNING"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="无法删除正在运行的任务"
        )

    # 软删除
    db.execute(
        update(TrainingJob)
        .where(TrainingJob.id == job_id)
        .values(deleted_at=datetime.now(timezone.utc))
    )
    db.commit()

    return None


# --- WebSocket for Real-time Progress ---
class ConnectionManager:
    """WebSocket 连接管理器"""

    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}

    async def connect(self, job_id: int, websocket: WebSocket):
        """连接 WebSocket"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, job_id: int, websocket: WebSocket):
        """断开 WebSocket"""
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def send_message(self, job_id: int, message: dict):
        """向指定任务的所有连接发送消息"""
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # 忽略发送失败


manager = ConnectionManager()


@router.websocket("/ws/jobs/{job_id}")
async def websocket_training_progress(websocket: WebSocket, job_id: int):
    """
    WebSocket 实时训练进度

    客户端连接后会实时收到:
    - 日志消息
    - Epoch 进度
    - 指标更新
    - 任务状态变化
    """
    await manager.connect(job_id, websocket)
    try:
        while True:
            # 保持连接
            data = await websocket.receive_text()
            # 可以处理客户端发送的命令（例如：取消任务）
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
