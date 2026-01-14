# src/routes/training.py
"""
训练平台 API 路由

提供训练任务的创建、查询、执行和监控功能
目前支持 Baseline 算法
"""

# --- WebSocket for Real-time Progress ---
import asyncio
import queue
from datetime import datetime
from typing import Annotated, Any, Dict, Optional

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

from src.config import get_local_now
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
    num_layers: list[int] = Field(default=[2])
    num_neurons: list[int] = Field(default=[128])
    num_epoch: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=1024, ge=1)
    lr: float = Field(default=0.001, gt=0)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    step_size: int = Field(default=50000, ge=1)
    gamma: float = Field(default=0.1, gt=0, le=1)
    lr_scheduler: str = Field(
        default="StepLR", pattern="^(StepLR|CosineAnnealing|ReduceLROnPlateau)$"
    )
    min_lr: float = Field(default=1e-6, ge=0.0)
    grad_clip: float = Field(default=0.0, ge=0.0)
    early_stopping_patience: int = Field(default=0, ge=0)
    monitor_metric: str = Field(default="val_loss", pattern="^(val_loss|RMSPE)$")
    num_rounds: int = Field(default=5, ge=1)
    random_seed: int = Field(default=1234)
    inputs_dynamical: str = Field(default="s_norm, t_norm")
    inputs_dim_dynamical: str = Field(default="inputs_dim")
    loss_mode: str = Field(default="Sum", pattern="^(Sum|AdpBal|Baseline)$")
    loss_weights: list[float] = Field(
        default=[1.0, 1.0, 1.0], min_length=3, max_length=3
    )


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
        "dropout_rate": request.dropout_rate,
        "weight_decay": request.weight_decay,
        "step_size": request.step_size,
        "gamma": request.gamma,
        "lr_scheduler": request.lr_scheduler,
        "min_lr": request.min_lr,
        "grad_clip": request.grad_clip,
        "early_stopping_patience": request.early_stopping_patience,
        "monitor_metric": request.monitor_metric,
        "num_rounds": request.num_rounds,
        "random_seed": request.random_seed,
        "inputs_dynamical": request.inputs_dynamical,
        "inputs_dim_dynamical": request.inputs_dim_dynamical,
        "loss_mode": request.loss_mode,
        "loss_weights": request.loss_weights,
    }

    training_job = TrainingJob(
        user_id=current_user.id,
        dataset_id=request.dataset_id,
        target=request.target,
        hyperparams=hyperparams,
        status="PENDING",
        progress=0.0,
        created_at=get_local_now(),
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
        # BASELINE算法需要考虑num_rounds
        total_epochs = request.num_epoch
        if algorithm == "BASELINE":
            num_rounds = request.num_rounds if request.num_rounds else 5
            total_epochs = request.num_epoch * num_rounds

        training_run = TrainingJobRun(
            job_id=job_id,
            algorithm=algorithm,
            status="PENDING",
            current_epoch=0,
            total_epochs=total_epochs,
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
    # 查询训练任务（过滤已删除的）
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == current_user.id,
            TrainingJob.deleted_at.is_(None),  # 过滤已删除的任务
        )
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
    - 自动过滤已删除的任务
    """
    query = db.query(TrainingJob).filter(
        TrainingJob.user_id == current_user.id,
        TrainingJob.deleted_at.is_(None),  # 过滤已删除的任务
    )

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
    # 验证任务所有权（过滤已删除的）
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == current_user.id,
            TrainingJob.deleted_at.is_(None),  # 过滤已删除的任务
        )
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
    limit: int = 1000,
):
    """
    获取训练日志（从日志文件读取）

    支持按日志级别筛选，返回最新的 limit 条日志
    """

    from src.config import settings

    # 验证任务所有权（过滤已删除的）
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == current_user.id,
            TrainingJob.deleted_at.is_(None),  # 过滤已删除的任务
        )
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="训练任务不存在"
        )

    # 查询日志文件路径
    log_record = (
        db.query(TrainingJobRunLog).filter(TrainingJobRunLog.run_id == run_id).first()
    )

    if not log_record:
        return {
            "run_id": run_id,
            "log_file_path": None,
            "logs": [],
            "message": "日志文件尚未创建",
        }

    # 读取日志文件
    log_file_path = settings.BASE_DIR / log_record.log_file_path

    if not log_file_path.exists():
        return {
            "run_id": run_id,
            "log_file_path": str(log_record.log_file_path),
            "logs": [],
            "message": "日志文件不存在",
        }

    try:
        # 读取日志文件的最后 limit 行
        with open(str(log_file_path), "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 解析日志行（格式：2026-01-13 00:45:23 - [INFO] - message）
        logs = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue

            # 简单解析（可以根据实际格式调整）
            parts = line.split(" - ", 2)
            if len(parts) >= 3:
                timestamp_str = parts[0]
                level_str = parts[1].strip("[]")
                message = parts[2]

                # 按级别过滤
                if level and level_str != level:
                    continue

                logs.append(
                    {
                        "timestamp": timestamp_str,
                        "level": level_str,
                        "message": message,
                    }
                )
            else:
                # 无法解析的行，作为普通消息
                logs.append(
                    {
                        "timestamp": "",
                        "level": "INFO",
                        "message": line,
                    }
                )

        return {
            "run_id": run_id,
            "log_file_path": str(log_record.log_file_path),
            "total_lines": len(lines),
            "returned_lines": len(logs),
            "logs": logs,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"读取日志文件失败: {str(e)}",
        )


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
        .values(deleted_at=get_local_now())
    )
    db.commit()

    return None


@router.get("/jobs/{job_id}/runs/{run_id}/logs/download")
async def download_training_logs(
    job_id: int,
    run_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    下载训练日志文件

    返回日志文件供前端下载
    """
    from fastapi.responses import FileResponse

    from src.config import settings

    # 验证任务所有权（过滤已删除的）
    job = (
        db.query(TrainingJob)
        .filter(
            TrainingJob.id == job_id,
            TrainingJob.user_id == current_user.id,
            TrainingJob.deleted_at.is_(None),  # 过滤已删除的任务
        )
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

    # 查询日志文件路径
    log_record = (
        db.query(TrainingJobRunLog).filter(TrainingJobRunLog.run_id == run_id).first()
    )

    if not log_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="日志文件尚未创建"
        )

    # 构建完整路径
    log_file_path = settings.BASE_DIR / log_record.log_file_path

    if not log_file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="日志文件不存在"
        )

    # 生成友好的文件名
    # 格式: training_job_{job_id}_{algorithm}_{timestamp}.log
    algorithm = str(run.algorithm).lower()
    timestamp = get_local_now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_job_{job_id}_{algorithm}_{timestamp}.log"

    # 返回文件
    return FileResponse(
        path=str(log_file_path),
        filename=filename,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


class ConnectionManager:
    """WebSocket 连接管理器（支持跨线程消息推送）"""

    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}
        # 线程安全的消息队列（多个job共享）
        self.message_queue: queue.Queue = queue.Queue()
        # 后台任务引用
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, job_id: int, websocket: WebSocket):
        """连接 WebSocket"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

        # 启动消息广播任务（如果尚未启动）
        if self._broadcast_task is None or self._broadcast_task.done():
            self._broadcast_task = asyncio.create_task(self._message_broadcaster())

    def disconnect(self, job_id: int, websocket: WebSocket):
        """断开 WebSocket"""
        if job_id in self.active_connections:
            try:
                self.active_connections[job_id].remove(websocket)
            except ValueError:
                pass
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    def push_message(self, job_id: int, message: Dict[str, Any]):
        """
        线程安全的消息推送（供Worker线程调用）

        Args:
            job_id: 训练任务ID
            message: 消息字典
        """
        try:
            self.message_queue.put_nowait((job_id, message))
        except queue.Full:
            pass  # 队列满时丢弃消息

    async def send_message(self, job_id: int, message: dict):
        """向指定任务的所有连接发送消息（异步方法）"""
        if job_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.append(connection)

            # 清理断开的连接
            for conn in dead_connections:
                try:
                    self.active_connections[job_id].remove(conn)
                except ValueError:
                    pass

    async def _message_broadcaster(self):
        """后台任务：从队列中读取消息并广播"""
        while True:
            try:
                # 非阻塞获取消息
                try:
                    job_id, message = self.message_queue.get_nowait()
                    await self.send_message(job_id, message)
                except queue.Empty:
                    # 队列为空时休眠，避免CPU空转
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"消息广播错误: {e}")
                await asyncio.sleep(0.1)


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
