# src/routes/testing.py
"""
测试平台 API 路由

提供基于模型版本的推理评估功能
"""

import asyncio
import queue
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.models import (
    BatteryUnit,
    ModelVersion,
    TestJob,
    TestJobBattery,
    User,
    get_db,
)
from src.routes.auth import get_current_user
from src.tasks.testing_worker import start_test_job

router = APIRouter()


# --- WebSocket for Real-time Progress ---
class ConnectionManager:
    """WebSocket 连接管理器（支持跨线程消息推送）"""

    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}
        self.message_queue: queue.Queue = queue.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None

    async def connect(self, job_id: int, websocket: WebSocket):
        """连接 WebSocket"""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

        # 启动消息广播任务
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
        """线程安全的消息推送（供Worker线程调用）"""
        try:
            self.message_queue.put_nowait((job_id, message))
        except queue.Full:
            pass

    async def send_message(self, job_id: int, message: dict):
        """向指定任务的所有连接发送消息（异步方法）"""
        if job_id in self.active_connections:
            dead_connections = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    dead_connections.append(connection)

            for conn in dead_connections:
                try:
                    self.active_connections[job_id].remove(conn)
                except ValueError:
                    pass

    async def _message_broadcaster(self):
        """后台任务：从队列中读取消息并广播"""
        while True:
            try:
                try:
                    job_id, message = self.message_queue.get_nowait()
                    await self.send_message(job_id, message)
                except queue.Empty:
                    await asyncio.sleep(0.05)
            except Exception as e:
                print(f"消息广播错误: {e}")
                await asyncio.sleep(0.1)


manager = ConnectionManager()


# --- Schemas ---
class CreateTestJobRequest(BaseModel):
    """创建测试任务请求"""

    model_version_id: int
    dataset_id: int
    target: str = Field(..., pattern="^(RUL|PCL|BOTH)$")
    battery_ids: list[int] = Field(..., min_length=1)
    horizon: int = Field(default=1, ge=1)


class TestJobResponse(BaseModel):
    """测试任务响应"""

    id: int
    user_id: int
    model_version_id: int
    dataset_id: int
    target: str
    horizon: int
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class TestJobDetailResponse(BaseModel):
    """测试任务详情响应"""

    job: TestJobResponse
    model_info: dict
    batteries: list[dict]


# --- API Endpoints ---


@router.post(
    "/jobs", response_model=TestJobResponse, status_code=status.HTTP_201_CREATED
)
async def create_test_job(
    request: CreateTestJobRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    创建测试任务

    - 验证模型版本所有权
    - 验证数据集访问权限
    - 验证电池ID有效性
    - 创建测试任务并启动后台Worker
    """
    # 1. 验证模型版本所有权
    model_version = (
        db.query(ModelVersion)
        .filter(
            ModelVersion.id == request.model_version_id,
            ModelVersion.user_id == current_user.id,
        )
        .first()
    )

    if not model_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="模型版本不存在或无权访问"
        )

    # 2. 验证数据集访问权限
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

    # 3. 验证电池ID有效性
    batteries = (
        db.query(BatteryUnit)
        .filter(
            BatteryUnit.id.in_(request.battery_ids),
            BatteryUnit.dataset_id == request.dataset_id,
        )
        .all()
    )

    if len(batteries) != len(request.battery_ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="部分电池ID无效"
        )

    # 4. 创建测试任务
    test_job = TestJob(
        user_id=current_user.id,
        model_version_id=request.model_version_id,
        dataset_id=request.dataset_id,
        target=request.target,
        horizon=request.horizon,
        status="PENDING",
        created_at=datetime.now(timezone.utc),
    )
    db.add(test_job)
    db.flush()

    job_id: int = test_job.id  # type: ignore[assignment]

    # 5. 创建电池关联
    for battery_id in request.battery_ids:
        test_battery = TestJobBattery(test_job_id=job_id, battery_id=battery_id)
        db.add(test_battery)

    db.commit()
    db.refresh(test_job)

    # 6. 启动后台测试任务
    start_test_job(job_id=job_id)

    return test_job


@router.get("/jobs/{job_id}", response_model=TestJobDetailResponse)
async def get_test_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    获取测试任务详情

    - 任务基本信息
    - 模型版本信息
    - 关联的电池列表
    """
    job = (
        db.query(TestJob)
        .filter(TestJob.id == job_id, TestJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="测试任务不存在"
        )

    # 查询模型信息
    model_version = (
        db.query(ModelVersion).filter(ModelVersion.id == job.model_version_id).first()
    )

    if not model_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="模型版本不存在"
        )

    model_info = {
        "id": model_version.id,
        "name": model_version.name,
        "version": model_version.version,
        "algorithm": model_version.algorithm,
    }

    # 查询电池信息
    job_batteries = (
        db.query(TestJobBattery, BatteryUnit)
        .join(BatteryUnit, TestJobBattery.battery_id == BatteryUnit.id)
        .filter(TestJobBattery.test_job_id == job_id)
        .all()
    )

    batteries = [
        {
            "battery_id": battery.id,
            "battery_code": battery.battery_code,
            "total_cycles": battery.total_cycles,
        }
        for _, battery in job_batteries
    ]

    return TestJobDetailResponse(
        job=TestJobResponse.model_validate(job),
        model_info=model_info,
        batteries=batteries,
    )


@router.get("/jobs", response_model=list[TestJobResponse])
async def list_test_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    status_filter: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """
    列出用户的测试任务

    - 支持按状态筛选
    - 分页查询
    """
    query = db.query(TestJob).filter(TestJob.user_id == current_user.id)

    if status_filter:
        query = query.filter(TestJob.status == status_filter)

    jobs = query.order_by(TestJob.created_at.desc()).offset(offset).limit(limit).all()

    return [TestJobResponse.model_validate(job) for job in jobs]


@router.get("/jobs/{job_id}/metrics")
async def get_test_metrics(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
):
    """
    获取测试指标

    返回整体指标和各电池指标
    """
    from src.models import TestJobBatteryMetric, TestJobMetricOverall

    # 验证任务所有权
    job = (
        db.query(TestJob)
        .filter(TestJob.id == job_id, TestJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="测试任务不存在"
        )

    # 查询整体指标
    overall_metrics = (
        db.query(TestJobMetricOverall)
        .filter(TestJobMetricOverall.test_job_id == job_id)
        .all()
    )

    # 查询各电池指标
    battery_metrics = (
        db.query(TestJobBatteryMetric)
        .filter(TestJobBatteryMetric.test_job_id == job_id)
        .all()
    )

    return {
        "job_id": job_id,
        "overall_metrics": [
            {"target": m.target, "metrics": m.metrics} for m in overall_metrics
        ],
        "battery_metrics": [
            {"battery_id": m.battery_id, "target": m.target, "metrics": m.metrics}
            for m in battery_metrics
        ],
    }


@router.get("/jobs/{job_id}/predictions")
async def get_test_predictions(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    battery_id: Optional[int] = None,
):
    """
    获取测试预测结果

    支持按电池ID筛选
    """
    from src.models import TestJobPrediction

    # 验证任务所有权
    job = (
        db.query(TestJob)
        .filter(TestJob.id == job_id, TestJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="测试任务不存在"
        )

    # 查询预测结果
    query = db.query(TestJobPrediction).filter(TestJobPrediction.test_job_id == job_id)

    if battery_id:
        query = query.filter(TestJobPrediction.battery_id == battery_id)

    predictions = query.order_by(
        TestJobPrediction.battery_id, TestJobPrediction.cycle_num
    ).all()

    return {
        "job_id": job_id,
        "predictions": [
            {
                "battery_id": p.battery_id,
                "cycle_num": p.cycle_num,
                "target": p.target,
                "y_true": p.y_true,
                "y_pred": p.y_pred,
            }
            for p in predictions
        ],
    }


@router.get("/jobs/{job_id}/logs")
async def get_test_logs(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    level: Optional[str] = None,
    limit: int = 1000,
):
    """
    获取测试日志（从日志文件读取）

    支持按日志级别筛选，返回最新的 limit 条日志
    """
    from pathlib import Path

    from src.config import settings
    from src.models import TestJobLog

    # 验证任务所有权
    job = (
        db.query(TestJob)
        .filter(TestJob.id == job_id, TestJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="测试任务不存在"
        )

    # 查询日志文件路径
    log_record = db.query(TestJobLog).filter(TestJobLog.test_job_id == job_id).first()

    if not log_record:
        return {
            "job_id": job_id,
            "log_file_path": None,
            "logs": [],
            "message": "日志文件尚未创建",
        }

    # 读取日志文件
    log_file_path = settings.BASE_DIR / log_record.log_file_path

    if not log_file_path.exists():
        return {
            "job_id": job_id,
            "log_file_path": str(log_record.log_file_path),
            "logs": [],
            "message": "日志文件不存在",
        }

    try:
        # 读取日志文件的最后 limit 行
        with open(log_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 解析日志行（格式：2026-01-13 00:45:23 - [INFO] - message）
        logs = []
        for line in lines[-limit:]:
            line = line.strip()
            if not line:
                continue

            # 简单解析
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
                # 无法解析的行
                logs.append(
                    {
                        "timestamp": "",
                        "level": "INFO",
                        "message": line,
                    }
                )

        return {
            "job_id": job_id,
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


@router.post("/jobs/{job_id}/export")
async def export_test_results(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Session = Depends(get_db),
    format: str = Query(default="CSV", pattern="^(CSV|XLSX)$"),
):
    """
    导出测试结果

    支持CSV和XLSX格式
    """
    from src.models import TestExport

    # 验证任务所有权
    job = (
        db.query(TestJob)
        .filter(TestJob.id == job_id, TestJob.user_id == current_user.id)
        .first()
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="测试任务不存在"
        )

    if bool(job.status != "SUCCEEDED"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="只能导出已完成的测试任务"
        )

    # 创建导出记录
    export_record = TestExport(
        test_job_id=job_id,
        user_id=current_user.id,
        format=format,
        file_path="",  # 将由worker填充
        created_at=datetime.now(timezone.utc),
    )
    db.add(export_record)
    db.commit()
    db.refresh(export_record)

    # TODO: 启动后台导出任务
    # start_export_job(export_id=export_record.id)

    return {
        "export_id": export_record.id,
        "status": "PENDING",
        "message": "导出任务已创建",
    }


@router.websocket("/ws/jobs/{job_id}")
async def websocket_testing_progress(websocket: WebSocket, job_id: int):
    """
    WebSocket 实时测试进度

    客户端连接后会实时收到:
    - 日志消息
    - 进度更新
    - 状态变化
    """
    await manager.connect(job_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(job_id, websocket)
