"""
训练任务后台Worker

在独立线程中执行训练任务，避免阻塞API请求。
通过回调机制实时记录日志和指标到数据库。
"""

from __future__ import annotations

import logging
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import torch
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.config import settings
from src.models import (
    ModelVersion,
    SessionLocal,
    TrainingJob,
    TrainingJobBattery,
    TrainingJobRun,
    TrainingJobRunLog,
    TrainingJobRunMetric,
)
from src.tasks.model.baseline_trainer import (
    BaselineTrainingConfig,
    TrainingCallbacks,
    train_baseline_from_database,
)
from src.tasks.model.bilstm_trainer import (
    BiLSTMTrainingConfig,
    train_bilstm_from_database,
)
from src.tasks.model.deephpm_trainer import (
    DeepHPMTrainingConfig,
    train_deephpm_from_database,
)


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            normalized[key] = float(value.detach().cpu().item())
        else:
            normalized[key] = value
    return normalized


class TrainingWorker:
    """训练Worker"""

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        self.db: Optional[Session] = None
        self.job: Optional[TrainingJob] = None
        self.current_run: Optional[TrainingJobRun] = None
        self.run_id: Optional[int] = None

        # WebSocket消息推送
        self._ws_manager: Optional[Any] = None
        self._init_websocket_manager()

        # 文件日志记录器
        self._file_logger: Optional[logging.Logger] = None
        self._log_file_path: Optional[Path] = None

    def _init_websocket_manager(self) -> None:
        """初始化WebSocket管理器引用"""
        try:
            from src.routes.training import manager

            self._ws_manager = manager
        except ImportError:
            self._ws_manager = None

    def _init_file_logger(self, user_id: int, job_id: int) -> None:
        """初始化文件日志记录器"""
        # 创建日志目录: storage/models/{user_id}/logs/
        log_dir = settings.MODEL_STORAGE_PATH / str(user_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件路径: training_job_{job_id}.log
        self._log_file_path = log_dir / f"training_job_{job_id}.log"

        # 创建专用logger
        logger_name = f"training_job_{job_id}"
        self._file_logger = logging.getLogger(logger_name)
        self._file_logger.setLevel(logging.DEBUG)

        # 清除已有的handlers（避免重复）
        self._file_logger.handlers.clear()

        # 文件handler
        file_handler = logging.FileHandler(
            self._log_file_path, mode="a", encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        self._file_logger.addHandler(file_handler)

        # 防止日志传播到root logger
        self._file_logger.propagate = False

    def _push_ws_message(self, message_type: str, data: dict[str, Any]) -> None:
        """推送WebSocket消息（线程安全）"""
        if self._ws_manager is not None:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            try:
                self._ws_manager.push_message(self.job_id, message)
            except Exception:
                pass  # 静默失败，不影响训练

    def _log(self, level: str, message: str) -> None:
        """记录日志到文件和WebSocket（不再写入数据库）"""
        # 1. 写入日志文件
        if self._file_logger:
            log_method = getattr(
                self._file_logger, level.lower(), self._file_logger.info
            )
            log_method(message)
        else:
            # 如果文件logger未初始化，降级到终端打印
            print(f"[{level}] {message}")

        # 2. WebSocket推送日志（仅推送关键日志）
        if self.current_run and level in ["WARNING", "ERROR"]:
            self._push_ws_message(
                "log",
                {
                    "run_id": self.run_id,
                    "algorithm": str(self.current_run.algorithm),
                    "level": level,
                    "message": message[:500],  # 限制WebSocket消息长度
                },
            )

    def _save_log_file_path(self) -> None:
        """保存日志文件路径到数据库（每个运行只记录一次）"""
        if self.db is None or self.run_id is None or self.job is None:
            return

        if self._log_file_path is None:
            return

        # 检查是否已存在
        existing = (
            self.db.query(TrainingJobRunLog)
            .filter(TrainingJobRunLog.run_id == self.run_id)
            .first()
        )

        if existing:
            return  # 已存在，跳过

        # 计算相对路径
        try:
            relative_path = self._log_file_path.relative_to(settings.BASE_DIR)
        except ValueError:
            relative_path = self._log_file_path

        log_record = TrainingJobRunLog(
            run_id=self.run_id,
            user_id=self.job.user_id,
            log_file_path=str(relative_path),
            created_at=datetime.now(timezone.utc),
        )

        try:
            self.db.add(log_record)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            if self._file_logger:
                self._file_logger.warning(f"日志文件路径保存失败: {str(e)}")

    def _save_epoch_metric(
        self, epoch: int, train_loss: float, val_loss: float, metrics: dict[str, Any]
    ) -> None:
        """保存epoch指标到数据库并推送WebSocket"""
        if self.db is None or self.run_id is None:
            return

        # 检查是否已存在该epoch的记录（避免重复插入）
        existing = (
            self.db.query(TrainingJobRunMetric)
            .filter(
                TrainingJobRunMetric.run_id == self.run_id,
                TrainingJobRunMetric.epoch == epoch,
            )
            .first()
        )

        if existing:
            # 如果已存在，跳过插入
            return

        normalized_metrics = _normalize_metrics(metrics)
        metric = TrainingJobRunMetric(
            run_id=self.run_id,
            epoch=epoch,
            train_loss=_to_float(train_loss),
            val_loss=_to_float(val_loss),
            metrics=normalized_metrics,
            created_at=datetime.now(timezone.utc),
        )

        try:
            self.db.add(metric)
            # 每10个epoch提交一次
            if epoch % 10 == 0:
                self.db.commit()
        except Exception as e:
            # 如果插入失败（例如唯一约束冲突），回滚并继续
            self.db.rollback()
            if self._file_logger:
                self._file_logger.warning(f"Epoch {epoch} 指标保存失败: {str(e)}")

        # WebSocket推送epoch进度
        if self.current_run:
            total_epochs_value = cast(int | None, self.current_run.total_epochs)
            epoch_data = {
                "run_id": self.run_id,
                "algorithm": str(self.current_run.algorithm),
                "epoch": epoch + 1,
                "total_epochs": int(total_epochs_value)
                if total_epochs_value is not None
                else 0,
                "train_loss": _to_float(train_loss),
                "val_loss": _to_float(val_loss),
            }
            # 添加额外指标
            epoch_data.update({k: _to_float(v) for k, v in normalized_metrics.items()})
            self._push_ws_message("epoch_progress", epoch_data)

    def _update_run_status(
        self, status: str, current_epoch: Optional[int] = None
    ) -> None:
        """更新运行状态并推送WebSocket"""
        if self.db is None or self.run_id is None or self.current_run is None:
            return

        status_value = cast(str | None, self.current_run.status)
        old_status = status_value if status_value is not None else "UNKNOWN"

        values: dict[str, Any] = {"status": status}

        if current_epoch is not None:
            values["current_epoch"] = current_epoch

        if status == "RUNNING" and self.current_run.started_at is None:
            values["started_at"] = datetime.now(timezone.utc)
        elif status in {"SUCCEEDED", "FAILED"}:
            values["finished_at"] = datetime.now(timezone.utc)

        self.db.execute(
            update(TrainingJobRun)
            .where(TrainingJobRun.id == self.run_id)
            .values(**values)
        )
        self.db.commit()

        # WebSocket推送状态变化
        if old_status != status:
            self._push_ws_message(
                "run_status_change",
                {
                    "run_id": self.run_id,
                    "algorithm": str(self.current_run.algorithm),
                    "old_status": old_status,
                    "new_status": status,
                    "current_epoch": current_epoch,
                },
            )

    def _update_job_status(self, status: str, progress: Optional[float] = None) -> None:
        """更新任务状态并推送WebSocket"""
        if self.db is None or self.job is None:
            return

        status_value = cast(str | None, self.job.status)
        old_status = status_value if status_value is not None else "UNKNOWN"

        values: dict[str, Any] = {"status": status}

        if progress is not None:
            values["progress"] = progress

        if status == "RUNNING" and self.job.started_at is None:
            values["started_at"] = datetime.now(timezone.utc)
        elif status in {"SUCCEEDED", "FAILED"}:
            values["finished_at"] = datetime.now(timezone.utc)

        self.db.execute(
            update(TrainingJob).where(TrainingJob.id == self.job_id).values(**values)
        )
        self.db.commit()

        # WebSocket推送任务进度
        if progress is not None or old_status != status:
            # 获取所有runs的状态统计
            all_runs = (
                self.db.query(TrainingJobRun)
                .filter(TrainingJobRun.job_id == self.job_id)
                .all()
            )

            completed_algorithms = [
                str(r.algorithm) for r in all_runs if str(r.status) == "SUCCEEDED"
            ]
            running_algorithms = [
                str(r.algorithm) for r in all_runs if str(r.status) == "RUNNING"
            ]
            pending_algorithms = [
                str(r.algorithm) for r in all_runs if str(r.status) == "PENDING"
            ]

            job_progress_value = cast(float | int | None, self.job.progress)
            progress_value = (
                float(progress)
                if progress is not None
                else float(job_progress_value)
                if job_progress_value is not None
                else 0.0
            )
            self._push_ws_message(
                "job_progress",
                {
                    "job_id": self.job_id,
                    "status": status,
                    "progress": progress_value,
                    "completed_algorithms": completed_algorithms,
                    "running_algorithms": running_algorithms,
                    "pending_algorithms": pending_algorithms,
                },
            )

    def _save_model_version(
        self,
        algorithm: str,
        model_state: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> int:
        """保存模型版本"""
        if self.db is None or self.job is None or self.run_id is None:
            return 0

        existing_versions = (
            self.db.query(ModelVersion)
            .filter(
                ModelVersion.user_id == self.job.user_id,
                ModelVersion.algorithm == algorithm,
            )
            .count()
        )

        version = f"v{existing_versions + 1}"

        checkpoint_dir = (
            settings.MODEL_STORAGE_PATH / str(self.job.user_id) / algorithm.lower()
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_filename = f"job_{self.job_id}_run_{self.run_id}_{version}.pth"
        checkpoint_path = checkpoint_dir / checkpoint_filename

        torch.save(
            {
                "model_state_dict": model_state,
                "config": config,
                "metrics": metrics,
                "job_id": self.job_id,
                "run_id": self.run_id,
            },
            checkpoint_path,
        )

        model_version = ModelVersion(
            user_id=self.job.user_id,
            run_id=self.run_id,
            algorithm=algorithm,
            name=f"{algorithm}_Job{self.job_id}",
            version=version,
            config=config,
            metrics=metrics,
            checkpoint_path=str(checkpoint_path.relative_to(settings.BASE_DIR)),
            created_at=datetime.now(timezone.utc),
        )

        self.db.add(model_version)
        self.db.commit()
        self.db.refresh(model_version)

        self._log("INFO", f"模型版本已保存: {model_version.name} {version}")
        return int(model_version.id)  # type: ignore[arg-type]

    def run(self) -> None:
        """执行训练任务"""
        self.db = SessionLocal()
        db = self.db
        if db is None:
            return

        try:
            self.job = (
                db.query(TrainingJob).filter(TrainingJob.id == self.job_id).first()
            )

            if self.job is None:
                print(f"训练任务 {self.job_id} 不存在")
                return

            # 初始化文件日志记录器
            user_id = cast(int, self.job.user_id)
            self._init_file_logger(user_id, self.job_id)

            self._update_job_status("RUNNING", progress=0.0)
            self._log("INFO", f"开始执行训练任务 #{self.job_id}")
            self._log("INFO", f"日志文件: {self._log_file_path}")

            runs = (
                db.query(TrainingJobRun)
                .filter(TrainingJobRun.job_id == self.job_id)
                .all()
            )

            if not runs:
                self._log("ERROR", "没有找到运行记录")
                self._update_job_status("FAILED")
                return

            job_batteries = (
                db.query(TrainingJobBattery)
                .filter(TrainingJobBattery.job_id == self.job_id)
                .all()
            )

            battery_ids_train = [
                int(jb.battery_id)  # type: ignore[arg-type]
                for jb in job_batteries
                if bool(jb.split_role == "train")  # type: ignore[arg-type]
            ]
            battery_ids_test = [
                int(jb.battery_id)  # type: ignore[arg-type]
                for jb in job_batteries
                if bool(jb.split_role == "test")  # type: ignore[arg-type]
            ]

            if not battery_ids_train:
                self._log("ERROR", "没有训练电池")
                self._update_job_status("FAILED")
                return

            if not battery_ids_test:
                self._log("WARNING", "没有测试电池，使用训练集的一部分作为测试集")
                battery_ids_test = [battery_ids_train.pop()]

            self._log(
                "INFO", f"训练电池: {battery_ids_train}, 测试电池: {battery_ids_test}"
            )

            total_runs = len(runs)
            for idx, run in enumerate(runs):
                self.current_run = run
                self.run_id = int(run.id)  # type: ignore[arg-type]

                self._log(
                    "INFO", f"开始训练算法: {run.algorithm} ({idx + 1}/{total_runs})"
                )
                self._update_run_status("RUNNING")

                # 保存日志文件路径到数据库（每个运行只记录一次）
                self._save_log_file_path()

                try:
                    if bool(run.algorithm == "BASELINE"):  # type: ignore[arg-type]
                        self._train_baseline(battery_ids_train, battery_ids_test)
                    elif bool(run.algorithm == "BILSTM"):  # type: ignore[arg-type]
                        self._train_bilstm(battery_ids_train, battery_ids_test)
                    elif bool(run.algorithm == "DEEPHPM"):  # type: ignore[arg-type]
                        self._train_deephpm(battery_ids_train, battery_ids_test)
                    else:
                        self._log("ERROR", f"不支持的算法: {run.algorithm}")
                        self._update_run_status("FAILED")
                        continue

                    self._update_run_status("SUCCEEDED")
                    self._log("INFO", f"算法 {run.algorithm} 训练成功")

                except Exception as exc:
                    if self.db is not None:
                        self.db.rollback()
                    self._log("ERROR", f"训练失败: {exc}")
                    self._log("ERROR", traceback.format_exc())
                    self._update_run_status("FAILED")

                progress = (idx + 1) / total_runs
                self._update_job_status("RUNNING", progress=progress)

            db.refresh(self.job)
            all_runs = (
                db.query(TrainingJobRun)
                .filter(TrainingJobRun.job_id == self.job_id)
                .all()
            )

            if all(bool(r.status == "SUCCEEDED") for r in all_runs):  # type: ignore[arg-type]
                self._update_job_status("SUCCEEDED", progress=1.0)
                self._log("INFO", "所有算法训练成功")
            else:
                self._update_job_status("FAILED", progress=1.0)
                self._log("WARNING", "部分算法训练失败")

        except Exception as exc:
            print(f"训练Worker错误: {exc}")
            if self.db is not None:
                self.db.rollback()
            traceback.print_exc()
            self._update_job_status("FAILED")

        finally:
            if self.db is not None:
                self.db.close()

    def _create_log_callback(self) -> Any:
        """创建通用的日志回调函数(包含WebSocket推送)"""

        def on_log(level: str, message: str) -> None:
            self._log(level, message)

            # 解析period级别的日志并推送详细损失信息
            # 格式: "Epoch: 10, Period: 152, Loss: 0.05782, Loss_U: 0.00166, Loss_F: 0.05616, Loss_F_t: 0.00000"
            if "Epoch:" in message and "Period:" in message and "Loss:" in message:
                try:
                    import re

                    match = re.search(
                        r"Epoch:\s*(\d+),\s*Period:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Loss_U:\s*([\d.]+),\s*Loss_F:\s*([\d.]+),\s*Loss_F_t:\s*([\d.]+)",
                        message,
                    )
                    if match and self.current_run:
                        epoch = int(match.group(1))
                        period = int(match.group(2))
                        loss = float(match.group(3))
                        loss_u = float(match.group(4))
                        loss_f = float(match.group(5))
                        loss_f_t = float(match.group(6))

                        # 推送period级别的训练详情
                        self._push_ws_message(
                            "training_detail",
                            {
                                "run_id": self.run_id,
                                "algorithm": str(self.current_run.algorithm),
                                "epoch": epoch,
                                "period": period,
                                "loss": loss,
                                "loss_U": loss_u,
                                "loss_F": loss_f,
                                "loss_F_t": loss_f_t,
                            },
                        )
                except Exception:
                    pass  # 解析失败时静默忽略

        return on_log

    def _train_baseline(
        self, battery_ids_train: list[int], battery_ids_test: list[int]
    ) -> None:
        """训练 Baseline 算法"""
        if self.job is None or self.db is None:
            return

        # 类型安全的获取 hyperparams
        job_hyperparams = self.job.hyperparams
        hyperparams: dict[str, Any] = (
            job_hyperparams if job_hyperparams is not None else {}
        )  # type: ignore[assignment]
        config = BaselineTrainingConfig(
            seq_len=hyperparams.get("seq_len", 1),
            perc_val=hyperparams.get("perc_val", 0.2),
            num_layers=hyperparams.get("num_layers", [2]),
            num_neurons=hyperparams.get("num_neurons", [128]),
            num_epoch=hyperparams.get("num_epoch", 2000),
            batch_size=hyperparams.get("batch_size", 1024),
            lr=hyperparams.get("lr", 0.001),
            step_size=hyperparams.get("step_size", 50000),
            gamma=hyperparams.get("gamma", 0.1),
            num_rounds=hyperparams.get("num_rounds", 1),
            random_seed=hyperparams.get("random_seed", 1234),
        )

        def on_log(level: str, message: str) -> None:
            self._log(level, message)

            # 解析period级别的日志并推送详细损失信息
            # 格式: "Epoch: 10, Period: 152, Loss: 0.05782, Loss_U: 0.00166, Loss_F: 0.05616, Loss_F_t: 0.00000"
            if "Epoch:" in message and "Period:" in message and "Loss:" in message:
                try:
                    import re

                    match = re.search(
                        r"Epoch:\s*(\d+),\s*Period:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Loss_U:\s*([\d.]+),\s*Loss_F:\s*([\d.]+),\s*Loss_F_t:\s*([\d.]+)",
                        message,
                    )
                    if match and self.current_run:
                        epoch = int(match.group(1))
                        period = int(match.group(2))
                        loss = float(match.group(3))
                        loss_u = float(match.group(4))
                        loss_f = float(match.group(5))
                        loss_f_t = float(match.group(6))

                        # 推送period级别的训练详情
                        self._push_ws_message(
                            "training_detail",
                            {
                                "run_id": self.run_id,
                                "algorithm": str(self.current_run.algorithm),
                                "epoch": epoch,
                                "period": period,
                                "loss": loss,
                                "loss_U": loss_u,
                                "loss_F": loss_f,
                                "loss_F_t": loss_f_t,
                            },
                        )
                except Exception:
                    pass  # 解析失败时静默忽略

        def on_epoch_end(
            epoch: int, train_loss: float, val_loss: float, metrics: dict[str, Any]
        ) -> None:
            self._save_epoch_metric(epoch, train_loss, val_loss, metrics)
            self._update_run_status("RUNNING", current_epoch=epoch + 1)

            if epoch % 50 == 0:
                self._log(
                    "INFO",
                    f"Epoch {epoch + 1}/{config.num_epoch}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
                )

        callbacks = TrainingCallbacks(on_log=on_log, on_epoch_end=on_epoch_end)

        results = train_baseline_from_database(
            db=self.db,
            battery_ids_train=battery_ids_train,
            battery_ids_test=battery_ids_test,
            config=config,
            callbacks=callbacks,
        )

        best_model_state = results.get("best_model_state")
        if best_model_state:
            self._save_model_version(
                algorithm="BASELINE",
                model_state=best_model_state,
                metrics=results.get("best_results", {}) or {},
                config=config.to_dict(),
            )

    def _train_bilstm(
        self, battery_ids_train: list[int], battery_ids_test: list[int]
    ) -> None:
        """训练 BiLSTM 算法"""
        if self.job is None or self.db is None:
            return

        job_hyperparams = self.job.hyperparams
        hyperparams: dict[str, Any] = (
            job_hyperparams if job_hyperparams is not None else {}
        )  # type: ignore[assignment]

        # BiLSTM 使用单个整数作为层数和隐藏维度
        num_layers_list = hyperparams.get("num_layers", [2, 3])
        num_neurons_list = hyperparams.get("num_neurons", [100, 150])

        config = BiLSTMTrainingConfig(
            seq_len=hyperparams.get("seq_len", 1),
            perc_val=hyperparams.get("perc_val", 0.2),
            num_layers=num_layers_list[0] if num_layers_list else 2,
            hidden_dim=num_neurons_list[0] if num_neurons_list else 100,
            num_epoch=hyperparams.get("num_epoch", 500),
            batch_size=hyperparams.get("batch_size", 32),
            lr=hyperparams.get("lr", 0.001),
            step_size=hyperparams.get("step_size", 100),
            gamma=hyperparams.get("gamma", 0.5),
            num_rounds=hyperparams.get("num_rounds", 1),
            random_seed=hyperparams.get("random_seed", 1234),
        )

        def on_log(level: str, message: str) -> None:
            self._log(level, message)

            # 解析period级别的日志并推送详细损失信息
            if "Epoch:" in message and "Period:" in message and "Loss:" in message:
                try:
                    import re

                    match = re.search(
                        r"Epoch:\s*(\d+),\s*Period:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Loss_U:\s*([\d.]+),\s*Loss_F:\s*([\d.]+),\s*Loss_F_t:\s*([\d.]+)",
                        message,
                    )
                    if match and self.current_run:
                        self._push_ws_message(
                            "training_detail",
                            {
                                "run_id": self.run_id,
                                "algorithm": str(self.current_run.algorithm),
                                "epoch": int(match.group(1)),
                                "period": int(match.group(2)),
                                "loss": float(match.group(3)),
                                "loss_U": float(match.group(4)),
                                "loss_F": float(match.group(5)),
                                "loss_F_t": float(match.group(6)),
                            },
                        )
                except Exception:
                    pass

        def on_epoch_end(
            epoch: int, train_loss: float, val_loss: float, metrics: dict[str, Any]
        ) -> None:
            self._save_epoch_metric(epoch, train_loss, val_loss, metrics)
            self._update_run_status("RUNNING", current_epoch=epoch + 1)

            if epoch % 50 == 0:
                self._log(
                    "INFO",
                    f"Epoch {epoch + 1}/{config.num_epoch}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
                )

        from src.tasks.model.bilstm_trainer import TrainingCallbacks as BiLSTMCallbacks

        callbacks = BiLSTMCallbacks(on_log=on_log, on_epoch_end=on_epoch_end)

        results = train_bilstm_from_database(
            db=self.db,
            battery_ids_train=battery_ids_train,
            battery_ids_test=battery_ids_test,
            config=config,
            callbacks=callbacks,
        )

        best_model_state = results.get("best_model_state")
        if best_model_state:
            self._save_model_version(
                algorithm="BILSTM",
                model_state=best_model_state,
                metrics=results.get("best_results", {}) or {},
                config=config.to_dict(),
            )

    def _train_deephpm(
        self, battery_ids_train: list[int], battery_ids_test: list[int]
    ) -> None:
        """训练 DeepHPM 算法"""
        if self.job is None or self.db is None:
            return

        job_hyperparams = self.job.hyperparams
        hyperparams: dict[str, Any] = (
            job_hyperparams if job_hyperparams is not None else {}
        )  # type: ignore[assignment]

        loss_weights_value = hyperparams.get("loss_weights", (1.0, 1.0, 1.0))
        loss_weights: tuple[float, float, float]
        if (
            isinstance(loss_weights_value, (list, tuple))
            and len(loss_weights_value) == 3
        ):
            loss_weights = (
                float(loss_weights_value[0]),
                float(loss_weights_value[1]),
                float(loss_weights_value[2]),
            )
        else:
            loss_weights = (1.0, 1.0, 1.0)

        config = DeepHPMTrainingConfig(
            seq_len=hyperparams.get("seq_len", 1),
            perc_val=hyperparams.get("perc_val", 0.2),
            num_layers=hyperparams.get("num_layers", [2]),
            num_neurons=hyperparams.get("num_neurons", [128]),
            num_epoch=hyperparams.get("num_epoch", 2000),
            batch_size=hyperparams.get("batch_size", 1024),
            lr=hyperparams.get("lr", 0.001),
            dropout_rate=hyperparams.get("dropout_rate", 0.2),
            weight_decay=hyperparams.get("weight_decay", 0.0),
            step_size=hyperparams.get("step_size", 50000),
            gamma=hyperparams.get("gamma", 0.1),
            lr_scheduler=hyperparams.get("lr_scheduler", "StepLR"),
            min_lr=hyperparams.get("min_lr", 1e-6),
            grad_clip=hyperparams.get("grad_clip", 0.0),
            early_stopping_patience=hyperparams.get("early_stopping_patience", 0),
            monitor_metric=hyperparams.get("monitor_metric", "val_loss"),
            num_rounds=hyperparams.get("num_rounds", 5),
            random_seed=hyperparams.get("random_seed", 1234),
            inputs_dynamical=hyperparams.get("inputs_dynamical", "s_norm, t_norm"),
            inputs_dim_dynamical=hyperparams.get("inputs_dim_dynamical", "inputs_dim"),
            loss_mode=hyperparams.get("loss_mode", "Sum"),
            loss_weights=loss_weights,
        )

        def on_log(level: str, message: str) -> None:
            self._log(level, message)

            # 解析period级别的日志并推送详细损失信息
            if "Epoch:" in message and "Period:" in message and "Loss:" in message:
                try:
                    import re

                    match = re.search(
                        r"Epoch:\s*(\d+),\s*Period:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Loss_U:\s*([\d.]+),\s*Loss_F:\s*([\d.]+),\s*Loss_F_t:\s*([\d.]+)",
                        message,
                    )
                    if match and self.current_run:
                        self._push_ws_message(
                            "training_detail",
                            {
                                "run_id": self.run_id,
                                "algorithm": str(self.current_run.algorithm),
                                "epoch": int(match.group(1)),
                                "period": int(match.group(2)),
                                "loss": float(match.group(3)),
                                "loss_U": float(match.group(4)),
                                "loss_F": float(match.group(5)),
                                "loss_F_t": float(match.group(6)),
                            },
                        )
                except Exception:
                    pass

        def on_epoch_end(
            epoch: int, train_loss: float, val_loss: float, metrics: dict[str, Any]
        ) -> None:
            self._save_epoch_metric(epoch, train_loss, val_loss, metrics)
            self._update_run_status("RUNNING", current_epoch=epoch + 1)

            if epoch % 50 == 0:
                self._log(
                    "INFO",
                    f"Epoch {epoch + 1}/{config.num_epoch}: "
                    f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}",
                )

        def on_hyperparameter_search(
            l_idx: int, n_idx: int, num_l: int, num_n: int, metrics: dict[str, float]
        ) -> None:
            self._log(
                "INFO",
                f"超参数搜索 [{l_idx + 1}, {n_idx + 1}]: "
                f"层数={num_l}, 神经元={num_n}, RMSPE={metrics.get('RMSPE', 0):.6f}",
            )

        from src.tasks.model.deephpm_trainer import (
            TrainingCallbacks as DeepHPMCallbacks,
        )

        callbacks = DeepHPMCallbacks(
            on_log=on_log,
            on_epoch_end=on_epoch_end,
            on_hyperparameter_search=on_hyperparameter_search,
        )

        results = train_deephpm_from_database(
            db=self.db,
            battery_ids_train=battery_ids_train,
            battery_ids_test=battery_ids_test,
            config=config,
            callbacks=callbacks,
        )

        best_model_state = results.get("best_model_state")
        if best_model_state:
            self._save_model_version(
                algorithm="DEEPHPM",
                model_state=best_model_state,
                metrics=results.get("best_results", {}) or {},
                config=config.to_dict(),
            )


def start_training_job(job_id: int) -> None:
    """启动训练任务（在后台线程中执行）"""

    def run_worker() -> None:
        worker = TrainingWorker(job_id=job_id)
        worker.run()

    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()

    print(f"训练任务 #{job_id} 已在后台启动")
