"""
测试任务后台Worker

在独立线程中执行测试任务，避免阻塞API请求。
"""

from __future__ import annotations

import logging
import threading
import traceback
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sqlalchemy import update
from sqlalchemy.orm import Session

from src.config import get_local_now, settings
from src.models import (
    CycleData,
    ModelVersion,
    SessionLocal,
    TestJob,
    TestJobBattery,
    TestJobBatteryMetric,
    TestJobLog,
    TestJobMetricOverall,
    TestJobPrediction,
)


def _to_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(value.item())
    return float(value)


def _calculate_rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算RMSPE (Root Mean Square Percentage Error)"""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return float(np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2)))


def _resolve_layers(config: dict, default_layers: list[int]) -> list[int]:
    layers = config.get("layers")
    if isinstance(layers, (list, tuple)) and layers:
        return [int(layer) for layer in layers]

    num_layers = config.get("num_layers")
    num_neurons = config.get("num_neurons")

    if isinstance(num_layers, (list, tuple)):
        num_layers = num_layers[0] if num_layers else None
    if isinstance(num_neurons, (list, tuple)):
        num_neurons = num_neurons[0] if num_neurons else None

    try:
        num_layers_int = int(num_layers) if num_layers is not None else None
        num_neurons_int = int(num_neurons) if num_neurons is not None else None
    except (TypeError, ValueError):
        return default_layers

    if not num_layers_int or not num_neurons_int:
        return default_layers

    return [num_neurons_int] * num_layers_int


def _infer_inputs_dim(state_dict: dict[str, Any]) -> Optional[int]:
    for key in ("surrogateNN.NN.0.weight",):
        if key in state_dict:
            weight = state_dict[key]
            if hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
    for key in ("lstm.weight_ih_l0", "lstm.weight_ih_l0_reverse"):
        if key in state_dict:
            weight = state_dict[key]
            if hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
    return None


def _coerce_scaler(value: Any, dim: int, default: float) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach().float().view(-1)
    else:
        try:
            tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
        except (TypeError, ValueError):
            tensor = torch.tensor([default], dtype=torch.float32)

    if tensor.numel() == 0:
        return torch.full((dim,), default, dtype=torch.float32)
    if tensor.numel() == 1:
        return tensor.repeat(dim)
    if tensor.numel() != dim:
        return torch.full((dim,), default, dtype=torch.float32)
    return tensor


def _normalize_scalers(
    scaler_inputs: Any,
    scaler_targets: Any,
    inputs_dim: int,
    outputs_dim: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    if not isinstance(scaler_inputs, (tuple, list)) or len(scaler_inputs) != 2:
        scaler_inputs = (0.0, 1.0)
    if not isinstance(scaler_targets, (tuple, list)) or len(scaler_targets) != 2:
        scaler_targets = (0.0, 1.0)

    mean_inputs = _coerce_scaler(scaler_inputs[0], inputs_dim, 0.0)
    std_inputs = _coerce_scaler(scaler_inputs[1], inputs_dim, 1.0)
    mean_targets = _coerce_scaler(scaler_targets[0], outputs_dim, 0.0)
    std_targets = _coerce_scaler(scaler_targets[1], outputs_dim, 1.0)
    return (mean_inputs, std_inputs), (mean_targets, std_targets)


def _build_feature_row(cycle: Any, inputs_dim: int) -> list[float]:
    values = [
        float(cycle.feature_1),
        float(cycle.feature_2),
        float(cycle.feature_3),
        float(cycle.feature_4),
        float(cycle.feature_5),
        float(cycle.feature_6),
        float(cycle.feature_7),
        float(cycle.feature_8),
    ]
    if inputs_dim > 8:
        cycle_val = float(cycle.cycle_num) if cycle.cycle_num is not None else 0.0
        values.append(cycle_val)
    if inputs_dim > len(values):
        values.extend([0.0] * (inputs_dim - len(values)))
    elif inputs_dim < len(values):
        values = values[:inputs_dim]
    return values


class TestingWorker:
    """测试Worker"""

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        self.db: Optional[Session] = None
        self.job: Optional[TestJob] = None

        # WebSocket消息推送
        self._ws_manager: Optional[Any] = None
        self._init_websocket_manager()

        # 文件日志记录器
        self._file_logger: Optional[logging.Logger] = None
        self._log_file_path: Optional[Path] = None

    def _init_websocket_manager(self) -> None:
        """初始化WebSocket管理器引用"""
        try:
            from src.routes.testing import manager

            self._ws_manager = manager
        except ImportError:
            self._ws_manager = None

    def _init_file_logger(self, user_id: int, job_id: int) -> None:
        """初始化文件日志记录器"""
        # 创建日志目录: storage/models/{user_id}/logs/
        log_dir = settings.MODEL_STORAGE_PATH / str(user_id) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件路径: test_job_{job_id}.log
        self._log_file_path = log_dir / f"test_job_{job_id}.log"

        # 创建专用logger
        logger_name = f"test_job_{job_id}"
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
                "timestamp": get_local_now().isoformat(),
            }
            try:
                self._ws_manager.push_message(self.job_id, message)
            except Exception:
                pass  # 静默失败，不影响测试

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
        if level in ["WARNING", "ERROR"]:
            self._push_ws_message(
                "log",
                {
                    "test_job_id": self.job_id,
                    "level": level,
                    "message": message[:500],  # 限制WebSocket消息长度
                },
            )

    def _save_log_file_path(self) -> None:
        """保存日志文件路径到数据库（每个测试任务只记录一次）"""
        if self.db is None or self.job is None:
            return

        if self._log_file_path is None:
            return

        # 检查是否已存在
        existing = (
            self.db.query(TestJobLog)
            .filter(TestJobLog.test_job_id == self.job_id)
            .first()
        )

        if existing:
            return  # 已存在，跳过

        # 计算相对路径
        try:
            relative_path = self._log_file_path.relative_to(settings.BASE_DIR)
        except ValueError:
            relative_path = self._log_file_path

        log_record = TestJobLog(
            test_job_id=self.job_id,
            user_id=self.job.user_id,
            log_file_path=str(relative_path),
            created_at=get_local_now(),
        )

        try:
            self.db.add(log_record)
            self.db.commit()
        except Exception as e:
            self.db.rollback()
            if self._file_logger:
                self._file_logger.warning(f"日志文件路径保存失败: {str(e)}")

    def _update_job_status(self, status: str) -> None:
        """更新任务状态并推送WebSocket"""
        if self.db is None or self.job is None:
            return

        status_value = cast(str | None, self.job.status)
        old_status = status_value if status_value is not None else "UNKNOWN"
        values: dict[str, Any] = {"status": status}

        if status == "RUNNING" and self.job.started_at is None:
            values["started_at"] = get_local_now()
        elif status in {"SUCCEEDED", "FAILED"}:
            values["finished_at"] = get_local_now()

        self.db.execute(
            update(TestJob).where(TestJob.id == self.job_id).values(**values)
        )
        self.db.commit()

        # WebSocket推送状态变化
        if old_status != status:
            self._push_ws_message(
                "status_change",
                {
                    "test_job_id": self.job_id,
                    "old_status": old_status,
                    "new_status": status,
                },
            )

    def _load_model(self, model_version: ModelVersion) -> tuple[Any, dict]:
        """加载模型"""
        checkpoint_path = settings.BASE_DIR / str(model_version.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") or {}
        state_dict = checkpoint.get("model_state_dict") or {}
        config_inputs_dim = config.get("inputs_dim")
        inferred_inputs_dim = _infer_inputs_dim(state_dict) if state_dict else None
        if inferred_inputs_dim is not None:
            if (
                config_inputs_dim is None
                or int(config_inputs_dim) != inferred_inputs_dim
            ):
                if config_inputs_dim is not None:
                    self._log(
                        "WARNING",
                        "检测到模型输入维度与配置不一致，已按checkpoint修正。",
                    )
                config["inputs_dim"] = int(inferred_inputs_dim)

        # 根据算法类型加载模型
        algorithm = str(model_version.algorithm)
        if algorithm == "BASELINE":
            from src.utils.training_utils import DataDrivenNN

            layers = _resolve_layers(config, default_layers=[2, 100])
            inputs_dim = int(config.get("inputs_dim") or 8)
            scaler_inputs, scaler_targets = _normalize_scalers(
                config.get("scaler_inputs", (0.0, 1.0)),
                config.get("scaler_targets", (0.0, 1.0)),
                inputs_dim=inputs_dim,
                outputs_dim=1,
            )

            model = DataDrivenNN(
                seq_len=int(config.get("seq_len") or 1),
                inputs_dim=inputs_dim,
                outputs_dim=1,
                layers=layers,
                scaler_inputs=scaler_inputs,
                scaler_targets=scaler_targets,
                dropout_rate=config.get("dropout_rate", 0.2),
            )
        elif algorithm == "BILSTM":
            from src.tasks.model.bilstm_trainer import BiLSTMModel

            hidden_dim = config.get("hidden_dim") or config.get("hidden_size") or 128
            num_layers = config.get("num_layers") or 2
            dropout_rate = config.get("dropout_rate") or config.get("dropout") or 0.2

            model = BiLSTMModel(
                input_dim=int(config.get("inputs_dim") or 8),
                hidden_dim=int(hidden_dim),
                output_dim=1,
                num_layers=int(num_layers),
                dropout_rate=float(dropout_rate),
            )
        elif algorithm == "DEEPHPM":
            from src.utils.training_utils import DeepHPMNN

            seq_len = int(config.get("seq_len") or 1)
            dropout_rate = float(config.get("dropout_rate") or 0.2)
            layers = _resolve_layers(config, default_layers=[2, 100])
            inputs_dim = int(config.get("inputs_dim") or 8)
            scaler_inputs, scaler_targets = _normalize_scalers(
                config.get("scaler_inputs", (0.0, 1.0)),
                config.get("scaler_targets", (0.0, 1.0)),
                inputs_dim=inputs_dim,
                outputs_dim=1,
            )

            model = DeepHPMNN(
                seq_len=seq_len,
                inputs_dim=inputs_dim,
                outputs_dim=1,
                layers=layers,
                scaler_inputs=scaler_inputs,
                scaler_targets=scaler_targets,
                inputs_dynamical=config.get("inputs_dynamical", "U"),
                inputs_dim_dynamical=config.get("inputs_dim_dynamical", "outputs_dim"),
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        model.load_state_dict(state_dict)
        model.eval()

        return model, config

    def _prepare_test_data(
        self, battery_ids: list[int], config: dict, model_target: str
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
        """准备测试数据（支持多步预测）

        Args:
            battery_ids: 待测试的电池ID列表
            config: 模型配置
            model_target: 模型训练目标 (RUL/PCL)

        Returns:
            X: 输入特征 [N, seq_len, inputs_dim]
            y: 真实标签 [N]
            battery_indices: 每个样本对应的电池ID
            cycle_nums: 每个样本对应的预测目标周期号
        """
        if self.db is None:
            raise RuntimeError("数据库连接未初始化")
        if self.job is None:
            raise RuntimeError("任务对象未初始化")

        seq_len = int(config.get("seq_len") or 1)
        inputs_dim = int(config.get("inputs_dim") or 8)
        horizon_value = cast(int, self.job.horizon)
        horizon = int(horizon_value)  # 预测步长

        self._log(
            "INFO",
            f"数据准备参数: seq_len={seq_len}, horizon={horizon}, inputs_dim={inputs_dim}",
        )

        all_features = []
        all_targets = []
        battery_indices = []
        cycle_nums = []

        for battery_id in battery_ids:
            cycles = (
                self.db.query(CycleData)
                .filter(CycleData.battery_id == battery_id)
                .order_by(CycleData.cycle_num)
                .all()
            )

            # 需要至少 seq_len + horizon 个周期才能进行预测
            min_cycles = seq_len + horizon - 1
            if len(cycles) < min_cycles:
                self._log(
                    "WARNING",
                    f"电池 {battery_id} 数据不足 (需要>={min_cycles}个周期，实际{len(cycles)}个)，跳过",
                )
                continue

            # 滑动窗口：使用 [i, i+seq_len) 的数据预测第 i+seq_len+horizon-1 周期的值
            for i in range(len(cycles) - seq_len - horizon + 1):
                # 输入窗口：前 seq_len 个周期
                window = cycles[i : i + seq_len]
                features = np.array([_build_feature_row(c, inputs_dim) for c in window])

                # 预测目标：未来第 horizon 个周期
                target_cycle = cycles[i + seq_len + horizon - 1]

                # 根据模型训练时的target选择正确的标签
                if model_target == "RUL":
                    target = target_cycle.rul if target_cycle.rul is not None else 0
                else:  # PCL or BOTH
                    target = target_cycle.pcl if target_cycle.pcl is not None else 0.0

                all_features.append(features)
                all_targets.append(target)
                battery_indices.append(battery_id)
                cycle_num_value = cast(Any, target_cycle.cycle_num)
                cycle_nums.append(
                    int(cycle_num_value) if cycle_num_value is not None else 0
                )

        if not all_features:
            raise ValueError("没有有效的测试数据")

        X = torch.FloatTensor(np.array(all_features))
        y = torch.FloatTensor(all_targets)

        self._log("INFO", f"生成测试样本: {len(X)} 个 (来自 {len(battery_ids)} 个电池)")

        return X, y, battery_indices, cycle_nums

    def _run_inference(self, model: Any, X: torch.Tensor) -> np.ndarray:
        """执行推理"""
        with torch.no_grad():
            predictions = model(X)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            return predictions.cpu().numpy().flatten()

    def _save_predictions(
        self,
        battery_indices: list[int],
        cycle_nums: list[int],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target: str,
    ) -> None:
        """保存预测结果"""
        if self.db is None:
            return

        for battery_id, cycle_num, true_val, pred_val in zip(
            battery_indices, cycle_nums, y_true, y_pred
        ):
            prediction = TestJobPrediction(
                test_job_id=self.job_id,
                battery_id=battery_id,
                cycle_num=cycle_num,
                target=target,
                y_true=_to_float(true_val),
                y_pred=_to_float(pred_val),
            )
            self.db.add(prediction)

        self.db.commit()

    def _calculate_and_save_metrics(
        self,
        battery_indices: list[int],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target: str,
    ) -> None:
        """计算并保存指标"""
        if self.db is None:
            return

        # 整体指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        rmspe = _calculate_rmspe(y_true, y_pred)

        overall_metrics = {
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2),
            "rmspe": float(rmspe),
        }

        overall_metric = TestJobMetricOverall(
            test_job_id=self.job_id,
            target=target,
            metrics=overall_metrics,
            created_at=get_local_now(),
        )
        self.db.add(overall_metric)

        # 各电池指标
        unique_batteries = list(set(battery_indices))
        for battery_id in unique_batteries:
            mask = np.array([b == battery_id for b in battery_indices])
            if not mask.any():
                continue

            y_true_battery = y_true[mask]
            y_pred_battery = y_pred[mask]

            mse_battery = mean_squared_error(y_true_battery, y_pred_battery)
            rmse_battery = np.sqrt(mse_battery)
            r2_battery = r2_score(y_true_battery, y_pred_battery)
            rmspe_battery = _calculate_rmspe(y_true_battery, y_pred_battery)

            battery_metrics = {
                "mse": float(mse_battery),
                "rmse": float(rmse_battery),
                "r2": float(r2_battery),
                "rmspe": float(rmspe_battery),
            }

            battery_metric = TestJobBatteryMetric(
                test_job_id=self.job_id,
                battery_id=battery_id,
                target=target,
                metrics=battery_metrics,
            )
            self.db.add(battery_metric)

        self.db.commit()
        self._log("INFO", f"指标已保存 - RMSE: {rmse:.4f}, R²: {r2:.4f}")

    def run(self) -> None:
        """执行测试任务"""
        self.db = SessionLocal()
        db = self.db

        try:
            self.job = db.query(TestJob).filter(TestJob.id == self.job_id).first()

            if self.job is None:
                print(f"测试任务 {self.job_id} 不存在")
                return

            # 初始化文件日志记录器
            user_id = cast(int, self.job.user_id)
            self._init_file_logger(user_id, self.job_id)

            self._update_job_status("RUNNING")
            self._log("INFO", f"开始执行测试任务 #{self.job_id}")
            self._log("INFO", f"日志文件: {self._log_file_path}")

            # 保存日志文件路径到数据库
            self._save_log_file_path()

            # 推送进度: 0% - 任务启动
            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 0.0,
                    "stage": "initializing",
                    "message": "任务启动",
                },
            )

            # 加载模型
            model_version = (
                db.query(ModelVersion)
                .filter(ModelVersion.id == self.job.model_version_id)
                .first()
            )

            if model_version is None:
                raise ValueError("模型版本不存在")

            self._log("INFO", f"加载模型: {model_version.name} {model_version.version}")

            # 推送进度: 10% - 加载模型
            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 0.1,
                    "stage": "loading_model",
                    "message": f"加载模型: {model_version.name} {model_version.version}",
                },
            )

            model, config = self._load_model(model_version)

            # 使用模型版本中存储的训练目标（而非用户选择的target）
            model_target = str(model_version.target)
            if model_target == "BOTH":
                self._log(
                    "WARNING",
                    "模型训练目标为 BOTH，当前仅支持单目标预测，已按 PCL 处理。",
                )
                model_target = "PCL"
            self._log("INFO", f"模型训练目标: {model_target}")

            # 获取测试电池
            job_batteries = (
                db.query(TestJobBattery)
                .filter(TestJobBattery.test_job_id == self.job_id)
                .all()
            )

            battery_ids = [int(jb.battery_id) for jb in job_batteries]  # type: ignore[arg-type]
            self._log("INFO", f"测试电池: {battery_ids}")

            # 准备测试数据
            self._log("INFO", "准备测试数据...")

            # 推送进度: 30% - 准备数据
            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 0.3,
                    "stage": "preparing_data",
                    "message": f"准备 {len(battery_ids)} 个电池的测试数据",
                },
            )

            X, y, battery_indices, cycle_nums = self._prepare_test_data(
                battery_ids, config, model_target
            )
            self._log("INFO", f"测试样本数: {len(X)}")

            # 执行推理
            self._log("INFO", "执行推理...")

            # 推送进度: 50% - 执行推理
            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 0.5,
                    "stage": "inference",
                    "message": f"推理 {len(X)} 个样本",
                },
            )

            y_pred = self._run_inference(model, X)
            y_true = y.numpy()

            self._log("INFO", f"保存{model_target}预测结果...")

            # 推送进度: 70% - 保存结果
            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 0.7,
                    "stage": "saving_predictions",
                    "message": f"保存{model_target}预测结果",
                },
            )

            self._save_predictions(
                battery_indices, cycle_nums, y_true, y_pred, model_target
            )

            # 计算并保存指标
            self._log("INFO", "计算指标...")

            # 推送进度: 90% - 计算指标
            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 0.9,
                    "stage": "calculating_metrics",
                    "message": "计算评估指标",
                },
            )

            self._calculate_and_save_metrics(
                battery_indices, y_true, y_pred, model_target
            )

            self._push_ws_message(
                "progress",
                {
                    "test_job_id": self.job_id,
                    "progress": 1.0,
                    "stage": "completed",
                    "message": "测试任务完成",
                },
            )
            self._update_job_status("SUCCEEDED")
            self._log("INFO", "测试任务完成")

        except Exception as exc:
            if self.db is not None:
                self.db.rollback()
            self._log("ERROR", f"测试失败: {exc}")
            self._log("ERROR", traceback.format_exc())
            self._update_job_status("FAILED")

        finally:
            if self.db is not None:
                self.db.close()


def start_test_job(job_id: int) -> None:
    """启动测试任务"""

    def _run_worker():
        worker = TestingWorker(job_id)
        worker.run()

    thread = threading.Thread(target=_run_worker, daemon=True)
    thread.start()
