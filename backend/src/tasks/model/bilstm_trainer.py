"""
BiLSTM 算法训练执行器

支持从数据库加载数据，接受外部参数，并提供训练进度回调
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.model.data_loader import create_chosen_cells_from_db


class BiLSTMModel(nn.Module):
    """
    双向LSTM模型用于电池健康状态（SoH）预测

    Args:
        input_dim: 输入特征的维度
        hidden_dim: LSTM隐藏层的维度
        output_dim: 输出的维度（通常为1，预测SoH）
        num_layers: LSTM的层数
        dropout_rate: Dropout比例
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout_rate: float = 0.2,
    ):
        super(BiLSTMModel, self).__init__()
        # LSTM的dropout仅在num_layers > 1时生效
        lstm_dropout = dropout_rate if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.unsqueeze(1)


@dataclass
class BiLSTMTrainingConfig:
    """BiLSTM 训练配置（基于预训练模型参数）"""

    # 数据参数
    seq_len: int = 1
    perc_val: float = 0.2
    target: str = "PCL"  # RUL/PCL/BOTH

    # 模型参数（来自预训练模型）
    num_layers: int = 2  # BiLSTM 使用单个整数而非列表
    hidden_dim: int = 128  # 隐藏层维度，对应预训练的 num_neurons
    dropout_rate: float = 0.2

    # 训练参数（来自预训练模型）
    num_epoch: int = 2000
    batch_size: int = 1024
    lr: float = 0.001
    weight_decay: float = 0.0
    step_size: int = 50000
    gamma: float = 0.1
    lr_scheduler: str = "StepLR"  # StepLR/CosineAnnealing/ReduceLROnPlateau
    min_lr: float = 1e-6
    grad_clip: float = 0.0

    # 早停
    early_stopping_patience: int = 0  # 0表示禁用
    monitor_metric: str = "val_loss"  # val_loss/RMSPE

    # 实验参数（来自预训练模型）
    num_rounds: int = 5
    random_seed: int = 1234

    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "seq_len": self.seq_len,
            "perc_val": self.perc_val,
            "target": self.target,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "dropout_rate": self.dropout_rate,
            "num_epoch": self.num_epoch,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "step_size": self.step_size,
            "gamma": self.gamma,
            "lr_scheduler": self.lr_scheduler,
            "min_lr": self.min_lr,
            "grad_clip": self.grad_clip,
            "early_stopping_patience": self.early_stopping_patience,
            "monitor_metric": self.monitor_metric,
            "num_rounds": self.num_rounds,
            "random_seed": self.random_seed,
            "device": self.device,
        }


@dataclass
class TrainingCallbacks:
    """训练回调函数集合"""

    on_epoch_end: Optional[
        Callable[[int, float, float, dict[str, float], int, int], None]
    ] = None  # (epoch, train_loss, val_loss, metrics, round_idx, num_rounds)
    on_training_end: Optional[Callable[[dict[str, Any]], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None


def train_bilstm_from_database(
    db: Session,
    battery_ids_train: list[int],
    battery_ids_test: list[int],
    config: BiLSTMTrainingConfig,
    callbacks: Optional[TrainingCallbacks] = None,
) -> dict[str, Any]:
    """
    从数据库训练 BiLSTM 模型

    Args:
        db: 数据库会话
        battery_ids_train: 训练电池ID列表
        battery_ids_test: 测试电池ID列表
        config: 训练配置
        callbacks: 回调函数

    Returns:
        训练结果字典
    """
    callbacks = callbacks or TrainingCallbacks()
    device = torch.device(config.device)

    def _log(level: str, message: str) -> None:
        if callbacks.on_log:
            callbacks.on_log(level, message)

    _log("INFO", f"初始化 BiLSTM 训练器，设备: {device}")
    _log("INFO", f"训练配置: {config.to_dict()}")

    # 1. 加载数据
    _log("INFO", "从数据库加载数据...")
    inputs_dict, targets_dict = create_chosen_cells_from_db(
        db=db,
        battery_ids_train=battery_ids_train,
        battery_ids_test=battery_ids_test,
        seq_len=config.seq_len,
        perc_val=config.perc_val,
    )

    inputs_train = inputs_dict["train"].to(device)
    inputs_val = inputs_dict["val"].to(device)
    inputs_test = inputs_dict["test"].to(device)

    # 根据 target 选择正确的标签列
    # targets shape: (N, seq_len, 2), 其中 [:, :, 0] 是 PCL, [:, :, 1] 是 RUL
    if config.target == "RUL":
        target_idx = 1
    else:  # PCL or BOTH (BOTH时只训练PCL，测试时分别处理)
        target_idx = 0

    targets_train = targets_dict["train"][:, :, target_idx : target_idx + 1].to(device)
    targets_val = targets_dict["val"][:, :, target_idx : target_idx + 1].to(device)
    targets_test = targets_dict["test"][:, :, target_idx : target_idx + 1].to(device)

    _log(
        "INFO",
        f"数据加载完成: 训练集={inputs_train.shape[0]}, 验证集={inputs_val.shape[0]}, 测试集={inputs_test.shape[0]}",
    )

    # 2. 设置随机种子
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    # 3. 训练模型
    _log(
        "INFO",
        f"开始训练 BiLSTM，层数={config.num_layers}, 隐藏维度={config.hidden_dim}",
    )

    torch.backends.cudnn.enabled = False
    model = BiLSTMModel(
        input_dim=inputs_train.shape[2],
        hidden_dim=config.hidden_dim,
        output_dim=1,
        num_layers=config.num_layers,
        dropout_rate=config.dropout_rate,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # 学习率调度器
    if config.lr_scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )
    elif config.lr_scheduler == "CosineAnnealing":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.num_epoch, eta_min=config.min_lr
        )
    elif config.lr_scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.gamma,
            patience=10,
            min_lr=config.min_lr,
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.step_size, gamma=config.gamma
        )

    criterion = nn.MSELoss()

    train_set = TensorDataset(inputs_train, targets_train)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

    # 早停相关
    best_val_metric = float("inf")
    patience_counter = 0
    best_model_state = None

    # 训练循环
    for epoch in range(config.num_epoch):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()

            # 梯度裁剪
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            optimizer.step()
            epoch_loss += loss.item()

        # 验证
        model.eval()
        with torch.no_grad():
            val_output = model(inputs_val)
            val_loss = criterion(val_output, targets_val).item()

            # 计算监控指标
            current_val_metric: float
            if config.monitor_metric == "RMSPE":
                if config.target == "RUL":
                    val_pred = val_output
                    val_true = targets_val
                else:
                    val_pred = 1.0 - val_output
                    val_true = 1.0 - targets_val

                mask = val_true != 0
                if torch.any(mask):
                    val_rmspe = torch.sqrt(
                        torch.mean(
                            ((val_pred[mask] - val_true[mask]) / val_true[mask]) ** 2
                        )
                    ).item()
                else:
                    val_rmspe = float("inf")
                current_val_metric = val_rmspe
            else:
                current_val_metric = val_loss

        avg_train_loss = epoch_loss / len(train_loader)

        _log(
            "INFO",
            f"Epoch {epoch + 1}/{config.num_epoch}: train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}",
        )

        if callbacks.on_epoch_end:
            callbacks.on_epoch_end(epoch + 1, avg_train_loss, val_loss, {}, 0, 1)

        # 学习率调度
        if config.lr_scheduler == "ReduceLROnPlateau":
            scheduler.step(current_val_metric)  # type: ignore
        else:
            scheduler.step()  # type: ignore

        # 早停逻辑
        if config.early_stopping_patience > 0:
            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                _log("INFO", f"验证指标改善: {best_val_metric:.6f}")
            else:
                patience_counter += 1
                _log(
                    "INFO",
                    f"验证指标未改善 ({patience_counter}/{config.early_stopping_patience})",
                )

                if patience_counter >= config.early_stopping_patience:
                    _log("INFO", f"触发早停，在 epoch {epoch + 1} 停止训练")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        _log("INFO", "恢复最佳模型权重")
                    break

    # 4. 评估模型
    _log("INFO", "评估模型...")
    model.eval()

    def eval_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        with torch.no_grad():
            pred = model(x)
            if config.target == "RUL":
                pred_eval = pred
                y_eval = y
            else:
                pred_eval = 1.0 - pred
                y_eval = 1.0 - y

            mae = torch.mean(torch.abs(pred_eval - y_eval)).item()
            mse = torch.mean((pred_eval - y_eval) ** 2).item()
            sse = torch.sum((pred_eval - y_eval) ** 2).item()

            mask = y_eval != 0
            if torch.any(mask):
                rmspe = torch.sqrt(
                    torch.mean(((pred_eval[mask] - y_eval[mask]) / y_eval[mask]) ** 2)
                ).item()
            else:
                rmspe = float("inf")

            ss_res = torch.sum((y_eval - pred_eval) ** 2)
            ss_tot = torch.sum((y_eval - torch.mean(y_eval)) ** 2)
            r2 = (1 - ss_res / ss_tot).item()

            return {"MAE": mae, "MSE": mse, "SSE": sse, "RMSPE": rmspe, "R2": r2}

    metrics_train = eval_metrics(inputs_train, targets_train)
    metrics_val = eval_metrics(inputs_val, targets_val)
    metrics_test = eval_metrics(inputs_test, targets_test)

    _log(
        "INFO",
        f"测试集 RMSPE: {metrics_test['RMSPE']:.6f}, R²: {metrics_test['R2']:.6f}",
    )

    # 5. 返回结果
    results = {
        "config": config.to_dict(),
        "best_model_state": model.state_dict(),
        "best_results": {
            "train": metrics_train,
            "val": metrics_val,
            "test": metrics_test,
        },
        "metric_mean": {
            "train": np.array([[metrics_train["RMSPE"]]]),
            "val": np.array([[metrics_val["RMSPE"]]]),
            "test": np.array([[metrics_test["RMSPE"]]]),
        },
    }

    if callbacks.on_training_end:
        callbacks.on_training_end(results)

    _log("INFO", "训练完成")
    return results
