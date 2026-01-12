"""
DeepHPM 算法训练执行器

支持从数据库加载数据，接受外部参数，并提供训练进度回调
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch
from sqlalchemy.orm import Session
from torch import optim
from torch.utils.data import DataLoader

from src.tasks.model.data_loader import create_chosen_cells_from_db
from src.utils.training_utils import (
    DeepHPMNN,
    My_loss,
    TensorDataset,
    standardize_tensor,
    train,
)


@dataclass
class DeepHPMTrainingConfig:
    """DeepHPM 训练配置（基于预训练模型参数）"""

    # 数据参数
    seq_len: int = 1
    perc_val: float = 0.2

    # 模型参数（来自预训练模型）
    num_layers: list[int] = None  # type: ignore
    num_neurons: list[int] = None  # type: ignore
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

    # DeepHPM 特定参数
    inputs_dynamical: str = "U"
    inputs_dim_dynamical: str = "outputs_dim"
    loss_mode: str = "Sum"  # "Sum" 或 "AdpBal"
    loss_weights: tuple[float, float, float] = (
        1.0,
        1.0,
        1.0,
    )  # (loss_U, loss_F, loss_F_t)

    def __post_init__(self):
        if self.num_layers is None:
            self.num_layers = [2]
        if self.num_neurons is None:
            self.num_neurons = [128]

    def to_dict(self) -> dict[str, Any]:
        """转换为字典"""
        return {
            "seq_len": self.seq_len,
            "perc_val": self.perc_val,
            "num_layers": self.num_layers,
            "num_neurons": self.num_neurons,
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
            "inputs_dynamical": self.inputs_dynamical,
            "inputs_dim_dynamical": self.inputs_dim_dynamical,
            "loss_mode": self.loss_mode,
            "loss_weights": self.loss_weights,
        }


@dataclass
class TrainingCallbacks:
    """训练回调函数集合"""

    on_epoch_end: Optional[Callable[[int, float, float, dict[str, float]], None]] = None
    on_training_end: Optional[Callable[[dict[str, Any]], None]] = None
    on_hyperparameter_search: Optional[
        Callable[[int, int, int, int, dict[str, float]], None]
    ] = None
    on_log: Optional[Callable[[str, str], None]] = None


def train_deephpm_from_database(
    db: Session,
    battery_ids_train: list[int],
    battery_ids_test: list[int],
    config: DeepHPMTrainingConfig,
    callbacks: Optional[TrainingCallbacks] = None,
) -> dict[str, Any]:
    """
    从数据库训练 DeepHPM 模型

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

    _log("INFO", f"初始化 DeepHPM 训练器，设备: {device}")
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

    targets_train = targets_dict["train"][:, :, 0:1].to(device)
    targets_val = targets_dict["val"][:, :, 0:1].to(device)
    targets_test = targets_dict["test"][:, :, 0:1].to(device)

    _log(
        "INFO",
        f"数据加载完成: 训练集={inputs_train.shape[0]}, 验证集={inputs_val.shape[0]}, 测试集={inputs_test.shape[0]}",
    )

    # 2. 超参数搜索
    metric_mean = {
        "train": np.zeros((len(config.num_layers), len(config.num_neurons))),
        "val": np.zeros((len(config.num_layers), len(config.num_neurons))),
        "test": np.zeros((len(config.num_layers), len(config.num_neurons))),
    }

    best_model = None
    best_results = {}
    best_rmspe = float("inf")

    for l_idx, num_l in enumerate(config.num_layers):
        for n_idx, num_n in enumerate(config.num_neurons):
            layers = num_l * [num_n]

            _log(
                "INFO",
                f"超参数搜索 [{l_idx + 1}/{len(config.num_layers)}, {n_idx + 1}/{len(config.num_neurons)}]: "
                f"层数={num_l}, 神经元={num_n}",
            )

            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)

            # 标准化
            inputs_train_norm, mean_inputs_train, std_inputs_train = standardize_tensor(
                inputs_train, mode="fit"
            )
            _, mean_targets_train, std_targets_train = standardize_tensor(
                targets_train, mode="fit"
            )

            train_set = TensorDataset(inputs_train, targets_train)
            train_loader = DataLoader(
                train_set,
                batch_size=config.batch_size,
                shuffle=True,
                drop_last=True,
            )

            torch.backends.cudnn.enabled = False
            model = DeepHPMNN(
                seq_len=config.seq_len,
                inputs_dim=inputs_train.shape[2],
                outputs_dim=1,
                layers=layers,
                scaler_inputs=(mean_inputs_train, std_inputs_train),
                scaler_targets=(mean_targets_train, std_targets_train),
                inputs_dynamical=config.inputs_dynamical,
                inputs_dim_dynamical=config.inputs_dim_dynamical,
                dropout_rate=config.dropout_rate,
            ).to(device)

            log_sigma_u = torch.zeros(())
            log_sigma_f = torch.zeros(())
            log_sigma_f_t = torch.zeros(())

            criterion = My_loss(mode=config.loss_mode, weights=config.loss_weights)
            params = [p for p in model.parameters()]
            optimizer = optim.Adam(
                params, lr=config.lr, weight_decay=config.weight_decay
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

            # 训练
            model, results_epoch = train(
                num_epoch=config.num_epoch,
                batch_size=config.batch_size,
                train_loader=train_loader,
                num_slices_train=inputs_train.shape[0],
                inputs_val=inputs_val,
                targets_val=targets_val,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                log_sigma_u=log_sigma_u,
                log_sigma_f=log_sigma_f,
                log_sigma_f_t=log_sigma_f_t,
            )

            # 手动调用 epoch 回调
            if callbacks.on_epoch_end:
                for epoch in range(config.num_epoch):
                    train_loss = results_epoch["loss_train"][epoch].item()
                    val_loss = results_epoch["loss_val"][epoch].item()
                    callbacks.on_epoch_end(epoch + 1, train_loss, val_loss, {})

            # 评估
            model.eval()

            def eval_metrics(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
                with torch.no_grad():
                    U_pred, _, _ = model(inputs=x)
                    pred_soh = 1.0 - U_pred
                    y_soh = 1.0 - y

                    rmspe = torch.sqrt(
                        torch.mean(((pred_soh - y_soh) / y_soh) ** 2)
                    ).item()
                    mse = torch.mean((pred_soh - y_soh) ** 2).item()

                    ss_res = torch.sum((pred_soh - y_soh) ** 2)
                    ss_tot = torch.sum((y_soh - torch.mean(y_soh)) ** 2)
                    r2 = (1 - ss_res / ss_tot).item()

                    return {"RMSPE": rmspe, "MSE": mse, "R2": r2}

            metrics_train = eval_metrics(inputs_train, targets_train)
            metrics_val = eval_metrics(inputs_val, targets_val)
            metrics_test = eval_metrics(inputs_test, targets_test)

            metric_mean["train"][l_idx, n_idx] = metrics_train["RMSPE"]
            metric_mean["val"][l_idx, n_idx] = metrics_val["RMSPE"]
            metric_mean["test"][l_idx, n_idx] = metrics_test["RMSPE"]

            # 保存最佳模型
            if metrics_val["RMSPE"] < best_rmspe:
                best_rmspe = metrics_val["RMSPE"]
                best_model = model
                best_results = {
                    "train": metrics_train,
                    "val": metrics_val,
                    "test": metrics_test,
                }

            if callbacks.on_hyperparameter_search:
                callbacks.on_hyperparameter_search(
                    l_idx, n_idx, num_l, num_n, metrics_test
                )

            _log(
                "INFO",
                f"测试集 RMSPE: {metrics_test['RMSPE']:.6f}, R²: {metrics_test['R2']:.6f}",
            )

    # 3. 返回结果
    results = {
        "config": config.to_dict(),
        "metric_mean": metric_mean,
        "best_model_state": best_model.state_dict() if best_model else None,
        "best_results": best_results,
    }

    if callbacks.on_training_end:
        callbacks.on_training_end(results)

    _log("INFO", "训练完成")
    return results
