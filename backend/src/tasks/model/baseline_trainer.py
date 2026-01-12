"""
Baseline 算法训练执行器

支持从数据库加载数据，接受外部参数，并提供训练进度回调
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
from sqlalchemy.orm import Session
from torch import optim
from torch.utils.data import DataLoader

from src.tasks.model.data_loader import create_chosen_cells_from_db
from src.utils.training_utils import (
    DataDrivenNN,
    My_loss,
    TensorDataset,
    standardize_tensor,
    train,
)


@dataclass
class BaselineTrainingConfig:
    """Baseline 训练配置（基于预训练模型参数）"""

    # 数据参数
    seq_len: int = 1
    perc_val: float = 0.2

    # 模型参数（来自预训练模型）
    num_layers: list[int] = field(default_factory=lambda: [2])
    num_neurons: list[int] = field(default_factory=lambda: [128])
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
        }


@dataclass
class TrainingCallbacks:
    """训练回调函数集合"""

    on_epoch_end: Optional[Callable[[int, float, float, dict[str, float]], None]] = None
    on_training_end: Optional[Callable[[dict[str, Any]], None]] = None
    on_hyperparameter_search: Optional[
        Callable[[int, int, int, int, dict[str, float]], None]
    ] = None
    on_log: Optional[Callable[[str, str], None]] = None  # (level, message)
    on_period_end: Optional[Callable[[int, int, float, float, float, float], None]] = (
        None  # (epoch, period, loss, loss_U, loss_F, loss_F_t)
    )


class BaselineTrainer:
    """Baseline 训练执行器"""

    def __init__(
        self,
        db: Session,
        config: BaselineTrainingConfig,
        callbacks: Optional[TrainingCallbacks] = None,
    ):
        """
        初始化训练器

        Args:
            db: 数据库会话
            config: 训练配置
            callbacks: 回调函数
        """
        self.db = db
        self.config = config
        self.callbacks = callbacks or TrainingCallbacks()
        self.device = torch.device(config.device)

        self._log("INFO", f"初始化 Baseline 训练器，设备: {self.device}")
        self._log("INFO", f"训练配置: {config.to_dict()}")

    def _log(self, level: str, message: str) -> None:
        """记录日志"""
        if self.callbacks.on_log:
            self.callbacks.on_log(level, message)
        else:
            print(f"[{level}] {message}")

    def train(
        self,
        battery_ids_train: list[int],
        battery_ids_test: list[int],
    ) -> dict[str, Any]:
        """
        执行训练

        Args:
            battery_ids_train: 训练电池ID列表
            battery_ids_test: 测试电池ID列表

        Returns:
            训练结果字典
        """
        self._log(
            "INFO",
            f"开始训练，训练电池: {battery_ids_train}, 测试电池: {battery_ids_test}",
        )

        # 1. 加载数据
        self._log("INFO", "从数据库加载数据...")
        inputs_dict, targets_dict = create_chosen_cells_from_db(
            db=self.db,
            battery_ids_train=battery_ids_train,
            battery_ids_test=battery_ids_test,
            seq_len=self.config.seq_len,
            perc_val=self.config.perc_val,
        )

        inputs_train = inputs_dict["train"].to(self.device)
        inputs_val = inputs_dict["val"].to(self.device)
        inputs_test = inputs_dict["test"].to(self.device)

        targets_train = targets_dict["train"][:, :, 0:1].to(self.device)  # PCL only
        targets_val = targets_dict["val"][:, :, 0:1].to(self.device)
        targets_test = targets_dict["test"][:, :, 0:1].to(self.device)

        self._log(
            "INFO",
            f"数据加载完成: 训练集={inputs_train.shape[0]}, 验证集={inputs_val.shape[0]}, 测试集={inputs_test.shape[0]}",
        )

        # 2. 超参数搜索
        metric_mean, metric_std, best_model, best_results = self._hyperparameter_search(
            inputs_train=inputs_train,
            targets_train=targets_train,
            inputs_val=inputs_val,
            targets_val=targets_val,
            inputs_test=inputs_test,
            targets_test=targets_test,
        )

        # 3. 返回结果
        results = {
            "config": self.config.to_dict(),
            "metric_mean": metric_mean,
            "metric_std": metric_std,
            "best_model_state": best_model.state_dict() if best_model else None,
            "best_results": best_results,
        }

        if self.callbacks.on_training_end:
            self.callbacks.on_training_end(results)

        self._log("INFO", "训练完成")
        return results

    def _hyperparameter_search(
        self,
        inputs_train: torch.Tensor,
        targets_train: torch.Tensor,
        inputs_val: torch.Tensor,
        targets_val: torch.Tensor,
        inputs_test: torch.Tensor,
        targets_test: torch.Tensor,
    ) -> tuple[dict, dict, Any, dict]:
        """执行超参数搜索"""
        metric_mean = {
            "train": np.zeros(
                (len(self.config.num_layers), len(self.config.num_neurons))
            ),
            "val": np.zeros(
                (len(self.config.num_layers), len(self.config.num_neurons))
            ),
            "test": np.zeros(
                (len(self.config.num_layers), len(self.config.num_neurons))
            ),
        }
        metric_std = {
            "train": np.zeros(
                (len(self.config.num_layers), len(self.config.num_neurons))
            ),
            "val": np.zeros(
                (len(self.config.num_layers), len(self.config.num_neurons))
            ),
            "test": np.zeros(
                (len(self.config.num_layers), len(self.config.num_neurons))
            ),
        }

        best_model = None
        best_results = {}

        for l_idx, num_l in enumerate(self.config.num_layers):
            for n_idx, num_n in enumerate(self.config.num_neurons):
                layers = num_l * [num_n]

                self._log(
                    "INFO",
                    f"超参数搜索 [{l_idx + 1}/{len(self.config.num_layers)}, {n_idx + 1}/{len(self.config.num_neurons)}]: "
                    f"层数={num_l}, 神经元={num_n}",
                )

                # 设置随机种子
                np.random.seed(self.config.random_seed)
                torch.manual_seed(self.config.random_seed)

                metric_rounds = {
                    "train": np.zeros(self.config.num_rounds),
                    "val": np.zeros(self.config.num_rounds),
                    "test": np.zeros(self.config.num_rounds),
                }

                for round_idx in range(self.config.num_rounds):
                    model, results_epoch = self._train_single_round(
                        inputs_train=inputs_train,
                        targets_train=targets_train,
                        inputs_val=inputs_val,
                        targets_val=targets_val,
                        layers=layers,
                    )

                    # 评估模型
                    metrics = self._evaluate_model(
                        model=model,
                        inputs_train=inputs_train,
                        targets_train=targets_train,
                        inputs_val=inputs_val,
                        targets_val=targets_val,
                        inputs_test=inputs_test,
                        targets_test=targets_test,
                    )

                    metric_rounds["train"][round_idx] = metrics["train"]["RMSPE"]
                    metric_rounds["val"][round_idx] = metrics["val"]["RMSPE"]
                    metric_rounds["test"][round_idx] = metrics["test"]["RMSPE"]

                    # 保存第一轮的模型作为最佳模型
                    if l_idx == 0 and n_idx == 0 and round_idx == 0:
                        best_model = model
                        best_results = metrics

                # 计算平均指标
                metric_mean["train"][l_idx, n_idx] = np.mean(metric_rounds["train"])
                metric_mean["val"][l_idx, n_idx] = np.mean(metric_rounds["val"])
                metric_mean["test"][l_idx, n_idx] = np.mean(metric_rounds["test"])

                metric_std["train"][l_idx, n_idx] = np.std(metric_rounds["train"])
                metric_std["val"][l_idx, n_idx] = np.std(metric_rounds["val"])
                metric_std["test"][l_idx, n_idx] = np.std(metric_rounds["test"])

                # 回调
                if self.callbacks.on_hyperparameter_search:
                    self.callbacks.on_hyperparameter_search(
                        l_idx,
                        n_idx,
                        num_l,
                        num_n,
                        {
                            "train_rmspe": metric_mean["train"][l_idx, n_idx],
                            "val_rmspe": metric_mean["val"][l_idx, n_idx],
                            "test_rmspe": metric_mean["test"][l_idx, n_idx],
                        },
                    )

        return metric_mean, metric_std, best_model, best_results

    def _train_single_round(
        self,
        inputs_train: torch.Tensor,
        targets_train: torch.Tensor,
        inputs_val: torch.Tensor,
        targets_val: torch.Tensor,
        layers: list[int],
    ) -> tuple[Any, dict]:
        """训练单轮"""
        inputs_dim = inputs_train.shape[2]
        outputs_dim = 1

        # 标准化
        _, mean_inputs_train, std_inputs_train = standardize_tensor(
            inputs_train, mode="fit"
        )
        _, mean_targets_train, std_targets_train = standardize_tensor(
            targets_train, mode="fit"
        )

        # 创建数据加载器
        train_set = TensorDataset(inputs_train, targets_train)
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        # 创建模型
        model = DataDrivenNN(
            seq_len=self.config.seq_len,
            inputs_dim=inputs_dim,
            outputs_dim=outputs_dim,
            layers=layers,
            scaler_inputs=(mean_inputs_train, std_inputs_train),
            scaler_targets=(mean_targets_train, std_targets_train),
            dropout_rate=self.config.dropout_rate,
        ).to(self.device)

        # 损失函数和优化器
        log_sigma_u = torch.zeros(())
        log_sigma_f = torch.zeros(())
        log_sigma_f_t = torch.zeros(())

        criterion = My_loss(mode="Baseline")
        params = [p for p in model.parameters()]
        optimizer = optim.Adam(
            params, lr=self.config.lr, weight_decay=self.config.weight_decay
        )

        # 学习率调度器
        if self.config.lr_scheduler == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.step_size, gamma=self.config.gamma
            )
        elif self.config.lr_scheduler == "CosineAnnealing":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.num_epoch, eta_min=self.config.min_lr
            )
        elif self.config.lr_scheduler == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.gamma,
                patience=10,
                min_lr=self.config.min_lr,
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=self.config.step_size, gamma=self.config.gamma
            )

        # 训练
        model, results_epoch = train(
            num_epoch=self.config.num_epoch,
            batch_size=self.config.batch_size,
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

        # 每个 epoch 的回调
        if self.callbacks.on_epoch_end and results_epoch:
            for epoch in range(len(results_epoch.get("loss_train", []))):
                train_loss = results_epoch["loss_train"][epoch]
                val_loss = results_epoch["loss_val"][epoch]
                self.callbacks.on_epoch_end(
                    epoch,
                    train_loss,
                    val_loss,
                    {},
                )

        return model, results_epoch

    def _evaluate_model(
        self,
        model: Any,
        inputs_train: torch.Tensor,
        targets_train: torch.Tensor,
        inputs_val: torch.Tensor,
        targets_val: torch.Tensor,
        inputs_test: torch.Tensor,
        targets_test: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """评估模型"""
        model.eval()

        results = {}

        for split_name, inputs, targets in [
            ("train", inputs_train, targets_train),
            ("val", inputs_val, targets_val),
            ("test", inputs_test, targets_test),
        ]:
            with torch.no_grad():
                U_pred, _, _ = model(inputs=inputs)
                U_pred_soh = 1.0 - U_pred
                targets_soh = 1.0 - targets

                # 计算指标
                RMSPE = torch.sqrt(
                    torch.mean(((U_pred_soh - targets_soh) / targets_soh) ** 2)
                )
                MSE = torch.mean((U_pred_soh - targets_soh) ** 2)
                SS_res = torch.sum((U_pred_soh - targets_soh) ** 2)
                SS_tot = torch.sum((targets_soh - torch.mean(targets_soh)) ** 2)
                R2 = 1 - (SS_res / SS_tot)

                results[split_name] = {
                    "RMSPE": RMSPE.item(),
                    "MSE": MSE.item(),
                    "R2": R2.item(),
                }

        return results


def train_baseline_from_database(
    db: Session,
    battery_ids_train: list[int],
    battery_ids_test: list[int],
    config: Optional[BaselineTrainingConfig] = None,
    callbacks: Optional[TrainingCallbacks] = None,
) -> dict[str, Any]:
    """
    从数据库训练 Baseline 模型（便捷函数）

    Args:
        db: 数据库会话
        battery_ids_train: 训练电池ID列表
        battery_ids_test: 测试电池ID列表
        config: 训练配置（如果为None则使用默认配置）
        callbacks: 回调函数

    Returns:
        训练结果字典
    """
    if config is None:
        config = BaselineTrainingConfig()

    trainer = BaselineTrainer(db=db, config=config, callbacks=callbacks)
    return trainer.train(
        battery_ids_train=battery_ids_train,
        battery_ids_test=battery_ids_test,
    )
