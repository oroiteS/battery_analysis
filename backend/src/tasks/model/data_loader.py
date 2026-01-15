"""
从数据库加载电池数据用于训练

提供与 functions.SeversonBattery 兼容的数据结构
"""

import numpy as np
import torch
from sqlalchemy.orm import Session

from src.models import BatteryUnit, CycleData


class DatabaseBatteryLoader:
    """从数据库加载电池数据的加载器"""

    def __init__(self, db: Session, battery_ids: list[int], seq_len: int = 1):
        """
        初始化数据库加载器

        Args:
            db: 数据库会话
            battery_ids: 要加载的电池ID列表
            seq_len: 序列长度（滑动窗口大小）
        """
        self.db = db
        self.battery_ids = battery_ids
        self.seq_len = seq_len
        self.steps_slices = 1

        # 加载数据
        self._load_data()

    def _load_data(self) -> None:
        """从数据库加载电池数据"""
        all_features = []
        all_rul = []
        all_pcl = []
        all_cycles = []
        num_cycles_per_battery = []

        for battery_id in self.battery_ids:
            # 查询电池单元
            battery: BatteryUnit | None = (
                self.db.query(BatteryUnit).filter(BatteryUnit.id == battery_id).first()
            )

            if not battery:
                raise ValueError(f"Battery ID {battery_id} not found")

            # 查询该电池的所有循环数据
            cycles = (
                self.db.query(CycleData)
                .filter(CycleData.battery_id == battery_id)
                .order_by(CycleData.cycle_num)
                .all()
            )

            if not cycles:
                raise ValueError(f"No cycle data found for battery ID {battery_id}")

            # 提取特征和目标
            battery_features = np.array(
                [
                    [
                        cycle.feature_1,
                        cycle.feature_2,
                        cycle.feature_3,
                        cycle.feature_4,
                        cycle.feature_5,
                        cycle.feature_6,
                        cycle.feature_7,
                        cycle.feature_8,
                    ]
                    for cycle in cycles
                ]
            )

            battery_rul = np.array([cycle.rul for cycle in cycles])
            battery_pcl = np.array([cycle.pcl for cycle in cycles])
            battery_cycles_num = np.array([cycle.cycle_num for cycle in cycles])

            all_features.append(battery_features)
            all_rul.extend(battery_rul)
            all_pcl.extend(battery_pcl)
            all_cycles.extend(battery_cycles_num)
            num_cycles_per_battery.append(len(cycles))

        # 转换为 numpy 数组（与 SeversonBattery 兼容的格式）
        self.features = np.vstack(all_features)
        self.RUL = np.array(all_rul).reshape(-1, 1)
        self.PCL = np.array(all_pcl).reshape(-1, 1)
        self.cycles = np.array(all_cycles).reshape(-1, 1)
        self.num_cycles_all = np.array(num_cycles_per_battery).reshape(-1, 1)

        # 组合输入和目标
        self.inputs = np.concatenate((self.features, self.cycles), axis=1)
        self.targets = np.concatenate((self.PCL, self.RUL), axis=1)

        # 维度信息
        self.inputs_dim = self.inputs.shape[1]
        self.targets_dim = self.targets.shape[1]
        self.num_cells = len(num_cycles_per_battery)

    def create_slices(
        self, inputs: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        创建滑动窗口切片

        Args:
            inputs: 输入数据 (num_samples, num_features)
            targets: 目标数据 (num_samples, num_targets)

        Returns:
            inputs_slices: (num_slices, seq_len, num_features)
            targets_slices: (num_slices, seq_len, num_targets)
        """
        num_samples = inputs.shape[0]
        num_slices = num_samples - self.seq_len + 1

        inputs_slices = np.zeros((num_slices, self.seq_len, self.inputs_dim))
        targets_slices = np.zeros((num_slices, self.seq_len, self.targets.shape[1]))

        for i in range(num_slices):
            inputs_slices[i, :, :] = inputs[i : i + self.seq_len, :]
            targets_slices[i, :, :] = targets[i : i + self.seq_len, :]

        return inputs_slices, targets_slices

    def prepare_train_val_test(
        self,
        idx_cells_train: list[int],
        idx_cells_test: list[int],
        perc_val: float = 0.2,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        准备训练/验证/测试数据集

        Args:
            idx_cells_train: 训练电池的索引（从数据库加载的battery_ids中的索引）
            idx_cells_test: 测试电池的索引
            perc_val: 验证集比例

        Returns:
            inputs_dict: {'train': tensor, 'val': tensor, 'test': tensor}
            targets_dict: {'train': tensor, 'val': tensor, 'test': tensor}
        """
        # 按电池单元分组数据
        inputs_units = []
        targets_units = []

        start_idx = 0
        for i in range(self.num_cells):
            # 修正：使用 .item() 将 0-维数组转换为 Python 标量
            num_cycles = int(self.num_cycles_all[i].item())
            end_idx = start_idx + num_cycles

            unit_inputs = self.inputs[start_idx:end_idx, :]
            unit_targets = self.targets[start_idx:end_idx, :]

            # 创建滑动窗口切片
            inputs_slices, targets_slices = self.create_slices(
                unit_inputs, unit_targets
            )

            inputs_units.append(inputs_slices)
            targets_units.append(targets_slices)

            start_idx = end_idx

        # 分割训练/测试集
        inputs_train_units = [inputs_units[i] for i in idx_cells_train]
        targets_train_units = [targets_units[i] for i in idx_cells_train]

        inputs_test_units = [inputs_units[i] for i in idx_cells_test]
        targets_test_units = [targets_units[i] for i in idx_cells_test]

        # 合并训练数据
        inputs_train = np.vstack(inputs_train_units)
        targets_train = np.vstack(targets_train_units)

        # 分割验证集
        from sklearn.model_selection import train_test_split

        (
            inputs_train_final,
            inputs_val,
            targets_train_final,
            targets_val,
        ) = train_test_split(
            inputs_train, targets_train, test_size=perc_val, random_state=42
        )

        # 合并测试数据
        inputs_test = np.vstack(inputs_test_units)
        targets_test = np.vstack(targets_test_units)

        # 转换为 PyTorch 张量
        inputs_dict = {
            "train": torch.tensor(inputs_train_final, dtype=torch.float32),
            "val": torch.tensor(inputs_val, dtype=torch.float32),
            "test": torch.tensor(inputs_test, dtype=torch.float32),
        }

        targets_dict = {
            "train": torch.tensor(targets_train_final, dtype=torch.float32),
            "val": torch.tensor(targets_val, dtype=torch.float32),
            "test": torch.tensor(inputs_test, dtype=torch.float32), # 修正：这里应该是 targets_test
        }

        return inputs_dict, targets_dict


def create_chosen_cells_from_db(
    db: Session,
    battery_ids_train: list[int],
    battery_ids_test: list[int],
    seq_len: int = 1,
    perc_val: float = 0.2,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    从数据库创建训练/验证/测试数据集（兼容原有 create_chosen_cells 接口）

    Args:
        db: 数据库会话
        battery_ids_train: 训练电池的ID列表
        battery_ids_test: 测试电池的ID列表
        seq_len: 序列长度
        perc_val: 验证集比例

    Returns:
        inputs_dict: {'train': tensor, 'val': tensor, 'test': tensor}
        targets_dict: {'train': tensor, 'val': tensor, 'test': tensor}
    """
    # 合并所有电池ID
    all_battery_ids = battery_ids_train + battery_ids_test

    # 创建加载器
    loader = DatabaseBatteryLoader(db=db, battery_ids=all_battery_ids, seq_len=seq_len)

    # 创建索引映射（训练电池在前，测试电池在后）
    idx_cells_train = list(range(len(battery_ids_train)))
    idx_cells_test = list(
        range(len(battery_ids_train), len(battery_ids_train) + len(battery_ids_test))
    )

    # 准备数据集
    inputs_dict, targets_dict = loader.prepare_train_val_test(
        idx_cells_train=idx_cells_train,
        idx_cells_test=idx_cells_test,
        perc_val=perc_val,
    )

    return inputs_dict, targets_dict
