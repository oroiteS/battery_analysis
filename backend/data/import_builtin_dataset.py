#!/usr/bin/env python3
"""
导入内置Severson电池数据集到数据库

此脚本将 power_soh/SeversonBattery.mat 文件解析并导入到数据库中
数据结构：所有电池的循环数据平铺存储，通过 Num_Cycles_Flt 识别每个电池的循环数
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.io

# 添加项目根目录到路径（必须在导入 src 模块之前）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models import SessionLocal, Dataset, CycleData, BatteryUnit, Base, engine  # noqa: E402


def load_mat_file(file_path: str):
    """加载.mat文件"""
    try:
        data = scipy.io.loadmat(file_path)
        print("使用 scipy.io.loadmat 成功加载文件")
        return data
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            data = {}
            for key in f.keys():
                if key not in ["#refs#", "#subsystem#"]:
                    data[key] = np.array(f[key])
            return data


def import_dataset(auto_mode: bool = False):
    """
    导入内置数据集
    
    Args:
        auto_mode: 如果为True，则在数据已存在时自动跳过，不进行交互询问
    """

    # 确保表存在
    print("正在检查并创建数据库表...")
    Base.metadata.create_all(bind=engine)

    # 数据文件路径
    mat_file = project_root / "data" / "SeversonBattery.mat"

    if not mat_file.exists():
        print(f"错误: 数据文件不存在: {mat_file}")
        return False

    # 创建数据库会话
    db = SessionLocal()

    try:
        # 检查是否已存在内置数据集
        existing_dataset = (
            db.query(Dataset).filter(Dataset.source_type == "BUILTIN").first()
        )

        if existing_dataset:
            print(f"\n内置数据集已存在 (ID: {existing_dataset.id})")
            
            if auto_mode:
                print("自动模式：检测到数据已存在，跳过导入。")
                return True
            
            response = input("是否删除现有数据并重新导入? (y/N): ")
            if response.lower() != "y":
                print("取消导入")
                return True  # 返回True表示状态符合预期（数据存在）

            # 删除现有数据集
            print("删除现有数据集...")
            # 手动删除关联数据（如果没有设置级联删除）
            for battery in existing_dataset.batteries:
                db.query(CycleData).filter(CycleData.battery_id == battery.id).delete()
            db.query(BatteryUnit).filter(
                BatteryUnit.dataset_id == existing_dataset.id
            ).delete()
            db.delete(existing_dataset)
            db.commit()

        print(f"开始加载数据文件: {mat_file}")
        data = load_mat_file(str(mat_file))

        # 提取数据（所有电池的数据平铺在一起）
        features = data["Features_mov_Flt"]  # shape: (total_cycles, 8)
        rul = data["RUL_Flt"].flatten()  # shape: (total_cycles,)
        pcl = data["PCL_Flt"].flatten()  # shape: (total_cycles,)
        cycles = data["Cycles_Flt"].flatten()  # shape: (total_cycles,)
        num_cycles_per_battery = data["Num_Cycles_Flt"].flatten()  # shape: (num_batteries,)

        # 训练/测试索引（1-based索引，需要转换为0-based）
        train_ind = data.get("train_ind", np.array([[]])).flatten() - 1
        test_ind = data.get("test_ind", np.array([[]])).flatten() - 1

        print("数据维度:")
        print(f"  Features: {features.shape}")
        print(f"  RUL: {rul.shape}")
        print(f"  PCL: {pcl.shape}")
        print(f"  Cycles: {cycles.shape}")
        print(f"  Num_Cycles_Flt: {num_cycles_per_battery.shape}")

        # 获取电池数量
        num_batteries = len(num_cycles_per_battery)
        total_cycles_count = features.shape[0]
        print(f"\n总共 {num_batteries} 个电池单元")
        print(f"总循环记录数: {total_cycles_count}")

        # 创建内置数据集
        print("\n创建内置数据集记录...")
        dataset = Dataset(
            owner_user_id=None,
            source_type="BUILTIN",
            name="Severson Battery Dataset",
            upload_id=None,
            feature_schema={
                "features": [
                    "feature_1",
                    "feature_2",
                    "feature_3",
                    "feature_4",
                    "feature_5",
                    "feature_6",
                    "feature_7",
                    "feature_8",
                ],
                "description": "8个移动平均滤波后的电池特征",
            },
            created_at=datetime.now(timezone.utc),
        )
        db.add(dataset)
        db.flush()  # 获取dataset.id

        print(f"数据集创建成功 (ID: {dataset.id})")

        # 导入每个电池的数据
        print("\n开始导入电池数据...")

        # 计算每个电池在数组中的起始索引
        start_idx = 0
        total_imported_cycles = 0

        for battery_idx in range(num_batteries):
            # 该电池的循环数
            num_cycles = int(num_cycles_per_battery[battery_idx])
            end_idx = start_idx + num_cycles

            if num_cycles == 0:
                print(f"  跳过电池 {battery_idx + 1}: 无有效数据")
                continue

            # 提取该电池的数据
            battery_features = features[start_idx:end_idx, :]  # (num_cycles, 8)
            battery_rul = rul[start_idx:end_idx]
            battery_pcl = pcl[start_idx:end_idx]
            battery_cycles = cycles[start_idx:end_idx]

            # 确定分组标签
            group_tag = None
            if len(train_ind) > 0 and battery_idx in train_ind:
                group_tag = "train"
            elif len(test_ind) > 0 and battery_idx in test_ind:
                group_tag = "test"

            # 创建电池单元记录
            battery_unit = BatteryUnit(
                dataset_id=dataset.id,
                battery_code=f"B{battery_idx + 1:03d}",  # B001, B002, ...
                group_tag=group_tag,
                total_cycles=num_cycles,
                nominal_capacity=1.1,  # 标称容量
            )
            db.add(battery_unit)
            db.flush()  # 获取battery_unit.id

            # 批量创建循环数据
            cycle_records = []
            for cycle_idx in range(num_cycles):
                cycle_record = CycleData(
                    battery_id=battery_unit.id,
                    cycle_num=int(battery_cycles[cycle_idx]),
                    feature_1=float(battery_features[cycle_idx, 0]),
                    feature_2=float(battery_features[cycle_idx, 1]),
                    feature_3=float(battery_features[cycle_idx, 2]),
                    feature_4=float(battery_features[cycle_idx, 3]),
                    feature_5=float(battery_features[cycle_idx, 4]),
                    feature_6=float(battery_features[cycle_idx, 5]),
                    feature_7=float(battery_features[cycle_idx, 6]),
                    feature_8=float(battery_features[cycle_idx, 7]),
                    pcl=float(battery_pcl[cycle_idx]),
                    rul=int(battery_rul[cycle_idx]),
                )
                cycle_records.append(cycle_record)

            # 批量插入
            db.bulk_save_objects(cycle_records)
            total_imported_cycles += num_cycles

            # 更新起始索引
            start_idx = end_idx

            # 每10个电池显示一次进度
            if (battery_idx + 1) % 10 == 0:
                print(
                    f"  已处理 {battery_idx + 1}/{num_batteries} 个电池 (累计 {total_imported_cycles} 条循环数据)..."
                )

        # 提交事务
        db.commit()

        print("\n✅ 导入成功!")
        print(f"  数据集ID: {dataset.id}")
        print(f"  电池数量: {num_batteries}")
        print(f"  总循环数: {total_imported_cycles}")
        print(f"  训练集电池: {len(train_ind)}")
        print(f"  测试集电池: {len(test_ind)}")

        return True

    except Exception as e:
        db.rollback()
        print(f"\n❌ 导入失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Severson 电池数据集导入工具")
    parser.add_argument("--auto", action="store_true", help="自动模式：如果数据已存在则跳过，不询问")
    args = parser.parse_args()

    print("=" * 60)
    print("Severson 电池数据集导入工具")
    print("=" * 60)

    success = import_dataset(auto_mode=args.auto)

    if success:
        print("\n数据导入完成（或已存在）！")
        sys.exit(0)
    else:
        print("\n数据导入失败！")
        sys.exit(1)
