#!/usr/bin/env python3
"""
将 SeversonBattery.mat 转换为 CSV 格式

生成的CSV文件可用于测试上传功能
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


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


def convert_mat_to_csv(
    mat_file_path: Path, output_csv_path: Path, max_batteries: int = 5
):
    """
    将MAT文件转换为CSV格式

    Args:
        mat_file_path: MAT文件路径
        output_csv_path: 输出CSV文件路径
        max_batteries: 最多转换多少个电池（避免文件过大）
    """
    print(f"开始加载MAT文件: {mat_file_path}")
    data = load_mat_file(str(mat_file_path))

    # 提取数据
    features = data["Features_mov_Flt"]
    rul = data["RUL_Flt"].flatten()
    pcl = data["PCL_Flt"].flatten()
    cycles = data["Cycles_Flt"].flatten()
    num_cycles_per_battery = data["Num_Cycles_Flt"].flatten()

    print(f"总电池数: {len(num_cycles_per_battery)}")
    print(f"将转换前 {max_batteries} 个电池到CSV")

    # 准备CSV数据
    csv_data = []
    start_idx = 0

    for battery_idx in range(min(max_batteries, len(num_cycles_per_battery))):
        num_cycles = int(num_cycles_per_battery[battery_idx])
        end_idx = start_idx + num_cycles

        if num_cycles == 0:
            continue

        # 提取该电池的数据
        battery_features = features[start_idx:end_idx, :]
        battery_rul = rul[start_idx:end_idx]
        battery_pcl = pcl[start_idx:end_idx]
        battery_cycles = cycles[start_idx:end_idx]

        battery_code = f"B{battery_idx + 1:03d}"

        # 添加到CSV数据
        for cycle_idx in range(num_cycles):
            row = {
                "battery_code": battery_code,
                "cycle_num": int(battery_cycles[cycle_idx]),
                "feature_1": float(battery_features[cycle_idx, 0]),
                "feature_2": float(battery_features[cycle_idx, 1]),
                "feature_3": float(battery_features[cycle_idx, 2]),
                "feature_4": float(battery_features[cycle_idx, 3]),
                "feature_5": float(battery_features[cycle_idx, 4]),
                "feature_6": float(battery_features[cycle_idx, 5]),
                "feature_7": float(battery_features[cycle_idx, 6]),
                "feature_8": float(battery_features[cycle_idx, 7]),
                "pcl": float(battery_pcl[cycle_idx]),
                "rul": int(battery_rul[cycle_idx]),
            }
            csv_data.append(row)

        start_idx = end_idx
        print(f"  已处理电池 {battery_code}: {num_cycles} 个循环")

    # 转换为DataFrame并保存
    df = pd.DataFrame(csv_data)
    df.to_csv(output_csv_path, index=False)

    print("\n✅ 转换完成!")
    print(f"  输出文件: {output_csv_path}")
    print(f"  总行数: {len(df)}")
    print(f"  电池数: {df['battery_code'].nunique()}")


if __name__ == "__main__":
    # 输入输出路径
    mat_file = project_root.parent / "power_soh" / "SeversonBattery.mat"
    output_csv = project_root / "test" / "sample_data.csv"

    if not mat_file.exists():
        print(f"❌ 错误: MAT文件不存在: {mat_file}")
        sys.exit(1)

    # 确保输出目录存在
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 转换（默认前5个电池，避免文件过大）
    max_batteries = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    convert_mat_to_csv(mat_file, output_csv, max_batteries)

    print("\n使用方法:")
    print("  # 测试上传此CSV文件")
    print("  node test/test_api.js")
    print("  ")
    print("  # 或转换更多电池")
    print("  python data/convert_mat_to_csv.py 10")
