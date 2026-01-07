import scipy.io
import numpy as np
import pandas as pd
import os
from typing import cast
from numpy.typing import NDArray

# ================= 配置区域 =================
MAT_FILE = 'data/SeversonBattery.mat'  # 替换为你的文件名
# ===========================================

def load_and_analyze():
    if not os.path.exists(MAT_FILE):
        print(f"❌ 错误: 找不到文件 {MAT_FILE}")
        return

    print(f"🔄 正在读取 {MAT_FILE} ...")
    data = scipy.io.loadmat(MAT_FILE)

    # 1. 提取基础数据
    # 特征 (99281, 8)
    features: NDArray[np.float64] = data['Features_mov_Flt']
    # 标量数据 (99281, 1)
    cycles: NDArray[np.float64] = data['Cycles_Flt']
    rul: NDArray[np.float64] = data['RUL_Flt']
    pcl: NDArray[np.float64] = data['PCL_Flt']
    
    # 单元统计 (124, 1)
    num_cycles_per_unit = data['Num_Cycles_Flt'].flatten()
    num_units = len(num_cycles_per_unit)

    # 索引 (修正为0-based)
    idx_train = set(data['train_ind'].flatten() - 1)
    idx_val = set(data['test_ind'].flatten() - 1)
    idx_test = set(data['secondary_test_ind'].flatten() - 1)

    print(f"✅ 读取成功: 共有 {num_units} 个电池单元, 总计 {len(cycles)} 条循环记录。")

    # ====================================================
    # 2. 构建表结构建议
    # ====================================================

    # --- 表 1: 电池单元信息表 (Unit Info) ---
    # 存储每个电池的元数据：ID，属于哪个集，总循环数等
    unit_data = []
    for i in range(num_units):
        # 判断该电池属于哪个数据集
        if i in idx_train: group = 'train'
        elif i in idx_val: group = 'validation'
        elif i in idx_test: group = 'test'
        else: group = 'unknown'
        
        unit_data.append({
            'unit_id': i + 1,        # 电池ID (从1开始)
            'dataset_group': group,  # 训练/验证/测试
            'total_cycles': int(num_cycles_per_unit[i]) # 该电池总共有多少个数据点
        })
    
    df_unit = pd.DataFrame(unit_data)

    # --- 表 2: 详细监测数据表 (Measurements) ---
    # 存储 99281 行详细时序数据
    
    # 关键步骤：生成每一行对应的 battery_unit_id
    # 利用 num_cycles_per_unit [100, 200, ...] 扩展成 [1,1...1, 2,2...2]
    unit_ids_expanded = np.repeat(np.arange(1, num_units + 1), num_cycles_per_unit)
    
    # 构建大表
    feature_columns: list[str] = [f'feature_{j+1}' for j in range(8)]
    df_measure = pd.DataFrame(features, columns=feature_columns)  # type: ignore[call-overload]
    df_measure['unit_id'] = unit_ids_expanded
    df_measure['cycle_num'] = cycles
    df_measure['pcl'] = pcl
    df_measure['rul'] = rul

    # 调整列顺序，把ID放在前面
    cols = ['unit_id', 'cycle_num'] + feature_columns + ['pcl', 'rul']
    df_measure = df_measure[cols]

    # ====================================================
    # 3. 输出预览和SQL设计建议
    # ====================================================
    
    print("\n" + "="*50)
    print("📊 数据预览 (Top 5 Rows)")
    print("="*50)
    print(df_measure.head().to_string())


if __name__ == "__main__":
    load_and_analyze()
