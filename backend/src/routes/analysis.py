import pandas as pd
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.src.models import get_db, CycleData, BatteryUnit
from backend.src.routes.auth import get_current_user
from backend.src.models import User

# 定义 Pydantic 响应模型 (与 OpenAPI 对应)
from pydantic import BaseModel


class FeatureStat(BaseModel):
    feature_name: str
    mean: float
    variance: float
    min_val: float
    max_val: float
    corr_rul: float | None
    corr_pcl: float | None


class TrendData(BaseModel):
    cycles: List[int]
    values: List[float]


class ScatterData(BaseModel):
    points: List[List[float]]  # [[x, y], [x, y]]


class PCLDistribution(BaseModel):
    pcl_values: List[float]


class CorrelationMatrix(BaseModel):
    features: List[str]
    matrix: List[List[float]]


router = APIRouter(prefix="/api/batteries", tags=["Analysis"])


# 辅助函数：将电池数据加载为 DataFrame
def get_battery_df(db: Session, battery_id: int) -> pd.DataFrame:
    # 1. 检查电池是否存在 (且未删除)
    stmt = select(CycleData).where(
        CycleData.battery_id == battery_id,
        CycleData.deleted_at.is_(None)
    ).order_by(CycleData.cycle_num)

    results = db.execute(stmt).scalars().all()

    if not results:
        raise HTTPException(status_code=404, detail="No cycle data found for this battery")

    # 2. 转换为 Pandas DataFrame
    data = [
        {
            "cycle_num": r.cycle_num,
            "feature_1": r.feature_1, "feature_2": r.feature_2,
            "feature_3": r.feature_3, "feature_4": r.feature_4,
            "feature_5": r.feature_5, "feature_6": r.feature_6,
            "feature_7": r.feature_7, "feature_8": r.feature_8,
            "pcl": r.pcl,
            "rul": r.rul
        }
        for r in results
    ]
    return pd.DataFrame(data)


# --- 接口实现 ---

# 1. 获取 8x6 统计表格
@router.get("/{battery_id}/stats", response_model=List[FeatureStat])
def get_battery_stats(
        battery_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    df = get_battery_df(db, battery_id)

    stats_list = []
    feature_cols = [f"feature_{i}" for i in range(1, 9)]

    for col in feature_cols:
        # 计算基础统计量
        mean_val = float(df[col].mean())
        var_val = float(df[col].var())
        min_val = float(df[col].min())
        max_val = float(df[col].max())

        # 计算相关性 (注意处理空值)
        corr_rul = float(df[col].corr(df['rul'])) if 'rul' in df and not df['rul'].isnull().all() else 0.0
        corr_pcl = float(df[col].corr(df['pcl'])) if 'pcl' in df and not df['pcl'].isnull().all() else 0.0

        stats_list.append(FeatureStat(
            feature_name=col,
            mean=round(mean_val, 4),
            variance=round(var_val, 4),
            min_val=round(min_val, 4),
            max_val=round(max_val, 4),
            corr_rul=round(corr_rul, 4),
            corr_pcl=round(corr_pcl, 4)
        ))

    return stats_list


# 2. 获取趋势图数据
@router.get("/{battery_id}/trend", response_model=TrendData)
def get_trend_data(
        battery_id: int,
        feature_name: str = Query(..., pattern=r"^feature_[1-8]$"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    # 优化：只查需要的列，不加载整个 DataFrame
    results = db.query(CycleData.cycle_num, getattr(CycleData, feature_name)) \
        .filter(CycleData.battery_id == battery_id, CycleData.deleted_at.is_(None)) \
        .order_by(CycleData.cycle_num).all()

    if not results:
        return TrendData(cycles=[], values=[])

    # 简单的降采样策略：如果数据点超过 2000，每隔 N 个点取一个
    total_points = len(results)
    step = 1
    if total_points > 2000:
        step = total_points // 2000

    sampled_results = results[::step]

    return TrendData(
        cycles=[r[0] for r in sampled_results],
        values=[r[1] for r in sampled_results]
    )


# 3. 获取散点图数据 (RUL vs Feature)
@router.get("/{battery_id}/scatter", response_model=ScatterData)
def get_scatter_data(
        battery_id: int,
        feature_name: str = Query("feature_1", pattern=r"^feature_[1-8]$"),
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    results = db.query(getattr(CycleData, feature_name), CycleData.rul) \
        .filter(
        CycleData.battery_id == battery_id,
        CycleData.deleted_at.is_(None),
        CycleData.rul.isnot(None)  # 过滤掉 RUL 为空的点
    ).all()

    # ECharts 格式: [[x1, y1], [x2, y2]]
    points = [[r[0], r[1]] for r in results]

    # 同样可以做降采样
    if len(points) > 2000:
        step = len(points) // 2000
        points = points[::step]

    return ScatterData(points=points)


# 4. 获取 PCL 分布数据
@router.get("/{battery_id}/pcl-distribution", response_model=PCLDistribution)
def get_pcl_distribution(
        battery_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    results = db.query(CycleData.pcl) \
        .filter(CycleData.battery_id == battery_id, CycleData.deleted_at.is_(None)) \
        .all()

    return PCLDistribution(pcl_values=[r[0] for r in results if r[0] is not None])


# 5. 获取相关性矩阵 (热力图)
@router.get("/{battery_id}/correlation-matrix", response_model=CorrelationMatrix)
def get_correlation_matrix(
        battery_id: int,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user)
):
    df = get_battery_df(db, battery_id)
    feature_cols = [f"feature_{i}" for i in range(1, 9)]

    # 计算相关矩阵
    corr_matrix = df[feature_cols].corr()

    # 填充 NaN (如果有常数列，相关性会是 NaN) 为 0
    corr_matrix = corr_matrix.fillna(0)

    # 转换为 8x8 二维数组
    matrix_data = corr_matrix.values.tolist()

    return CorrelationMatrix(
        features=feature_cols,
        matrix=matrix_data
    )