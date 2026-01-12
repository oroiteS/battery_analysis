"""
异步解析用户上传的电池数据文件

支持格式: MAT, CSV, XLSX
数据结构必须与内置数据集一致
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.io

from src.models import BatteryUnit, CycleData, Dataset, DataUpload, SessionLocal


class ParseError(Exception):
    pass


def parse_mat_file(file_path: Path) -> dict:
    """解析 MAT 文件"""
    try:
        data = scipy.io.loadmat(str(file_path))
    except NotImplementedError:
        # MATLAB v7.3 格式
        with h5py.File(file_path, "r") as f:
            data = {}
            for key in f.keys():
                if key not in ["#refs#", "#subsystem#"]:
                    data[key] = np.array(f[key])

    # 验证必需字段
    required_fields = [
        "Features_mov_Flt",
        "RUL_Flt",
        "PCL_Flt",
        "Cycles_Flt",
        "Num_Cycles_Flt",
    ]
    missing_fields = [f for f in required_fields if f not in data]

    if missing_fields:
        raise ParseError(f"MAT文件缺少必需字段: {', '.join(missing_fields)}")

    # 提取数据
    features = data["Features_mov_Flt"]
    rul = data["RUL_Flt"].flatten()
    pcl = data["PCL_Flt"].flatten()
    cycles = data["Cycles_Flt"].flatten()
    num_cycles_per_battery = data["Num_Cycles_Flt"].flatten()

    # 验证数据形状
    total_cycles = features.shape[0]
    if features.shape[1] != 8:
        raise ParseError(f"Features_mov_Flt必须有8列，当前: {features.shape[1]}列")

    if (
        len(rul) != total_cycles
        or len(pcl) != total_cycles
        or len(cycles) != total_cycles
    ):
        raise ParseError(
            "RUL_Flt, PCL_Flt, Cycles_Flt的长度必须与Features_mov_Flt的行数一致"
        )

    return {
        "features": features,
        "rul": rul,
        "pcl": pcl,
        "cycles": cycles,
        "num_cycles_per_battery": num_cycles_per_battery,
    }


def parse_csv_or_excel(file_path: Path) -> dict:
    """解析 CSV 或 XLSX 文件"""
    # 读取文件
    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        raise ParseError(f"不支持的文件格式: {file_path.suffix}")

    # 验证必需列
    required_columns = [
        "battery_code",
        "cycle_num",
        "feature_1",
        "feature_2",
        "feature_3",
        "feature_4",
        "feature_5",
        "feature_6",
        "feature_7",
        "feature_8",
        "pcl",
        "rul",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ParseError(f"文件缺少必需列: {', '.join(missing_columns)}")

    # 按 battery_code 分组
    battery_codes = df["battery_code"].unique().tolist()

    # 重组为与 MAT 文件相同的结构
    all_features = []
    all_rul = []
    all_pcl = []
    all_cycles = []
    num_cycles_per_battery = []

    for battery_code in battery_codes:
        battery_df = df[df["battery_code"] == battery_code].sort_values(by="cycle_num")  # type: ignore[call-overload]

        # 提取特征
        features = battery_df[
            [
                "feature_1",
                "feature_2",
                "feature_3",
                "feature_4",
                "feature_5",
                "feature_6",
                "feature_7",
                "feature_8",
            ]
        ].values

        all_features.append(features)
        all_rul.extend(battery_df["rul"].values)
        all_pcl.extend(battery_df["pcl"].values)
        all_cycles.extend(battery_df["cycle_num"].values)
        num_cycles_per_battery.append(len(battery_df))

    return {
        "features": np.vstack(all_features),
        "rul": np.array(all_rul),
        "pcl": np.array(all_pcl),
        "cycles": np.array(all_cycles),
        "num_cycles_per_battery": np.array(num_cycles_per_battery),
        "battery_codes": battery_codes,  # CSV/XLSX 包含电池编号
    }


def parse_uploaded_file(upload_id: int) -> dict[str, Any]:
    """
    解析上传的文件并插入数据库

    Args:
        upload_id: DataUpload 记录的 ID
    """
    from src.config import settings

    db = SessionLocal()
    upload: DataUpload | None = None

    try:
        # 获取上传记录
        upload = db.query(DataUpload).filter(DataUpload.id == upload_id).first()
        if not upload:
            raise ParseError(f"上传记录不存在: ID={upload_id}")

        # 更新状态为处理中
        from sqlalchemy import update

        db.execute(
            update(DataUpload)
            .where(DataUpload.id == upload_id)
            .values(status="PROCESSING")
        )
        db.commit()

        # 拼接完整路径（stored_path 是相对路径）
        file_path = settings.UPLOAD_PATH / str(upload.stored_path)

        if not file_path.exists():
            raise ParseError(f"文件不存在: {file_path}")

        # 根据文件格式解析
        file_ext = file_path.suffix.lower()

        battery_codes: list[Any] | None
        if file_ext == ".mat":
            parsed_data = parse_mat_file(file_path)
            battery_codes = None  # MAT文件自动生成编号
        elif file_ext in [".csv", ".xlsx", ".xls"]:
            parsed_data = parse_csv_or_excel(file_path)
            battery_codes = parsed_data.pop("battery_codes")
        else:
            raise ParseError(
                f"不支持的文件格式: {file_ext}。支持的格式: .mat, .csv, .xlsx"
            )

        # 提取数据
        features = parsed_data["features"]
        rul = parsed_data["rul"]
        pcl = parsed_data["pcl"]
        cycles = parsed_data["cycles"]
        num_cycles_per_battery = parsed_data["num_cycles_per_battery"]

        num_batteries = len(num_cycles_per_battery)

        # 创建数据集
        dataset = Dataset(
            owner_user_id=upload.user_id,
            source_type="UPLOAD",
            name=f"Upload_{upload.original_filename}",
            upload_id=upload.id,
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
                "description": f"用户上传数据集 - {upload.original_filename}",
            },
            created_at=datetime.now(timezone.utc),
        )
        db.add(dataset)
        db.flush()

        # 导入电池数据
        start_idx = 0
        total_imported_cycles = 0

        for battery_idx in range(num_batteries):
            num_cycles = int(num_cycles_per_battery[battery_idx])
            end_idx = start_idx + num_cycles

            if num_cycles == 0:
                continue

            # 提取该电池的数据
            battery_features = features[start_idx:end_idx, :]
            battery_rul = rul[start_idx:end_idx]
            battery_pcl = pcl[start_idx:end_idx]
            battery_cycles = cycles[start_idx:end_idx]

            # 电池编号
            if battery_codes is not None:
                battery_code = str(battery_codes[battery_idx])
            else:
                battery_code = f"B{battery_idx + 1:03d}"

            # 创建电池单元
            battery_unit = BatteryUnit(
                dataset_id=dataset.id,
                battery_code=battery_code,
                group_tag=None,
                total_cycles=num_cycles,
                nominal_capacity=1.1,
            )
            db.add(battery_unit)
            db.flush()

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

            db.bulk_save_objects(cycle_records)
            total_imported_cycles += num_cycles
            start_idx = end_idx

        # 更新上传状态为成功
        from sqlalchemy import update

        db.execute(
            update(DataUpload)
            .where(DataUpload.id == upload.id)
            .values(
                status="SUCCEEDED",
                processed_at=datetime.now(timezone.utc),
                error_message=None,
            )
        )
        db.commit()

        return {
            "success": True,
            "dataset_id": dataset.id,
            "num_batteries": num_batteries,
            "total_cycles": total_imported_cycles,
        }

    except ParseError as e:
        # 解析错误
        db.rollback()
        if upload:
            from sqlalchemy import update

            db.execute(
                update(DataUpload)
                .where(DataUpload.id == upload.id)
                .values(
                    status="FAILED",
                    error_message=str(e),
                    processed_at=datetime.now(timezone.utc),
                )
            )
            db.commit()
        return {"success": False, "error": str(e)}

    except Exception as e:
        # 其他错误
        db.rollback()
        if upload:
            from sqlalchemy import update

            db.execute(
                update(DataUpload)
                .where(DataUpload.id == upload.id)
                .values(
                    status="FAILED",
                    error_message=f"未知错误: {str(e)}",
                    processed_at=datetime.now(timezone.utc),
                )
            )
            db.commit()
        return {"success": False, "error": f"未知错误: {str(e)}"}

    finally:
        db.close()
