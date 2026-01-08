# src/models.py
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.config import settings

# --- 数据库连接 ---
engine = create_engine(
    settings.DATABASE_URL, pool_pre_ping=True, pool_size=10, max_overflow=20
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- 枚举定义 ---
SourceTypeEnum = Enum("BUILTIN", "UPLOAD", name="source_type_enum")
GroupTagEnum = Enum("train", "val", "test", name="group_tag_enum")
AlgorithmEnum = Enum("BASELINE", "BILSTM", "DEEPHPM", name="algorithm_enum")
JobStatusEnum = Enum(
    "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", name="job_status_enum"
)
LogLevelEnum = Enum("DEBUG", "INFO", "WARNING", "ERROR", name="log_level_enum")
TargetEnum = Enum("RUL", "PCL", "BOTH", name="target_enum")
ExportFormatEnum = Enum("CSV", "XLSX", name="export_format_enum")
UploadStatusEnum = Enum(
    "PENDING", "PROCESSING", "SUCCEEDED", "FAILED", name="upload_status_enum"
)


# --- 1. 用户与认证 ---
class User(Base):
    __tablename__ = "user"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_name = Column(String(50), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    uploads = relationship("DataUpload", back_populates="user")
    datasets = relationship("Dataset", back_populates="owner")
    training_jobs = relationship("TrainingJob", back_populates="user")
    model_versions = relationship("ModelVersion", back_populates="user")
    test_jobs = relationship("TestJob", back_populates="user")


# --- 2. 数据管理 ---
class DataUpload(Base):
    __tablename__ = "data_upload"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    original_filename = Column(String(255), nullable=False)
    stored_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    status = Column(UploadStatusEnum, nullable=False, default="PENDING")
    error_message = Column(String(2000), nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    processed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="uploads")
    dataset = relationship("Dataset", back_populates="upload", uselist=False)


Index("idx_upload_user", DataUpload.user_id)
Index("idx_upload_status", DataUpload.status)


class Dataset(Base):
    __tablename__ = "dataset"
    __table_args__ = (
        UniqueConstraint("upload_id", name="uk_dataset_upload"),
        CheckConstraint(
            "(source_type = 'BUILTIN' AND owner_user_id IS NULL AND upload_id IS NULL) OR "
            "(source_type = 'UPLOAD' AND owner_user_id IS NOT NULL AND upload_id IS NOT NULL)",
            name="chk_builtin_nulls",
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    owner_user_id = Column(BigInteger, ForeignKey("user.id"), nullable=True)
    source_type = Column(SourceTypeEnum, nullable=False)
    name = Column(String(120), nullable=False)
    upload_id = Column(BigInteger, ForeignKey("data_upload.id"), nullable=True)
    feature_schema = Column(JSON, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    owner = relationship("User", back_populates="datasets")
    upload = relationship("DataUpload", back_populates="dataset")
    batteries = relationship("BatteryUnit", back_populates="dataset")
    training_jobs = relationship("TrainingJob", back_populates="dataset")
    test_jobs = relationship("TestJob", back_populates="dataset")


Index("idx_dataset_owner", Dataset.owner_user_id)


class BatteryUnit(Base):
    __tablename__ = "battery_unit"
    __table_args__ = (
        UniqueConstraint("dataset_id", "battery_code", name="uk_dataset_battery"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id = Column(BigInteger, ForeignKey("dataset.id"), nullable=False)
    battery_code = Column(String(64), nullable=False)
    group_tag = Column(GroupTagEnum, nullable=True)
    total_cycles = Column(Integer, nullable=False)
    nominal_capacity = Column(Float, nullable=True)

    dataset = relationship("Dataset", back_populates="batteries")
    cycle_data = relationship("CycleData", back_populates="battery")
    training_job_batteries = relationship(
        "TrainingJobBattery", back_populates="battery"
    )
    test_job_batteries = relationship("TestJobBattery", back_populates="battery")
    test_battery_metrics = relationship(
        "TestJobBatteryMetric", back_populates="battery"
    )
    test_predictions = relationship("TestJobPrediction", back_populates="battery")


Index("idx_battery_dataset", BatteryUnit.dataset_id)


class CycleData(Base):
    __tablename__ = "cycle_data"
    __table_args__ = (
        UniqueConstraint("battery_id", "cycle_num", name="uk_battery_cycle"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    battery_id = Column(BigInteger, ForeignKey("battery_unit.id"), nullable=False)
    cycle_num = Column(Integer, nullable=False)
    feature_1 = Column(Float, nullable=False)
    feature_2 = Column(Float, nullable=False)
    feature_3 = Column(Float, nullable=False)
    feature_4 = Column(Float, nullable=False)
    feature_5 = Column(Float, nullable=False)
    feature_6 = Column(Float, nullable=False)
    feature_7 = Column(Float, nullable=False)
    feature_8 = Column(Float, nullable=False)
    pcl = Column(Float, nullable=True)
    rul = Column(Integer, nullable=True)

    battery = relationship("BatteryUnit", back_populates="cycle_data")


Index("idx_cycle_battery", CycleData.battery_id)


# --- 3. 训练平台 ---
class TrainingJob(Base):
    __tablename__ = "training_job"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    dataset_id = Column(BigInteger, ForeignKey("dataset.id"), nullable=False)
    target = Column(TargetEnum, nullable=False)
    hyperparams = Column(JSON, nullable=True)
    status = Column(JobStatusEnum, nullable=False, default="PENDING")
    progress = Column(Numeric(6, 5), nullable=False, default=0.0)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="training_jobs")
    dataset = relationship("Dataset", back_populates="training_jobs")
    batteries = relationship("TrainingJobBattery", back_populates="job")
    runs = relationship("TrainingJobRun", back_populates="job")


Index("idx_training_user", TrainingJob.user_id)
Index("idx_training_status", TrainingJob.status)
Index("idx_training_dataset", TrainingJob.dataset_id)
Index("idx_training_created", TrainingJob.created_at)


class TrainingJobBattery(Base):
    __tablename__ = "training_job_battery"

    job_id = Column(BigInteger, ForeignKey("training_job.id"), primary_key=True)
    battery_id = Column(BigInteger, ForeignKey("battery_unit.id"), primary_key=True)
    split_role = Column(GroupTagEnum, nullable=False, default="train")

    job = relationship("TrainingJob", back_populates="batteries")
    battery = relationship("BatteryUnit", back_populates="training_job_batteries")


Index("idx_job_battery", TrainingJobBattery.battery_id)


class TrainingJobRun(Base):
    __tablename__ = "training_job_run"
    __table_args__ = (UniqueConstraint("job_id", "algorithm", name="uk_job_algorithm"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(BigInteger, ForeignKey("training_job.id"), nullable=False)
    algorithm = Column(AlgorithmEnum, nullable=False)
    status = Column(JobStatusEnum, nullable=False, default="PENDING")
    current_epoch = Column(Integer, nullable=False, default=0)
    total_epochs = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    job = relationship("TrainingJob", back_populates="runs")
    metrics = relationship("TrainingJobRunMetric", back_populates="run")
    logs = relationship("TrainingJobRunLog", back_populates="run")
    model_version = relationship("ModelVersion", back_populates="run", uselist=False)


Index("idx_run_job", TrainingJobRun.job_id)
Index("idx_run_status", TrainingJobRun.status)


class TrainingJobRunMetric(Base):
    __tablename__ = "training_job_run_metric"
    __table_args__ = (UniqueConstraint("run_id", "epoch", name="uk_run_epoch"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(BigInteger, ForeignKey("training_job_run.id"), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    run = relationship("TrainingJobRun", back_populates="metrics")


Index("idx_metric_run", TrainingJobRunMetric.run_id)


class TrainingJobRunLog(Base):
    __tablename__ = "training_job_run_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    run_id = Column(BigInteger, ForeignKey("training_job_run.id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    level = Column(LogLevelEnum, nullable=False)
    message = Column(String(2000), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    run = relationship("TrainingJobRun", back_populates="logs")


Index("idx_log_run", TrainingJobRunLog.run_id)
Index("idx_log_user", TrainingJobRunLog.user_id)
Index("idx_log_run_time", TrainingJobRunLog.run_id, TrainingJobRunLog.created_at)


# --- 4. 模型版本管理 ---
class ModelVersion(Base):
    __tablename__ = "model_version"
    __table_args__ = (
        UniqueConstraint("run_id", name="uk_model_run"),
        UniqueConstraint("user_id", "name", "version", name="uk_model_version"),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    run_id = Column(BigInteger, ForeignKey("training_job_run.id"), nullable=False)
    algorithm = Column(AlgorithmEnum, nullable=False)
    name = Column(String(120), nullable=False)
    version = Column(String(32), nullable=False)
    config = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    checkpoint_path = Column(String(500), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    user = relationship("User", back_populates="model_versions")
    run = relationship("TrainingJobRun", back_populates="model_version")
    test_jobs = relationship("TestJob", back_populates="model_version")


Index("idx_model_user", ModelVersion.user_id)
Index("idx_model_algorithm", ModelVersion.algorithm)


# --- 5. 测试平台 ---
class TestJob(Base):
    __tablename__ = "test_job"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    model_version_id = Column(
        BigInteger, ForeignKey("model_version.id"), nullable=False
    )
    dataset_id = Column(BigInteger, ForeignKey("dataset.id"), nullable=False)
    target = Column(TargetEnum, nullable=False)
    horizon = Column(Integer, nullable=False, default=1)
    status = Column(JobStatusEnum, nullable=False, default="PENDING")
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="test_jobs")
    model_version = relationship("ModelVersion", back_populates="test_jobs")
    dataset = relationship("Dataset", back_populates="test_jobs")
    batteries = relationship("TestJobBattery", back_populates="test_job")
    overall_metrics = relationship("TestJobMetricOverall", back_populates="test_job")
    battery_metrics = relationship("TestJobBatteryMetric", back_populates="test_job")
    predictions = relationship("TestJobPrediction", back_populates="test_job")
    logs = relationship("TestJobLog", back_populates="test_job")
    exports = relationship("TestExport", back_populates="test_job")


Index("idx_test_user", TestJob.user_id)
Index("idx_test_status", TestJob.status)
Index("idx_test_model", TestJob.model_version_id)
Index("idx_test_dataset", TestJob.dataset_id)
Index("idx_test_created", TestJob.created_at)


class TestJobBattery(Base):
    __tablename__ = "test_job_battery"

    test_job_id = Column(BigInteger, ForeignKey("test_job.id"), primary_key=True)
    battery_id = Column(BigInteger, ForeignKey("battery_unit.id"), primary_key=True)

    test_job = relationship("TestJob", back_populates="batteries")
    battery = relationship("BatteryUnit", back_populates="test_job_batteries")


Index("idx_test_battery", TestJobBattery.battery_id)


class TestJobMetricOverall(Base):
    __tablename__ = "test_job_metric_overall"
    __table_args__ = (UniqueConstraint("test_job_id", "target", name="uk_test_target"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    test_job_id = Column(BigInteger, ForeignKey("test_job.id"), nullable=False)
    target = Column(Enum("RUL", "PCL", name="metric_target_enum"), nullable=False)
    metrics = Column(JSON, nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    test_job = relationship("TestJob", back_populates="overall_metrics")


Index("idx_metric_test", TestJobMetricOverall.test_job_id)


class TestJobBatteryMetric(Base):
    __tablename__ = "test_job_battery_metric"
    __table_args__ = (
        UniqueConstraint(
            "test_job_id", "battery_id", "target", name="uk_test_battery_target"
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    test_job_id = Column(BigInteger, ForeignKey("test_job.id"), nullable=False)
    battery_id = Column(BigInteger, ForeignKey("battery_unit.id"), nullable=False)
    target = Column(
        Enum("RUL", "PCL", name="battery_metric_target_enum"), nullable=False
    )
    metrics = Column(JSON, nullable=False)

    test_job = relationship("TestJob", back_populates="battery_metrics")
    battery = relationship("BatteryUnit", back_populates="test_battery_metrics")


Index("idx_test_battery_metric", TestJobBatteryMetric.test_job_id)
Index("idx_battery_metric", TestJobBatteryMetric.battery_id)


class TestJobPrediction(Base):
    __tablename__ = "test_job_prediction"
    __table_args__ = (
        UniqueConstraint(
            "test_job_id", "battery_id", "target", "cycle_num", name="uk_pred_sample"
        ),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    test_job_id = Column(BigInteger, ForeignKey("test_job.id"), nullable=False)
    battery_id = Column(BigInteger, ForeignKey("battery_unit.id"), nullable=False)
    cycle_num = Column(Integer, nullable=False)
    target = Column(Enum("RUL", "PCL", name="pred_target_enum"), nullable=False)
    y_true = Column(Float, nullable=True)
    y_pred = Column(Float, nullable=False)

    test_job = relationship("TestJob", back_populates="predictions")
    battery = relationship("BatteryUnit", back_populates="test_predictions")


Index("idx_pred_test", TestJobPrediction.test_job_id)
Index("idx_pred_battery", TestJobPrediction.battery_id)


class TestJobLog(Base):
    __tablename__ = "test_job_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    test_job_id = Column(BigInteger, ForeignKey("test_job.id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    level = Column(LogLevelEnum, nullable=False)
    message = Column(String(2000), nullable=False)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    test_job = relationship("TestJob", back_populates="logs")


Index("idx_test_log", TestJobLog.test_job_id)
Index("idx_test_log_user", TestJobLog.user_id)
Index("idx_test_log_time", TestJobLog.test_job_id, TestJobLog.created_at)


class TestExport(Base):
    __tablename__ = "test_export"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    test_job_id = Column(BigInteger, ForeignKey("test_job.id"), nullable=False)
    user_id = Column(BigInteger, ForeignKey("user.id"), nullable=False)
    format = Column(ExportFormatEnum, nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(BigInteger, nullable=True)
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )

    test_job = relationship("TestJob", back_populates="exports")


Index("idx_export_test", TestExport.test_job_id)
Index("idx_export_user", TestExport.user_id)
Index("idx_export_created", TestExport.created_at)
