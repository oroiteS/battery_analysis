from sqlalchemy import create_engine, Column, Integer, BigInteger, String, Float, ForeignKey, DateTime, JSON, Enum, Numeric, Text, UniqueConstraint, Index, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from datetime import datetime, timezone
from src.config import settings

# --- 数据库连接核心 ---
# 创建引擎
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# 创建 Session 工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 基类
Base = declarative_base()


# --- 数据库依赖 ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- ORM 模型定义 ---

class User(Base):
    __tablename__ = 'user'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_name = Column(String(50), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    path = Column(String(500), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    training_jobs = relationship('TrainingJob', back_populates='user')
    models = relationship('Model', back_populates='user')
    predictions = relationship('Prediction', back_populates='user')
    ensembles = relationship('ModelEnsemble', back_populates='user')
    ensemble_predictions = relationship('EnsemblePrediction', back_populates='user')


class BatteryUnit(Base):
    __tablename__ = 'battery_unit'

    id = Column(Integer, primary_key=True)
    group_tag = Column(Enum('train', 'val', 'test', name='group_tag_enum'), nullable=True)
    total_cycles = Column(Integer, nullable=False)
    nominal_capacity = Column(Float, nullable=True)

    cycle_data = relationship('CycleData', back_populates='battery')
    predictions = relationship('Prediction', back_populates='battery')
    ensemble_predictions = relationship('EnsemblePrediction', back_populates='battery')


class CycleData(Base):
    __tablename__ = 'cycle_data'
    __table_args__ = (
        UniqueConstraint('battery_id', 'cycle_num', name='uk_battery_cycle'),
        Index('idx_cycle_battery', 'battery_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    battery_id = Column(Integer, ForeignKey('battery_unit.id'), nullable=False)
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

    battery = relationship('BatteryUnit', back_populates='cycle_data')


class TrainingJob(Base):
    __tablename__ = 'training_job'
    __table_args__ = (
        Index('idx_job_status', 'status'),
        Index('idx_job_user', 'user_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('user.id'), nullable=False)
    algorithm = Column(Enum('baseline', 'bilstm', 'deephpm', name='algorithm_enum'), nullable=False)
    status = Column(Enum('queued', 'running', 'succeeded', 'failed', 'stopped', name='status_enum'), nullable=False)
    model_name = Column(String(120), nullable=True)
    hyperparams = Column(JSON, nullable=True)
    progress = Column(Numeric(6, 5), nullable=False, default=0.0)
    current_epoch = Column(Integer, nullable=False, default=0)
    total_epochs = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    user = relationship('User', back_populates='training_jobs')
    batteries = relationship('TrainingJobBattery', back_populates='job')
    metrics = relationship('TrainingJobMetric', back_populates='job')
    logs = relationship('TrainingJobLog', back_populates='job')
    models = relationship('Model', back_populates='job')


class TrainingJobBattery(Base):
    __tablename__ = 'training_job_battery'
    __table_args__ = (
        Index('idx_job_battery', 'battery_id'),
    )

    job_id = Column(BigInteger, ForeignKey('training_job.id'), primary_key=True)
    battery_id = Column(Integer, ForeignKey('battery_unit.id'), primary_key=True)
    split_role = Column(Enum('train', 'val', 'test', name='split_role_enum'), nullable=False, default='train', primary_key=True)

    job = relationship('TrainingJob', back_populates='batteries')


class TrainingJobMetric(Base):
    __tablename__ = 'training_job_metric'
    __table_args__ = (
        UniqueConstraint('job_id', 'epoch', name='uk_job_epoch'),
        Index('idx_metric_job', 'job_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(BigInteger, ForeignKey('training_job.id'), nullable=False)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)

    job = relationship('TrainingJob', back_populates='metrics')


class TrainingJobLog(Base):
    __tablename__ = 'training_job_log'
    __table_args__ = (
        Index('idx_log_job', 'job_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(BigInteger, ForeignKey('training_job.id'), nullable=False)
    level = Column(Enum('DEBUG', 'INFO', 'WARNING', 'ERROR', name='log_level_enum'), nullable=False)
    message = Column(String(2000), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    job = relationship('TrainingJob', back_populates='logs')


class Model(Base):
    __tablename__ = 'model'
    __table_args__ = (
        UniqueConstraint('algorithm', 'name', 'version', name='uk_model_version'),
        Index('idx_model_job', 'job_id'),
        Index('idx_model_user', 'user_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('user.id'), nullable=False)
    algorithm = Column(Enum('baseline', 'bilstm', 'deephpm', name='algorithm_enum'), nullable=False)
    name = Column(String(120), nullable=False)
    version = Column(String(32), nullable=False)
    job_id = Column(BigInteger, ForeignKey('training_job.id'), nullable=True)
    metrics = Column(JSON, nullable=True)
    file_path = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship('User', back_populates='models')
    job = relationship('TrainingJob', back_populates='models')
    predictions = relationship('Prediction', back_populates='model')
    ensemble_members = relationship('ModelEnsembleMember', back_populates='model')


class Prediction(Base):
    __tablename__ = 'prediction'
    __table_args__ = (
        Index('idx_pred_model', 'model_id'),
        Index('idx_pred_battery', 'battery_id'),
        Index('idx_pred_user', 'user_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('user.id'), nullable=False)
    model_id = Column(BigInteger, ForeignKey('model.id'), nullable=False)
    battery_id = Column(Integer, ForeignKey('battery_unit.id'), nullable=True)
    input_source = Column(Enum('battery', 'payload', name='input_source_enum'), nullable=False)
    input_payload = Column(JSON, nullable=True)
    soh = Column(Float, nullable=True)
    rul = Column(Float, nullable=True)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship('User', back_populates='predictions')
    model = relationship('Model', back_populates='predictions')
    battery = relationship('BatteryUnit', back_populates='predictions')


class ModelEnsemble(Base):
    __tablename__ = 'model_ensemble'
    __table_args__ = (
        Index('idx_ensemble_user', 'user_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('user.id'), nullable=False)
    name = Column(String(120), nullable=False)
    strategy = Column(Enum('compare', 'avg', 'weighted_avg', 'stacking', name='strategy_enum'), nullable=False)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship('User', back_populates='ensembles')
    members = relationship('ModelEnsembleMember', back_populates='ensemble')
    predictions = relationship('EnsemblePrediction', back_populates='ensemble')


class ModelEnsembleMember(Base):
    __tablename__ = 'model_ensemble_member'

    ensemble_id = Column(BigInteger, ForeignKey('model_ensemble.id'), primary_key=True)
    model_id = Column(BigInteger, ForeignKey('model.id'), primary_key=True)
    weight = Column(Numeric(8, 6), nullable=True)

    ensemble = relationship('ModelEnsemble', back_populates='members')
    model = relationship('Model', back_populates='ensemble_members')


class EnsemblePrediction(Base):
    __tablename__ = 'ensemble_prediction'
    __table_args__ = (
        Index('idx_ensemble_pred', 'ensemble_id'),
        Index('idx_ensemble_pred_user', 'user_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey('user.id'), nullable=False)
    ensemble_id = Column(BigInteger, ForeignKey('model_ensemble.id'), nullable=False)
    battery_id = Column(Integer, ForeignKey('battery_unit.id'), nullable=True)
    input_payload = Column(JSON, nullable=True)
    soh = Column(Float, nullable=True)
    rul = Column(Float, nullable=True)
    result = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))

    user = relationship('User', back_populates='ensemble_predictions')
    ensemble = relationship('ModelEnsemble', back_populates='predictions')
    battery = relationship('BatteryUnit', back_populates='ensemble_predictions')
    details = relationship('EnsemblePredictionDetail', back_populates='ensemble_prediction')


class EnsemblePredictionDetail(Base):
    __tablename__ = 'ensemble_prediction_detail'
    __table_args__ = (
        Index('idx_ensemble_pred_detail', 'ensemble_prediction_id'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ensemble_prediction_id = Column(BigInteger, ForeignKey('ensemble_prediction.id'), nullable=False)
    model_id = Column(BigInteger, ForeignKey('model.id'), nullable=False)
    soh = Column(Float, nullable=True)
    rul = Column(Float, nullable=True)
    result = Column(JSON, nullable=True)

    ensemble_prediction = relationship('EnsemblePrediction', back_populates='details')
