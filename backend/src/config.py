# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 路径定位
current_path = Path(__file__).resolve()
src_dir = current_path.parent
base_dir = src_dir.parent
env_path = base_dir / ".env"

# 2. 加载 .env
load_dotenv(dotenv_path=env_path)


class Settings:
    # 基础路径
    BASE_DIR: Path = base_dir

    # 应用配置
    ENV: str = os.getenv("APP_ENV", "production")
    PORT: int = int(os.getenv("PORT", 8000))
    SECRET_KEY: str = os.getenv("SECRET_KEY", "default_secret_key")

    # 数据库配置
    DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
    DB_PORT: int = int(os.getenv("DB_PORT", 3306))
    DB_USER: str = os.getenv("DB_USER", "root")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "root")
    DB_NAME: str = os.getenv("DB_NAME", "battery_db")

    # 路径配置 (自动创建目录)
    UPLOAD_PATH: Path = BASE_DIR / os.getenv("UPLOAD_PATH", "storage/uploads")
    MODEL_STORAGE_PATH: Path = BASE_DIR / os.getenv(
        "MODEL_STORAGE_PATH", "storage/models"
    )

    def __init__(self):
        # 初始化时确保目录存在
        self.UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
        self.MODEL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

        # 验证 SECRET_KEY 安全性
        if not self.SECRET_KEY or len(self.SECRET_KEY) < 32:
            # 在开发环境中，如果未设置或太短，可以给一个警告而不是报错，或者使用默认值
            # 但为了安全起见，生产环境应该强制要求
            if self.ENV == "production":
                 raise ValueError(
                    "SECRET_KEY must be at least 32 characters long. "
                    "Set a strong SECRET_KEY in your .env file."
                )
            else:
                print("Warning: SECRET_KEY is weak or not set properly.")

    @property
    def DATABASE_URL(self) -> str:
        """构造 SQLAlchemy 连接字符串"""
        # 确保端口是整数
        try:
            port = int(self.DB_PORT)
        except ValueError:
            port = 3306
            
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{port}/{self.DB_NAME}"


settings = Settings()
