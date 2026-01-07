from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.routes import api_router
from src.models import Base, engine

# 1. 定义 lifespan (生命周期) 管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动逻辑 (Startup) ---
    # 仅作为示例：在开发环境下自动创建表结构
    # 生产环境通常推荐使用 Alembic 进行迁移
    Base.metadata.create_all(bind=engine)
    print(f"Server started on port {settings.PORT} in {settings.ENV} mode.")
    print(f"Database connected: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    
    yield  # 应用运行期间，在此处挂起
    
    # --- 关闭逻辑 (Shutdown) ---
    print("Server shutting down...")
    # 如果有数据库连接池清理工作，可以在这里执行

# 2. 实例化 FastAPI，并传入 lifespan 参数
app = FastAPI(
    title="储能电池寿命分析及算法测试平台 API",
    version="1.0.0",
    description="集成数据可视化、深度学习算法训练及电池寿命预测功能。",
    lifespan=lifespan  # <--- 在这里注册
)

# 3. 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. 注册所有路由
app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    """基础健康检查"""
    return {
        "status": "ok",
        "env": settings.ENV,
        "database": "connected"
    }
