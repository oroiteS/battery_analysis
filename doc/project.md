# 储能电池寿命分析及算法测试平台 - 项目规划

## 1. 项目概述

基于 B/S 架构的储能电池全生命周期数据管理与算法测试平台。集成数据可视化、深度学习算法训练（Baseline, BiLSTM, DeepHPM）、电池寿命（RUL/PCL）预测功能。

## 2. 技术栈

- **后端**: Python >=3.13, FastAPI, PyTorch (CUDA 12.9), MySQL (Docker), SQLAlchemy
- **前端**: Vue 3, Vite, TypeScript, Pinia, Vue Router
- **包管理**: uv (Python), pnpm (Frontend)

## 3. 项目目录结构

```
backend/
├── main.py                    # FastAPI 入口（当前结构）
├── power_soh/                 # 算法核心模块
│   ├── functions.py           # 数据处理与模型定义
│   ├── SoH_CaseA_*.py         # 训练脚本
│   ├── Settings/              # 超参数配置
│   ├── results/               # 训练结果
│   └── SeversonBattery.mat    # 数据集
├── pyproject.toml             # 依赖配置
└── .venv/                     # 虚拟环境

frontend/
├── src/
│   ├── __tests__/             # 单元测试
│   ├── router/                # 路由配置
│   ├── stores/                # Pinia 状态管理
│   ├── App.vue
│   └── main.ts
├── public/
├── index.html
├── package.json
├── tsconfig*.json
├── vite.config.ts
└── vitest.config.ts
```

**规划中的重构结构**:
```
backend/app/
├── algorithms/                # 算法模块（待迁移）
├── routers/                   # API 路由
├── core/                      # 核心配置
├── models/                    # ORM 模型
└── main.py
```

## 4. 环境配置

### 4.1 后端环境

```bash
cd backend
uv sync                        # 安装依赖
source .venv/bin/activate      # 激活环境（Linux/macOS）
# .venv\Scripts\activate       # Windows
```

### 4.2 数据库环境

```bash
# 启动 MySQL 容器
docker run -d \
  --name battery-mysql \
  -p 13306:3306 \
  -e MYSQL_ROOT_PASSWORD=root \
  mysql:8.0

# 创建数据库
docker exec -it battery-mysql mysql -uroot -proot -e \
  "CREATE DATABASE battery_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

连接信息：
- Host: localhost
- Port: 13306
- User: root
- Password: root
- Database: battery_db

### 4.3 前端环境

```bash
cd frontend
pnpm install
pnpm dev
```

## 5. 数据库设计

### battery_metadata (电池元数据)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT (PK) | 电池 ID |
| group_id | INT | 组别 (Train/Test/Val) |
| total_cycles | INT | 总循环次数 |
| nominal_capacity | FLOAT | 标称容量 |

### cycle_data (循环数据)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT (PK) | 自增主键 |
| battery_id | INT (FK) | 关联电池ID |
| cycle_index | INT | 循环圈数 |
| voltage | FLOAT | 电压 |
| current | FLOAT | 电流 |
| temperature | FLOAT | 温度 |
| pcl | FLOAT | 容量衰减百分比 |
| rul | FLOAT | 剩余寿命 |

**注**: 数据量大，建议对 battery_id 和 cycle_index 建立索引

### training_logs (训练记录)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | INT (PK) | 记录ID |
| model_name | VARCHAR | 算法名称 |
| hyperparams | JSON | 超参数 |
| metrics | JSON | 评估指标 |
| model_path | VARCHAR | 模型文件路径 |
| created_at | DATETIME | 训练时间 |

## 6. 开发实施步骤

### 阶段一：数据工程

1. 配置数据库连接（SQLAlchemy）
2. 编写数据迁移脚本，读取 .mat 文件
3. 批量插入数据到 MySQL

### 阶段二：后端 API

1. 封装算法训练函数
2. 实现 RESTful API：
   - `GET /api/batteries` - 电池列表
   - `GET /api/batteries/{id}/cycles` - 循环数据
   - `POST /api/train` - 启动训练
   - `POST /api/predict` - 预测接口

### 阶段三：前端开发

1. 搭建布局（侧边栏导航）
2. 集成 ECharts 图表（折线图、热力图）
3. 实现训练交互（参数提交、进度展示）

## 7. 运行命令

```bash
# 后端（当前结构）
cd backend
uv run uvicorn main:app --reload

# 算法训练
cd backend/power_soh
python SoH_CaseA_Baseline.py
python SOH_CaseA_BiLSTM.py
python SoH_CaseA_DeepHPM_Sum.py

# 前端
cd frontend
pnpm dev
```
