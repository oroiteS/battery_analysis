# 🔋 储能电池寿命分析及算法测试平台

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-009688.svg?style=flat&logo=fastapi)
![Vue 3](https://img.shields.io/badge/Vue-3.0+-4FC08D.svg?style=flat&logo=vue.js)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0%20CUDA%2011.8-EE4C2C.svg?style=flat&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 项目概述

本项目是一个基于 **B/S 架构** 的储能电池全生命周期数据管理与算法测试平台。旨在通过深度学习算法对电池健康状态（SoH）进行预测，并提供可视化分析与模型对比能力。

### ✨ 核心功能

- **内置数据集**: 使用 Severson Battery 数据集（`.mat`），一键导入 MySQL，按 train/val/test 划分。
- **数据分析**: 统计、趋势、相关性、分布分析，可导出 XLSX 报告。
- **训练平台**: Baseline / BiLSTM / DeepHPM 三种算法训练与指标对比，支持超参配置。
- **模型管理**: 模型版本记录、指标回溯与模型文件下载。
- **测试平台**: RUL/PCL/BOTH 预测与曲线对比，支持 CSV/XLSX 导出。
- **实时进度**: 训练与测试任务通过 WebSocket 推送日志与进度。

---

## 🛠 技术栈

### Backend (后端)
- **Core**: Python 3.10+, FastAPI
- **ML/DL**: PyTorch 2.5.0 (CUDA 11.8)
- **Database**: MySQL 8.0, SQLAlchemy
- **Auth**: JWT (python-jose / passlib)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)

### Frontend (前端)
- **Framework**: Vue 3 + Vite
- **UI**: Element Plus
- **Visualization**: ECharts
- **Network**: Axios
- **Package Manager**: pnpm

---

## 🚀 快速开始

### 1. 环境准备
- Python 3.10+
- Node 18+ / pnpm
- MySQL 8.0
- uv (推荐)

### 2. 后端启动

```bash
cd backend
cp .env.example .env
# 修改 .env: SECRET_KEY (>=32 chars) 与 DB_* 配置
uv sync

# 初始化内置数据集 (SeversonBattery.mat -> MySQL)
uv run python data/import_builtin_dataset.py

# 启动 API 服务
uv run main.py
```

### 3. 前端启动

```bash
cd frontend
pnpm install
pnpm dev
```

### 4. 访问入口
- API 文档: `http://localhost:8000/docs`
- API Base: `http://localhost:8000/api/v1`
- Web 前端: `http://localhost:5173`


---

## 📦 数据集与说明

- 数据文件: `backend/data/SeversonBattery.mat`
- 导入脚本: `backend/data/import_builtin_dataset.py`
- 数据字段: 8 个特征 + RUL / PCL 标签
- MVP 默认仅使用内置数据集（上传接口已保留但未启用）

---

## 📂 目录结构

```text
.
├── backend/                   # 后端工程
│   ├── main.py                # FastAPI 入口
│   ├── src/                   # 业务代码
│   │   ├── routes/            # API 路由
│   │   ├── tasks/             # 训练/测试 Worker
│   │   ├── config.py          # 配置与环境
│   │   └── models.py          # 数据库模型
│   ├── data/                  # 数据集与导入脚本
│   └── pyproject.toml         # 后端依赖
│
├── frontend/                  # 前端工程
│   ├── src/                   # Vue 源代码
│   │   ├── views/             # 页面
│   │   └── components/        # 图表/组件
│   ├── package.json           # 前端依赖
│   └── vite.config.ts         # Vite 配置
│
├── doc/                       # 项目文档与 OpenAPI
│   ├── openapi.yaml
│   └── project.md
│
└── LICENSE
```
