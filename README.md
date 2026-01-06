# 🔋 储能电池寿命分析及算法测试平台

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg?style=flat&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688.svg?style=flat&logo=fastapi)
![Vue 3](https://img.shields.io/badge/Vue-3.0+-4FC08D.svg?style=flat&logo=vue.js)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.9-EE4C2C.svg?style=flat&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 项目概述

本项目是一个基于 **B/S 架构** 的储能电池全生命周期数据管理与算法测试平台。旨在通过深度学习算法对电池健康状态（SoH）进行精准预测，并提供直观的数据可视化分析工具。

### ✨ 核心功能

- **数据管理**: 支持 `.mat` 格式电池数据集的解析与 MySQL 持久化存储。
- **算法集成**: 内置多种深度学习模型，支持从 Baseline 到高阶算法的对比测试：
  - Baseline
  - BiLSTM (双向长短期记忆网络)
  - DeepHPM (深度高性能模型)
- **寿命预测**: 提供剩余使用寿命 (RUL) 和 容量衰减百分比 (PCL) 的实时预测。
- **可视化看板**: 基于 ECharts 的多维数据展示（电压、电流、温度、SOH 衰减曲线）。

---

## 🛠 技术栈

### Backend (后端)
- **Core**: Python 3.13+, FastAPI
- **ML/DL**: PyTorch (适配 CUDA 12.9)
- **Database**: MySQL 8.0 (Docker), SQLAlchemy (ORM)
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (极致性能的 Python 包管理工具)

### Frontend (前端)
- **Framework**: Vue 3, Element Plus
- **Visualization**: ECharts
- **Network**: Axios

---

## 📂 目录结构

```text
.
├── backend/                   # 后端工程
│   ├── main.py                # FastAPI 入口
│   ├── power_soh/             # 核心算法模块 (Dataset, Models, Training)
│   ├── pyproject.toml         # uv 依赖配置
│   └── .venv/                 # 虚拟环境
│
├── frontend/                  # 前端工程
│   ├── src/                   # Vue 源代码
│   └── vite.config.js         # Vite 配置
│
└── docker-compose.yml         # (可选) 容器编排
