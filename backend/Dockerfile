# 使用官方 Python 镜像
FROM python:3.11-slim

# --- 新增：接收构建参数与环境变量设置 ---
# 1. 接收 docker-compose 传来的参数 (默认值为清华源)
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    UV_INDEX_URL=$UV_INDEX_URL \
    PIP_INDEX_URL=$PIP_INDEX_URL

WORKDIR /app

RUN pip install uv

COPY pyproject.toml uv.lock ./

ENV UV_PROJECT_ENVIRONMENT=/usr/local

RUN uv sync --frozen

COPY . .

EXPOSE 8000

CMD python wait_for_db.py && \
    python data/import_builtin_dataset.py --auto && \
    python main.py
