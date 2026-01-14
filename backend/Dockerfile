# 使用官方 Python 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装 uv
RUN pip install uv

# 复制依赖文件
COPY pyproject.toml .

# 安装依赖
RUN uv pip install --system --no-cache-dir -r pyproject.toml

# 复制所有后端代码到工作目录
COPY . .

# 暴露端口
EXPOSE 8000

# 使用 shell 形式的 CMD 来串联命令
# 1. 等待数据库
# 2. 导入数据
# 3. 启动应用
CMD python wait_for_db.py && \
    python data/import_builtin_dataset.py --auto && \
    uvicorn main:app --host 0.0.0.0 --port 8000
