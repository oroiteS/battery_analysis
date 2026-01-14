#!/bin/bash
set -e

# 等待数据库就绪
python wait_for_db.py

# 运行数据导入脚本
echo "Running dataset import..."
python data/import_builtin_dataset.py --auto

# 启动应用
echo "Starting application..."
exec "$@"
