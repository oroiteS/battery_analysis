import uvicorn
import os
import sys

# 将当前目录添加到 sys.path，确保能正确导入 src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import app
from src.config import settings

if __name__ == "__main__":
    # 使用 config.py 中读取的端口
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
