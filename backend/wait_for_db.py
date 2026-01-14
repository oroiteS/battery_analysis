import socket
import time
import os
import sys

def wait_for_db():
    host = os.getenv("DB_HOST", "db")
    port = int(os.getenv("DB_PORT", 3306))
    # 增加超时时间到 300 秒 (5分钟)
    timeout = 300
    start_time = time.time()

    print(f"Waiting for database at {host}:{port}...")
    
    # 尝试解析主机名，验证 DNS 是否正常
    try:
        ip_address = socket.gethostbyname(host)
        print(f"Hostname '{host}' resolved to {ip_address}")
    except socket.gaierror as e:
        print(f"DNS resolution failed for '{host}': {e}")
        # 不立即退出，继续尝试，因为 DNS 可能稍后就绪

    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print("Database is up and accepting connections!")
                return True
        except (OSError, ConnectionRefusedError):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"Timeout waiting for database at {host}:{port} after {elapsed:.0f} seconds")
                return False
            
            # 每 5 秒打印一次等待信息，避免刷屏
            if int(elapsed) % 5 == 0:
                print(f"Still waiting... ({elapsed:.0f}s elapsed)")
            
            time.sleep(1)

if __name__ == "__main__":
    # 禁用缓冲，确保日志实时输出
    sys.stdout.reconfigure(line_buffering=True)
    if not wait_for_db():
        sys.exit(1)
