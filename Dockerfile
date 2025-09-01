FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/cache

COPY debian.sources /etc/apt/sources.list.d/

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装Python依赖
COPY requirements.txt .
ENV HTTP_PROXY http://10.0.0.112:7890
ENV HTTPS_PROXY http://10.0.0.112:7890
RUN pip install --no-cache-dir -r requirements.txt
COPY pip.conf /etc/pip.conf

# 复制应用代码
COPY embedding_service.py .
COPY start.sh .

# 创建缓存目录
RUN mkdir -p /app/cache

# 给启动脚本执行权限
RUN chmod +x start.sh

# 暴露端口
EXPOSE 8888

# 启动命令
CMD ["./start.sh"]
