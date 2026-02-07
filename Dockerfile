# DeepSeek-OCR-WebUI Dockerfile
# 基于 NVIDIA PyTorch 镜像，支持 GPU 加速
# 使用最新镜像，采用 transformers 引擎（更稳定）

# Stage 1: Build frontend
FROM node:20-slim AS frontend-builder
WORKDIR /build
# Copy frontend directory - if it doesn't exist, create empty structure
RUN mkdir -p frontend/dist
COPY frontend ./frontend/
WORKDIR /build/frontend
RUN if [ -f package.json ]; then \
        if [ -f package-lock.json ]; then \
            npm ci; \
        else \
            npm install; \
        fi && \
        npm run build && \
        rm -rf node_modules .npm; \
    else \
        echo "No package.json found, keeping empty dist directory"; \
    fi

# Stage 2: Main application
FROM nvcr.io/nvidia/pytorch:25.09-py3

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    VLLM_USE_V1=0 \
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    CUDA_HOME=/usr/local/cuda

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
COPY DeepSeek-OCR-master ./DeepSeek-OCR-master

# 安装 Python 依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 安装 FastAPI 和其他 Web 依赖（替代 vLLM）
RUN pip install \
    fastapi==0.119.1 \
    uvicorn[standard]==0.38.0 \
    python-multipart==0.0.20 \
    python-decouple==3.8

# 复制应用代码
COPY web_service_unified.py .
COPY web_service.py .
COPY web_service_gpu.py .
COPY gpu_manager.py .
COPY ocr_ui_modern.html .
COPY backends ./backends
COPY i18n.js .

# 复制前端构建产物（从构建阶段）
RUN mkdir -p ./frontend/dist
COPY --from=frontend-builder /build/frontend/dist ./frontend/dist

# 暴露端口
EXPOSE 8001

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 启动服务 (使用 unified 版本支持 Vue 3 前端)
CMD ["python", "web_service_unified.py"]
