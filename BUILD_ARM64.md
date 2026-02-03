# 🏗️ 构建 ARM64 版本指南

本指南说明如何构建适用于 ARM64 架构（Apple Silicon、ARM 服务器）的 Docker 镜像。

## 📋 前置要求

1. **Docker Desktop** (Mac) 或 **Docker** (Linux ARM64)
2. **Docker Buildx** (用于跨平台构建)
3. 至少 **8GB RAM** 和 **20GB 磁盘空间**

## 🚀 快速开始

### 方法 1: 使用 Docker Compose (推荐)

**重要**: 如果 `frontend` 目录不存在，请先创建它：

```bash
# 如果 frontend 目录不存在，先创建空目录结构
mkdir -p frontend/dist

# 或者先构建前端（推荐）
cd frontend && npm ci && npm run build && cd ..
```

然后构建并启动：

```bash
# 构建并启动 ARM64 版本
docker compose -f docker-compose.arm64.yml build
docker compose -f docker-compose.arm64.yml up -d

# 查看日志
docker compose -f docker-compose.arm64.yml logs -f
```

### 方法 2: 使用 Docker Buildx (跨平台构建)

如果你在 x86_64 机器上构建 ARM64 镜像：

```bash
# 1. 创建并使用 buildx builder
docker buildx create --name arm64-builder --use
docker buildx inspect --bootstrap

# 2. 构建 ARM64 镜像
docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.arm64 \
  --tag deepseek-ocr-webui:arm64 \
  --load \
  .

# 3. 运行容器
docker run -d \
  --name deepseek-ocr-webui-arm64 \
  --platform linux/arm64 \
  -p 8001:8001 \
  --shm-size=8g \
  -e FORCE_BACKEND=cpu \
  deepseek-ocr-webui:arm64
```

### 方法 3: 在 ARM64 机器上直接构建

如果你已经在 ARM64 机器上（如 Apple Silicon Mac 或 ARM 服务器）：

```bash
# 直接构建
docker build -f Dockerfile.arm64 -t deepseek-ocr-webui:arm64 .

# 运行
docker run -d \
  --name deepseek-ocr-webui-arm64 \
  -p 8001:8001 \
  --shm-size=8g \
  -e FORCE_BACKEND=cpu \
  deepseek-ocr-webui:arm64
```

## 🔧 构建选项

### 构建时指定平台

```bash
docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.arm64 \
  --tag deepseek-ocr-webui:arm64 \
  .
```

### 推送到 Docker Hub

```bash
# 登录 Docker Hub
docker login

# 构建并推送
docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.arm64 \
  --tag your-username/deepseek-ocr-webui:arm64 \
  --push \
  .
```

### 多平台构建（同时构建 x86_64 和 ARM64）

```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --file Dockerfile.arm64 \
  --tag your-username/deepseek-ocr-webui:latest \
  --push \
  .
```

## ⚙️ 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `FORCE_BACKEND` | 强制使用后端类型 | `cpu` (ARM64 默认) |
| `HF_HOME` | HuggingFace 模型缓存目录 | `/app/models` |
| `TRANSFORMERS_CACHE` | Transformers 缓存目录 | `/app/models` |

## 📝 注意事项

1. **性能**: ARM64 版本使用 CPU 后端，性能会比 GPU 版本慢
2. **内存**: 建议至少 8GB RAM，模型加载需要较大内存
3. **首次启动**: 首次运行时会下载模型，可能需要较长时间
4. **Apple Silicon**: 如果在 Apple Silicon Mac 上运行，建议使用原生 Python 环境而不是 Docker（性能更好）

## 🐛 故障排除

### 构建失败：找不到 ARM64 基础镜像

确保使用支持 ARM64 的基础镜像。`Dockerfile.arm64` 使用 `python:3.12-slim`，它支持多架构。

### 运行时错误：模型加载失败

检查网络连接和磁盘空间。首次运行需要下载约 15GB 的模型文件。

### 性能问题

ARM64 CPU 版本性能有限。如果可能，考虑：
- 使用 Apple Silicon Mac 的原生 Python 环境（支持 MPS 加速）
- 使用 ARM64 GPU 服务器（如果有）

## 📚 相关文档

- [多平台支持指南](./README_MULTIPLATFORM.md)
- [Docker Hub 部署](./DOCKER_HUB.md)
- [快速开始](./QUICK_START.md)

