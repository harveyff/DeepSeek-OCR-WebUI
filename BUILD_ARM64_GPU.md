# 🚀 ARM64 GPU 版本构建指南

本指南说明如何构建和使用 ARM64 GPU 版本的 DeepSeek-OCR-WebUI。

## ⚠️ 重要说明

### Apple Silicon Mac (M1/M2/M3/M4)

**Docker 限制**: Docker Desktop 在 macOS 上运行在 Linux 虚拟机中，**无法直接访问 Apple Silicon 的 MPS (Metal Performance Shaders)**。

**推荐方案**: 
- ✅ 使用**原生 Python 环境**（性能最佳，支持 MPS 加速）
- ❌ 避免使用 Docker（无法使用 GPU 加速）

### ARM64 Linux 服务器

如果您的 ARM64 Linux 服务器有 NVIDIA GPU，可以使用 CUDA 版本（需要 ARM64 CUDA 支持）。

## 🍎 Apple Silicon Mac - 原生环境（推荐）

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装 PyTorch with MPS support
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
pip install fastapi==0.119.1 uvicorn[standard]==0.38.0 python-multipart==0.0.20 python-decouple==3.8
```

### 2. 运行服务

```bash
# 使用 unified 服务（自动检测 MPS）
python web_service_unified.py

# 或者强制使用 MPS
FORCE_BACKEND=mps python web_service_unified.py
```

### 3. 验证 MPS 是否可用

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

## 🐳 Docker 方式（仅用于测试）

虽然 Docker 无法真正使用 MPS，但可以用于测试或开发：

### 构建镜像

**重要**: 如果 `frontend` 目录不存在，请先创建它：

```bash
# 如果 frontend 目录不存在，先创建空目录结构
mkdir -p frontend/dist

# 或者先构建前端（推荐）
cd frontend && npm ci && npm run build && cd ..
```

然后构建 Docker 镜像：

```bash
# 在 ARM64 机器上构建
docker build -f Dockerfile.arm64.gpu -t deepseek-ocr-webui:arm64-gpu .

# 或使用 buildx（跨平台）
docker buildx build \
  --platform linux/arm64 \
  --file Dockerfile.arm64.gpu \
  --tag deepseek-ocr-webui:arm64-gpu \
  --load \
  .
```

### 运行容器

```bash
docker run -d \
  --name deepseek-ocr-webui-arm64-gpu \
  -p 8001:8001 \
  --shm-size=8g \
  -e FORCE_BACKEND=mps \
  deepseek-ocr-webui:arm64-gpu
```

### 使用 Docker Compose

```bash
docker compose -f docker-compose.arm64.gpu.yml build
docker compose -f docker-compose.arm64.gpu.yml up -d
```

## 🔧 配置说明

### 环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `FORCE_BACKEND` | 强制使用后端类型 | `mps` (ARM64 GPU) |
| `HF_HOME` | HuggingFace 模型缓存目录 | `/app/models` |
| `TRANSFORMERS_CACHE` | Transformers 缓存目录 | `/app/models` |

### 后端选择

`web_service_unified.py` 会自动检测平台：

1. **Apple Silicon Mac** → 使用 MPS backend
2. **NVIDIA GPU** → 使用 CUDA backend  
3. **其他** → 使用 CPU backend

可以通过 `FORCE_BACKEND` 环境变量强制指定：
- `FORCE_BACKEND=mps` - 强制使用 MPS（Apple Silicon）
- `FORCE_BACKEND=cuda` - 强制使用 CUDA（NVIDIA GPU）
- `FORCE_BACKEND=cpu` - 强制使用 CPU

## 📊 性能对比

| 平台 | 后端 | 性能 | 推荐度 |
|------|------|------|--------|
| Apple Silicon Mac (原生) | MPS | ⭐⭐⭐⭐⭐ | ✅ 最佳 |
| Apple Silicon Mac (Docker) | CPU | ⭐⭐ | ❌ 不推荐 |
| ARM64 Linux + NVIDIA | CUDA | ⭐⭐⭐⭐ | ✅ 可用 |
| ARM64 Linux (无 GPU) | CPU | ⭐⭐ | ⚠️ 较慢 |

## 🐛 故障排除

### MPS 不可用

**问题**: `torch.backends.mps.is_available()` 返回 `False`

**解决方案**:
1. 确保在 macOS 12.0+ (Monterey 或更高版本)
2. 确保使用 Apple Silicon (M1/M2/M3/M4)
3. 确保安装了支持 MPS 的 PyTorch 版本
4. 如果使用 Docker，这是正常的（Docker 无法访问 MPS）

### 模型加载失败

**问题**: 模型下载或加载失败

**解决方案**:
1. 检查网络连接
2. 确保有足够的磁盘空间（约 15GB）
3. 检查 HuggingFace 访问权限

### 性能问题

**问题**: 推理速度慢

**解决方案**:
1. **Apple Silicon Mac**: 使用原生 Python 环境而不是 Docker
2. 确保 MPS 后端正常工作
3. 检查系统资源（内存、CPU）

## 📚 相关文档

- [多平台支持指南](./README_MULTIPLATFORM.md)
- [ARM64 CPU 版本](./BUILD_ARM64.md)
- [快速开始](./QUICK_START.md)

## 💡 最佳实践

### Apple Silicon Mac

```bash
# 1. 使用原生环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install torch torchvision torchaudio
pip install -r requirements.txt
pip install fastapi uvicorn[standard] python-multipart python-decouple

# 3. 运行服务
python web_service_unified.py

# 4. 验证 MPS
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### ARM64 Linux 服务器

```bash
# 如果有 NVIDIA GPU，使用 CUDA 版本
docker build -f Dockerfile -t deepseek-ocr-webui:gpu .
docker run --gpus all -p 8001:8001 deepseek-ocr-webui:gpu
```

