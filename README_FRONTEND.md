# DeepSeek-OCR WebUI 前端版本

这是 DeepSeek-OCR WebUI 的前端版本，仅包含 Web 界面，不包含后端服务。适用于已经通过 vLLM 运行 DeepSeek-OCR 后端的场景。

## 功能特点

- ✅ 纯前端实现，轻量级
- ✅ 使用 nginx 服务静态文件
- ✅ 支持通过环境变量配置后端 URL
- ✅ 支持所有原有功能（文档识别、OCR、Find 模式、Freeform 等）

## 快速开始

### 使用 Docker Compose（推荐）

1. 设置环境变量并启动：

```bash
# 设置你的 vLLM 后端 URL
export VLLM_SERVER_URL=http://your-vllm-server:8000

# 使用前端版本的 docker-compose
docker-compose -f docker-compose.frontend.yml up -d
```

2. 访问 Web UI：

打开浏览器访问 `http://localhost:8001`

### 使用 Docker 直接运行

```bash
# 构建镜像
docker build -f Dockerfile.frontend -t deepseek-ocr-webui-frontend .

# 运行容器（设置后端 URL）
docker run -d \
  -p 8001:80 \
  -e VLLM_SERVER_URL=http://your-vllm-server:8000 \
  --name deepseek-ocr-webui-frontend \
  deepseek-ocr-webui-frontend
```

### 环境变量

- `VLLM_SERVER_URL`: vLLM 后端服务的 URL（必需）
  - 示例: `http://localhost:8000`
  - 示例: `http://192.168.1.100:8000`
  - 如果未设置，前端将尝试使用当前域名作为后端地址

## 后端 API 要求

前端需要后端提供以下 API 端点：

- `POST /ocr` - OCR 识别接口
- `POST /pdf-to-images` - PDF 转图片接口

### OCR 接口格式

```json
POST /ocr
Content-Type: multipart/form-data

{
  "file": <图片文件>,
  "prompt_type": "document|ocr|free|figure|describe|find|freeform",
  "find_term": "<查找关键词>",  // find 模式需要
  "custom_prompt": "<自定义提示词>",  // freeform 模式需要
  "grounding": true|false
}
```

响应格式：
```json
{
  "success": true,
  "text": "<识别结果文本>",
  "raw_text": "<原始文本>",
  "boxes": [
    {
      "label": "<标签>",
      "box": [x1, y1, x2, y2]
    }
  ],
  "image_dims": {"w": 1920, "h": 1080},
  "prompt_type": "document",
  "metadata": {...}
}
```

### PDF 转图片接口格式

```json
POST /pdf-to-images
Content-Type: multipart/form-data

{
  "file": <PDF文件>
}
```

响应格式：
```json
{
  "success": true,
  "images": [
    {
      "data": "data:image/png;base64,...",
      "name": "page_1.png",
      "width": 1920,
      "height": 1080,
      "page_number": 1
    }
  ],
  "page_count": 5,
  "total_pages": 5,
  "original_filename": "document.pdf"
}
```

## 文件说明

- `Dockerfile.frontend` - 前端 Dockerfile（使用 nginx）
- `docker-compose.frontend.yml` - 前端版本的 docker-compose 配置
- `nginx.conf` - nginx 配置文件
- `docker-entrypoint.sh` - 启动脚本（处理环境变量注入）
- `ocr_ui_modern.html` - 前端 HTML 文件（已修改支持环境变量配置）

## 与原版的区别

| 特性 | 原版 | 前端版本 |
|------|------|----------|
| 后端服务 | 包含 FastAPI 后端 | 仅前端 |
| 模型加载 | 自动加载模型 | 需要外部 vLLM 服务 |
| Docker 镜像 | 较大（包含 PyTorch） | 较小（仅 nginx） |
| 启动时间 | 较长（需要加载模型） | 快速 |
| 资源占用 | 高（GPU/内存） | 低（仅 Web 服务） |

## 故障排除

### 前端无法连接到后端

1. 检查 `VLLM_SERVER_URL` 环境变量是否正确设置
2. 检查后端服务是否正常运行
3. 检查网络连接和防火墙设置
4. 查看浏览器控制台的错误信息

### CORS 错误

如果遇到 CORS 错误，需要在后端服务中配置 CORS 头：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 许可证

与原项目相同。

