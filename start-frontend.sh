#!/bin/bash
# DeepSeek-OCR WebUI 前端快速启动脚本

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 DeepSeek-OCR WebUI 前端启动脚本${NC}"
echo ""

# 检查 VLLM_SERVER_URL 环境变量
if [ -z "$VLLM_SERVER_URL" ]; then
    echo -e "${YELLOW}⚠️  未设置 VLLM_SERVER_URL 环境变量${NC}"
    echo "请输入你的 vLLM 后端 URL（例如: http://localhost:8000）:"
    read -r VLLM_SERVER_URL
    
    if [ -z "$VLLM_SERVER_URL" ]; then
        echo -e "${RED}❌ 错误: 必须提供后端 URL${NC}"
        exit 1
    fi
    
    export VLLM_SERVER_URL
fi

echo -e "${GREEN}✅ 后端 URL: $VLLM_SERVER_URL${NC}"
echo ""

# 检查 docker-compose 是否可用
if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 docker 或 docker-compose${NC}"
    exit 1
fi

# 使用 docker-compose 或 docker compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo -e "${GREEN}📦 构建并启动容器...${NC}"
$COMPOSE_CMD -f docker-compose.frontend.yml up -d --build

echo ""
echo -e "${GREEN}✅ 启动完成！${NC}"
echo ""
echo -e "访问地址: ${GREEN}http://localhost:8001${NC}"
echo ""
echo "查看日志: $COMPOSE_CMD -f docker-compose.frontend.yml logs -f"
echo "停止服务: $COMPOSE_CMD -f docker-compose.frontend.yml down"

