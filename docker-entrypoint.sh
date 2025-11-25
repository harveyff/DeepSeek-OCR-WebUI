#!/bin/sh
set -e

# 默认后端 URL（如果未设置环境变量）
VLLM_SERVER_URL=${VLLM_SERVER_URL:-}

# 替换 HTML 中的环境变量占位符
if [ -n "$VLLM_SERVER_URL" ]; then
    echo "🔧 配置后端 URL: $VLLM_SERVER_URL"
    # 使用 awk 进行替换，更可靠
    awk -v url="$VLLM_SERVER_URL" '{gsub(/\$\{VLLM_SERVER_URL:-\}/, url); print}' /usr/share/nginx/html/index.html > /tmp/index.html
    mv /tmp/index.html /usr/share/nginx/html/index.html
else
    echo "⚠️  未设置 VLLM_SERVER_URL，将使用默认值（当前域名）"
    # 替换为空字符串，让前端使用默认值
    sed -i "s|\${VLLM_SERVER_URL:-}||g" /usr/share/nginx/html/index.html
fi

# 执行 nginx 启动命令
exec "$@"

