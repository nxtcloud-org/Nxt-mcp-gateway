#!/bin/bash

# NxtCloud MCP Gateway Docker 빌드 스크립트

echo "🐳 NxtCloud MCP Gateway Docker 이미지 빌드 중..."

# 이미지 태그 설정
IMAGE_NAME="nxtcloud/mcp-gateway"
VERSION="latest"

# Docker 이미지 빌드
docker build -t ${IMAGE_NAME}:${VERSION} .

if [ $? -eq 0 ]; then
    echo "✅ Docker 이미지 빌드 완료: ${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "🚀 실행 방법:"
    echo "docker run -p 8501:8501 ${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "또는 docker-compose 사용:"
    echo "docker-compose up -d"
else
    echo "❌ Docker 이미지 빌드 실패"
    exit 1
fi