#!/bin/bash
set -e

echo "=== PlantCAD2 Docker Build Script ==="
echo ""

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

if [ ! -d "models/PlantCAD2-Large-l48-d1536" ]; then
    echo "Error: Base model not found at models/PlantCAD2-Large-l48-d1536"
    exit 1
fi

if [ ! -d "app" ]; then
    echo "Error: app/ directory not found"
    exit 1
fi

echo "Building Docker image..."
docker build \
    -f docker/deploy/Dockerfile \
    -t plantcad2-inference:latest \
    -t plantcad2-inference:$(date +%Y%m%d) \
    .

echo ""
echo "=== Build Complete ==="
echo "Image: plantcad2-inference:latest"
echo ""
echo "To run:"
echo "  docker run -d --gpus '\"device=3\"' -p 8005:8005 --name plantcad2 plantcad2-inference:latest"
echo ""
echo "To check:"
echo "  docker logs plantcad2"
echo "  curl http://localhost:8005/health"
