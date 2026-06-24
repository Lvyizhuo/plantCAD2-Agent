#!/bin/bash
set -e

echo "=== PlantCAD2 Inference Service ==="
echo ""

GPU_DEVICE=${1:-3}
PORT=${2:-8005}

echo "GPU Device: $GPU_DEVICE"
echo "Port: $PORT"
echo ""

docker run -d \
    --gpus "\"device=$GPU_DEVICE\"" \
    -p $PORT:8005 \
    -e CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
    -e PLANTCAD2_DEVICE=cuda:0 \
    -e PLANTCAD2_PRELOAD_LORA=true \
    -v plantcad2-logs:/workspace/logs \
    --name plantcad2 \
    --restart unless-stopped \
    plantcad2-inference:latest

echo ""
echo "=== Service Started ==="
echo "Container: plantcad2"
echo "Port: $PORT"
echo ""
echo "Useful commands:"
echo "  docker logs -f plantcad2        # View logs"
echo "  docker exec -it plantcad2 bash  # Enter container"
echo "  docker stop plantcad2           # Stop service"
echo "  docker restart plantcad2        # Restart service"
echo ""
echo "Health check:"
echo "  curl http://localhost:$PORT/health"
