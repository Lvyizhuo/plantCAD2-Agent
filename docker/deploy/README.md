# PlantCAD2 Docker 部署指南

## 目录结构

```
docker/deploy/
├── Dockerfile
├── .dockerignore
├── build.sh
├── docker-compose.yml
└── README.md
```

## 打包步骤

### 1. 在服务器上打包镜像

```bash
cd ~/ntt/lvyizhuo/PlantCaduceus

# 确保模型文件存在
ls -la models/PlantCAD2-Large-l48-d1536/
ls -la models/*plantcad2_large/

# 执行打包
chmod +x docker/deploy/build.sh
bash docker/deploy/build.sh
```

### 2. 验证镜像

```bash
docker images | grep plantcad2
```

## 启动服务

### 方式一：使用 docker-compose（推荐）

```bash
cd ~/ntt/lvyizhuo/PlantCaduceus/docker/deploy

# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止
docker-compose down
```

### 方式二：使用 docker run

```bash
docker run -d \
    --gpus '"device=3"' \
    -p 8005:8005 \
    -e CUDA_VISIBLE_DEVICES=3 \
    -e PLANTCAD2_DEVICE=cuda:0 \
    -e PLANTCAD2_PRELOAD_LORA=true \
    --name plantcad2 \
    plantcad2-inference:latest
```

### 方式三：使用 docker run（简化版）

```bash
docker run -d --gpus all -p 8005:8005 --name plantcad2 plantcad2-inference:latest
```

## 验证服务

```bash
# 健康检查
curl http://localhost:8005/health

# 查看日志
docker logs -f plantcad2

# 测试推理
curl -X POST http://localhost:8005/embeddings \
    -H "Content-Type: application/json" \
    -d '{"sequence": "ATCGATCGATCG"}'
```

## 迁移到生产环境

### 1. 保存镜像

```bash
# 在开发服务器上保存
docker save plantcad2-inference:latest | gzip > plantcad2-inference-$(date +%Y%m%d).tar.gz

# 或者只保存镜像（不压缩）
docker save plantcad2-inference:latest > plantcad2-inference-$(date +%Y%m%d).tar
```

### 2. 传输到生产环境

```bash
# 使用 scp
scp plantcad2-inference-*.tar.gz user@production-server:/path/to/

# 或者使用 rsync
rsync -avz plantcad2-inference-*.tar.gz user@production-server:/path/to/
```

### 3. 在生产环境加载镜像

```bash
# 加载镜像
docker load < plantcad2-inference-*.tar.gz

# 验证
docker images | grep plantcad2
```

### 4. 启动服务

```bash
# 使用 docker-compose
docker-compose up -d

# 或者使用 docker run
docker run -d \
    --gpus '"device=0"' \
    -p 8005:8005 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PLANTCAD2_DEVICE=cuda:0 \
    --name plantcad2 \
    plantcad2-inference:latest
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CUDA_VISIBLE_DEVICES` | GPU 设备号 | `3` |
| `PLANTCAD2_DEVICE` | 推理设备 | `cuda:0` |
| `PLANTCAD2_MODEL_PATH` | 基础模型路径 | `models/PlantCAD2-Large-l48-d1536` |
| `PLANTCAD2_LORA_PATH` | LoRA 模型路径 | `models` |
| `PLANTCAD2_PRELOAD_LORA` | 是否预加载 LoRA | `true` |

## 常见问题

### 1. GPU 不可用

检查 NVIDIA 驱动和 Docker GPU 支持：

```bash
# 检查驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2. 模型加载失败

检查模型文件是否正确打包：

```bash
docker exec plantcad2 ls -la /workspace/models/
```

### 3. 端口冲突

修改 `docker-compose.yml` 中的端口映射：

```yaml
ports:
  - "8006:8005"  # 改为 8006
```

### 4. 内存不足

增加 Docker 内存限制：

```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

## 镜像大小

预计镜像大小：
- 基础镜像：~8GB
- 模型文件：~6.7GB
- 依赖包：~2GB
- **总计：~17GB**

## 性能优化

### 1. 使用多阶段构建（可选）

如果需要减小镜像大小，可以使用多阶段构建。

### 2. 使用模型挂载（可选）

如果模型文件很大，可以挂载而不是打包：

```yaml
volumes:
  - /path/to/models:/workspace/models:ro
```

### 3. 使用 GPU 调度

在生产环境使用 GPU 调度器（如 Kubernetes）管理资源。
