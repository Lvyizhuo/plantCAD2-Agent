# PlantCAD2 推理服务接口文档

## 服务概述

PlantCAD2 是一个基于 Mamba2 架构的 DNA 语言模型（694M 参数），支持 8192 bp 上下文。本服务提供 REST API 接口，支持以下功能：

- **3 个基础功能**：嵌入提取、变异打分、掩码预测（使用基座模型）
- **7 个 LoRA 任务**：ACR 预测、表达量预测、翻译效率预测（使用微调适配器）

## 快速开始

### 启动服务

```bash
CUDA_VISIBLE_DEVICES=3 PLANTCAD2_DEVICE=cuda:0 uvicorn app.main:app --host 0.0.0.0 --port 8005
```

### 健康检查

```bash
curl http://localhost:8005/health
```

响应：
```json
{"status": "ok", "model_loaded": true, "device": "cuda:0"}
```

---

## 接口列表

### 1. 嵌入提取

**POST** `/embeddings`

提取 DNA 序列每个位置的 1536 维向量表示。

```bash
curl -X POST http://localhost:8005/embeddings \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTACGTACGT...", "normalize": true}'
```

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列（IUPAC 碱基，最长 8192 bp） |
| normalize | bool | ❌ | 是否 L2 归一化，默认 true |

**响应**：

```json
{
  "embeddings": [[0.012, -0.034, ...], ...],
  "shape": [512, 1536],
  "sequence_length": 512
}
```

---

### 2. 变异打分

**POST** `/variant-score`

评估单核苷酸变异的致病性。

```bash
curl -X POST http://localhost:8005/variant-score \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "ACGTACGTACGT...",
    "position": 100,
    "ref_allele": "A",
    "alt_alleles": ["G", "C", "T"]
  }'
```

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | 包含变异位点的上下文序列 |
| position | int | ✅ | 变异位点的 0-based 位置 |
| ref_allele | string | ✅ | 参考碱基（A/C/G/T） |
| alt_alleles | string[] | ✅ | 变异碱基列表（最多 3 个） |

**响应**：

```json
{
  "scores": {"G": -0.80, "C": -0.76, "T": 0.77},
  "ref_prob": 0.246,
  "alt_probs": {"G": 0.110, "C": 0.115, "T": 0.529}
}
```

**分数解读**：

| LLR | 含义 |
|-----|------|
| < -2 | 强烈保守，变异可能有害 |
| -2 ~ 0 | 中度保守 |
| 0 ~ 2 | 弱保守，变异影响较小 |
| > 2 | 不保守，变异碱基更常见 |

---

### 3. 掩码预测

**POST** `/masked-predict`

预测指定位置各碱基的概率分布。

```bash
curl -X POST http://localhost:8005/masked-predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTACGTACGT...", "positions": [100, 200, 255]}'
```

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列（IUPAC 碱基） |
| positions | int[] | ✅ | 要预测的位置列表（最多 100 个） |

**响应**：

```json
{
  "predictions": {
    "100": {"A": 0.246, "C": 0.115, "G": 0.110, "T": 0.529},
    "200": {"A": 0.261, "C": 0.197, "G": 0.060, "T": 0.481},
    "255": {"A": 0.967, "C": 0.006, "G": 0.013, "T": 0.013}
  }
}
```

---

### 4. LoRA 功能预测

**POST** `/predict`

使用微调后的 LoRA 适配器执行特定生物学任务的预测。

```bash
curl -X POST http://localhost:8005/predict \
  -H "Content-Type: application/json" \
  -d '{"sequence": "ACGTACGTACGT...", "task": "acr_arabidopsis"}'
```

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列（IUPAC 碱基，建议 ≥ 600 bp） |
| task | string | ✅ | 任务名称 |

**可用任务**：

| 任务 | 说明 | 输出类型 |
|------|------|----------|
| acr_arabidopsis | 拟南芥 ACR 预测 | 二分类 |
| acr_nine_species | 九物种 ACR 预测 | 二分类 |
| acr_cell_type | 细胞类型 ACR 预测 | 多标签分类（92 类） |
| expression_on_off | 表达量开/关预测 | 二分类 |
| expression_absolute | 表达量绝对值预测 | 回归 |
| translation_on_off | 翻译效率开/关预测 | 二分类 |
| translation_absolute | 翻译效率绝对值预测 | 回归 |

**响应示例**：

二分类：
```json
{"task": "acr_arabidopsis", "prediction": "POSITIVE", "probability": 0.87}
```

多标签分类：
```json
{"task": "acr_cell_type", "prediction": "MULTI_LABEL", "probabilities": [0.12, 0.03, ...], "num_labels": 92}
```

回归：
```json
{"task": "expression_absolute", "prediction": 3.45}
```

---

## 错误处理

### 参数校验错误（422）

```json
{
  "detail": "请求参数校验失败",
  "errors": ["body -> sequence: String should match pattern '^[ACGTN]+$'"]
}
```

### 业务错误（400/404）

```json
{"detail": "Position 100 is out of range for sequence of length 50"}
```

### 服务器错误（500）

```json
{"detail": "服务器内部错误"}
```

---

## IUPAC 碱基代码

支持标准 IUPAC 碱基代码：

| 代码 | 含义 |
|------|------|
| A, C, G, T | 标准碱基 |
| N | 任意碱基（转为 [UNK]） |
| R | A 或 G |
| Y | C 或 T |
| M | A 或 C |
| K | G 或 T |
| S | G 或 C |
| W | A 或 T |
| H | A 或 C 或 T |
| V | A 或 C 或 G |
| D | A 或 G 或 T |
| B | C 或 G 或 T |

---

## 日志

日志位置：`logs/plantcad2_YYYYMMDD.log`

日志内容：
- 请求参数校验失败详情
- 服务器错误堆栈跟踪
- 模型加载状态

---

## 测试结果

详见 `results/` 目录下的测试报告。

| 任务 | 准确率 | 平均耗时 |
|------|--------|----------|
| acr_nine_species | 94.55% | 0.324s |
| acr_arabidopsis | 86.61% | 0.329s |
| expression_on_off | 55.50% | 0.323s |
| translation_on_off | 58.00% | 0.316s |
