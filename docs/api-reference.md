# PlantCAD2 推理服务 API 参考文档

## 服务信息

| 项目 | 说明 |
|------|------|
| 框架 | FastAPI |
| 启动命令 | `CUDA_VISIBLE_DEVICES=2 PLANTCAD2_DEVICE=cuda:0 uvicorn app.main:app --host 0.0.0.0 --port 8005` |
| 模型 | PlantCAD2-Large-l48-d1536（694M 参数） |
| 最大序列长度 | 8192 bp |
| 推荐上下文 | ≥ 600 bp（LoRA 任务）/ ≥ 2048 bp（变异打分） |

## 功能分类

| 端点 | 功能 | 模型要求 | 说明 |
|------|------|----------|------|
| GET /health | 健康检查 | — | 服务状态检测 |
| POST /embeddings | 嵌入提取 | **基座模型** | PlantCAD2 自带，无需额外权重 |
| POST /variant-score | 变异打分 | **基座模型** | PlantCAD2 自带的零样本能力 |
| POST /masked-predict | 掩码预测 | **基座模型** | PlantCAD2 自带的掩码语言模型能力 |
| POST /predict | LoRA 预测 | **基座模型 + LoRA 适配器** | 需要下载对应的微调权重才能使用 |

---

## 1. 健康检查

**GET** `/health`

检查服务是否就绪、模型是否加载完成。

### 响应

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda:0"
}
```

---

## 2. 嵌入提取

**POST** `/embeddings`

提取 DNA 序列每个位置的 1536 维向量表示。使用 RCPS（反向互补参数共享）技术，输出为正向和反向互补嵌入的平均值。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列（ACGT，最长 8192 bp） |
| normalize | bool | ❌ | 是否 L2 归一化，默认 true |

```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAA...",
  "normalize": true
}
```

### 响应

| 字段 | 类型 | 说明 |
|------|------|------|
| embeddings | float[][] | 每个位置的 1536 维向量 |
| shape | int[] | 形状，如 [512, 1536] |
| sequence_length | int | token 化后的序列长度 |

```json
{
  "embeddings": [[0.012, -0.034, ...], ...],
  "shape": [512, 1536],
  "sequence_length": 512
}
```

### 用途

- 序列相似性比较
- 聚类、降维可视化
- 作为下游机器学习模型的特征输入

---

## 3. 变异打分

**POST** `/variant-score`

零样本评估单核苷酸变异（SNV）的致病性。方法：遮盖目标位置，比较参考碱基和变异碱基的预测概率，计算对数似然比（LLR）。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | 包含变异位点的上下文 DNA 序列 |
| position | int | ✅ | 变异位点在序列中的 0-based 位置 |
| ref_allele | string | ✅ | 参考碱基（A/C/G/T） |
| alt_alleles | string[] | ✅ | 变异碱基列表 |

```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAA...",
  "position": 100,
  "ref_allele": "A",
  "alt_alleles": ["G", "C", "T"]
}
```

### 响应

| 字段 | 类型 | 说明 |
|------|------|------|
| scores | object | 每个变异碱基的 LLR 分数 |
| ref_prob | float | 参考碱基的预测概率 |
| alt_probs | object | 每个变异碱基的预测概率 |

```json
{
  "scores": {"G": -0.80, "C": -0.76, "T": 0.77},
  "ref_prob": 0.246,
  "alt_probs": {"G": 0.110, "C": 0.115, "T": 0.529}
}
```

### 分数解读

| LLR 范围 | 含义 |
|----------|------|
| < -2 | 强烈保守，变异可能有害 |
| -2 ~ 0 | 中度保守 |
| 0 ~ 2 | 弱保守，变异影响较小 |
| > 2 | 不保守，变异碱基更常见 |

---

## 4. LoRA 功能预测（需要额外适配器权重）

**POST** `/predict`

> **前提条件**：需要在 `models/` 目录下下载对应的 LoRA 适配器权重。仅靠基座模型无法使用此端点。

使用微调后的 LoRA 适配器执行特定生物学任务的预测。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列（建议 ≥ 600 bp） |
| task | string | ✅ | 任务名称，见下表 |

```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAA...",
  "task": "acr_arabidopsis"
}
```

### 可选任务

| task | 任务说明 | 输出类型 | num_labels | LoRA 权重目录 |
|------|----------|----------|------------|---------------|
| acr_arabidopsis | ACR 预测（拟南芥训练集） | 二分类 | 2 | cross_species_acr_train_on_arabidopsis_plantcad2_large |
| acr_nine_species | ACR 预测（九物种训练集） | 二分类 | 2 | cross_species_acr_train_on_nine_species_plantcad2_large |
| acr_cell_type | ACR 预测（细胞类型特异性） | 多标签分类 | 92 | cell_type_specific_acr_plantcad2_large |
| expression_on_off | 叶片表达量（开/关） | 二分类 | 2 | cross_species_leaf_on_off_expression_plantcad2_large |
| expression_absolute | 叶片表达量（绝对值） | 回归 | 1 | cross_species_leaf_absolute_expression_plantcad2_large |
| translation_on_off | 叶片翻译效率（开/关） | 二分类 | 2 | cross_species_leaf_on_off_translation_plantcad2_large |
| translation_absolute | 叶片翻译效率（绝对值） | 回归 | 1 | cross_species_leaf_absolute_translation_plantcad2_large |

### 响应 — 二分类任务

```json
{
  "task": "acr_arabidopsis",
  "prediction": "POSITIVE",
  "probability": 0.87
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| task | string | 任务名称 |
| prediction | string | POSITIVE 或 NEGATIVE |
| probability | float | POSITIVE 的概率（0~1） |

### 响应 — 多标签分类任务（acr_cell_type）

```json
{
  "task": "acr_cell_type",
  "prediction": "MULTI_LABEL",
  "probabilities": [0.12, 0.03, 0.78, ...],
  "num_labels": 92
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| task | string | 任务名称 |
| prediction | string | 固定为 MULTI_LABEL |
| probabilities | float[] | 92 个细胞类型的概率（sigmoid） |
| num_labels | int | 标签数量（92） |

### 响应 — 回归任务

```json
{
  "task": "expression_absolute",
  "prediction": 3.45
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| task | string | 任务名称 |
| prediction | float | 预测的连续值 |

---

## 5. 掩码位置预测

**POST** `/masked-predict`

对指定位置进行遮盖，预测该位置各碱基（A/C/G/T）的概率分布。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列 |
| positions | int[] | ✅ | 要遮盖的 0-based 位置列表 |

```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAA...",
  "positions": [100, 200, 255]
}
```

### 响应

```json
{
  "predictions": {
    "100": {"A": 0.246, "C": 0.115, "G": 0.110, "T": 0.529},
    "200": {"A": 0.261, "C": 0.197, "G": 0.060, "T": 0.481},
    "255": {"A": 0.967, "C": 0.006, "G": 0.013, "T": 0.013}
  }
}
```

### 用途

- 识别保守位点（某个碱基概率 > 0.9）
- 评估序列的可变性
- 辅助设计实验（如 CRISAM 靶点选择）

---

## 错误码

| HTTP 状态码 | 含义 |
|------------|------|
| 200 | 成功 |
| 400 | 请求参数错误（序列过长、位置越界、task 不存在等） |
| 404 | LoRA 适配器目录未找到 |
| 500 | 推理内部错误 |
