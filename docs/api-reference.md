# PlantCAD2 推理服务 API 参考文档

## 服务信息

| 项目 | 说明 |
|------|------|
| 框架 | FastAPI |
| 日志 | loguru（控制台 + 文件轮转） |
| 启动命令 | `CUDA_VISIBLE_DEVICES=3 PLANTCAD2_DEVICE=cuda:0 uvicorn app.main:app --host 0.0.0.0 --port 8005` |
| 模型 | PlantCAD2-Large-l48-d1536（694M 参数） |
| 最大序列长度 | 8192 bp |
| 支持的碱基 | IUPAC 标准（A,C,G,T,N,R,Y,M,K,S,W,H,B,V,D） |
| 推荐上下文 | ≥ 600 bp（LoRA 任务）/ ≥ 2048 bp（变异打分） |

## 功能分类

> **通俗说明**
>
> PlantCAD2 是一个"读过大量植物 DNA 的语言模型"。就像 ChatGPT 理解人类语言一样，它理解 DNA 的"语法"。基于这个能力，它能做以下事情：
>
> - **嵌入提取**：把一段 DNA 序列变成一组数字（向量），方便计算机比较和分析不同序列之间的相似性。
> - **变异打分**：DNA 中某个碱基发生了变化（比如 A 变成了 G），模型评估这个变化是"正常的"还是"异常的"。分数越高表示变化越罕见，可能意味着功能影响越大。
> - **掩码预测**：把 DNA 中某个位置遮住，让模型猜那里应该是什么碱基。如果模型很确定（某个碱基概率 > 0.9），说明这个位置在进化上很保守，不太能随便变。
> - **LoRA 功能预测**：在基座模型基础上叠加一个"专项微调模块"，让它能预测具体的生物学功能。不同模块对应不同任务（见下表）。

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
| sequence | string | ✅ | DNA 序列（IUPAC 碱基，最长 8192 bp） |
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

> **通俗说明**：每个人的 DNA 都会有少量差异（变异），大多数是无害的，但有些会导致疾病或性状改变。这个功能让模型评估一个具体的碱基变化是否"罕见"——如果模型认为这个位置"本来就应该长这样"的概率很低，说明这个变化可能有生物学意义。

零样本评估单核苷酸变异（SNV）的致病性。方法：遮盖目标位置，比较参考碱基和变异碱基的预测概率，计算对数似然比（LLR）。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | 包含变异位点的上下文 DNA 序列（IUPAC 碱基） |
| position | int | ✅ | 变异位点在序列中的 0-based 位置 |
| ref_allele | string | ✅ | 参考碱基（A/C/G/T） |
| alt_alleles | string[] | ✅ | 变异碱基列表（A/C/G/T，最多 3 个） |

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
| sequence | string | ✅ | DNA 序列（IUPAC 碱基，建议 ≥ 600 bp） |
| task | string | ✅ | 任务名称，见下表 |

```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAA...",
  "task": "acr_arabidopsis"
}
```

### 可选任务

#### ACR 预测（活跃顺式调控元件）

> **通俗说明**：ACR（Accessible Chromatin Region）是 DNA 中"开放"的区域。你可以把 DNA 想象成一本书，有些章节被折起来了（关闭/压缩），有些是翻开的（开放）。只有翻开的部分才能被细胞"读取"并调控基因的开关。ACR 预测就是判断一段 DNA 是否属于这种"翻开的"调控区域。
>
> - **POSITIVE** = 这段 DNA 很可能是活跃的调控元件
> - **NEGATIVE** = 这段 DNA 可能不是活跃的调控元件

| task | 任务说明 | 输出类型 | num_labels | LoRA 权重目录 |
|------|----------|----------|------------|---------------|
| acr_arabidopsis | 用拟南芥（一种模式植物）的数据训练，预测 DNA 是否为活跃调控区域 | 二分类 | 2 | cross_species_acr_train_on_arabidopsis_plantcad2_large |
| acr_nine_species | 用 9 种植物的数据联合训练，泛化能力更强，适合非拟南芥物种 | 二分类 | 2 | cross_species_acr_train_on_nine_species_plantcad2_large |
| acr_cell_type | 预测 DNA 在 92 种不同细胞类型中是否为活跃调控区域（同一个基因在不同细胞中的调控状态不同） | 多标签分类 | 92 | cell_type_specific_acr_plantcad2_large |

#### 表达量预测

> **通俗说明**：基因表达量衡量一个基因"工作强度"——DNA 上的基因需要被"读取"（转录）才能产生蛋白质发挥功能。表达量越高，说明这个基因越活跃。
>
> - **on/off（开/关）**：判断基因在某个组织中是否在工作（是/否）
> - **absolute（绝对值）**：预测基因具体的表达水平（一个连续数值，越高越活跃）

| task | 任务说明 | 输出类型 | num_labels | LoRA 权重目录 |
|------|----------|----------|------------|---------------|
| expression_on_off | 预测基因在叶片中是否表达（开/关） | 二分类 | 2 | cross_species_leaf_on_off_expression_plantcad2_large |
| expression_absolute | 预测基因在叶片中的表达水平（连续数值） | 回归 | 1 | cross_species_leaf_absolute_expression_plantcad2_large |

#### 翻译效率预测

> **通俗说明**：基因转录成 mRNA 后，还需要被"翻译"成蛋白质。翻译效率衡量这个过程的效率——同样多的 mRNA，翻译效率越高，产生的蛋白质越多。
>
> - **on/off（开/关）**：判断 mRNA 是否会被翻译成蛋白质
> - **absolute（绝对值）**：预测具体的翻译效率数值

| task | 任务说明 | 输出类型 | num_labels | LoRA 权重目录 |
|------|----------|----------|------------|---------------|
| translation_on_off | 预测 mRNA 在叶片中是否会被翻译（开/关） | 二分类 | 2 | cross_species_leaf_on_off_translation_plantcad2_large |
| translation_absolute | 预测 mRNA 在叶片中的翻译效率（连续数值） | 回归 | 1 | cross_species_leaf_absolute_translation_plantcad2_large |

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

> **通俗说明**：类似"完形填空"——把 DNA 序列中某个位置遮住，让模型根据上下文猜那里应该是什么碱基。如果模型非常确定（某个碱基概率 > 0.9），说明这个位置在进化上高度保守，几乎所有植物都保持这个碱基不变，通常是功能上很重要的位置。

对指定位置进行遮盖，预测该位置各碱基（A/C/G/T）的概率分布。

### 请求

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| sequence | string | ✅ | DNA 序列（IUPAC 碱基） |
| positions | int[] | ✅ | 要遮盖的 0-based 位置列表（最多 100 个） |

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
- 序列的可变性
- 辅助设计实验（如 CRISPR 靶点选择）

---

## 6. 错误处理

### 错误响应格式

#### 参数校验错误（422）

```json
{
  "detail": "请求参数校验失败",
  "errors": [
    "body -> sequence: String should match pattern '^[ACGTN]+$'",
    "body -> position: Input should be greater than or equal to 0"
  ]
}
```

#### 业务错误（400/404）

```json
{
  "detail": "Position 100 is out of range for sequence of length 50"
}
```

#### 服务器错误（500）

```json
{
  "detail": "服务器内部错误"
}
```

### HTTP 状态码

| 状态码 | 含义 | 说明 |
|--------|------|------|
| 200 | 成功 | 请求处理成功 |
| 400 | 请求错误 | 序列过长、位置越界、task 不存在等 |
| 404 | 未找到 | LoRA 适配器目录未找到 |
| 422 | 校验失败 | 参数格式错误、包含无效字符等 |
| 500 | 服务器错误 | 推理内部错误 |

---

## 7. 日志

服务使用 loguru 记录日志，支持控制台输出和文件轮转。

### 日志位置

```
logs/
├── plantcad2_20260618.log    ← 当天日志
├── plantcad2_20260617.log    ← 历史日志（自动轮转）
└── ...
```

### 日志内容

- 请求参数校验失败详情（字段、错误原因、输入值）
- 服务器错误堆栈跟踪
- 模型加载状态
- 推理耗时统计

### 日志级别

| 级别 | 说明 |
|------|------|
| DEBUG | 详细调试信息 |
| INFO | 一般信息（模型加载、请求处理） |
| WARNING | 警告（参数校验失败、LoRA 加载失败） |
| ERROR | 错误（服务器内部错误） |

---

## 8. IUPAC 碱基代码

服务支持标准 IUPAC 碱基代码：

| 代码 | 含义 | 说明 |
|------|------|------|
| A | Adenine | 腺嘌呤 |
| C | Cytosine | 胞嘧啶 |
| G | Guanine | 鸟嘌呤 |
| T | Thymine | 胸腺嘧啶 |
| N | Any | 任意碱基（转为 [UNK] token） |
| R | Purine | A 或 G |
| Y | Pyrimidine | C 或 T |
| M | Amino | A 或 C |
| K | Keto | G 或 T |
| S | Strong | G 或 C |
| W | Weak | A 或 T |
| H | not G | A 或 C 或 T |
| V | not T | A 或 C 或 G |
| D | not C | A 或 G 或 T |
| B | not A | C 或 G 或 T |

**说明**：N 会被 tokenizer 自动转换为 [UNK] token，模型会根据上下文推断该位置的碱基。其他模糊碱基代码也会被转换为对应的 token。

---

## 9. 测试结果

详见 `results/` 目录下的测试报告：

- `20260618_test_report.md` — 最新测试报告
- `20260617_test_report.md` — 历史测试报告

### 测试摘要（2026-06-18）

| 任务 | 准确率 | 样本数 | 平均耗时 |
|------|--------|--------|----------|
| acr_nine_species | 94.55% | 1,100 | 0.324s |
| acr_arabidopsis | 86.61% | 2,300 | 0.329s |
| acr_cell_type | — | 100 | 0.320s |
| expression_on_off | 55.50% | 400 | 0.323s |
| expression_absolute | — | 400 | 0.318s |
| translation_on_off | 58.00% | 100 | 0.316s |
| translation_absolute | — | 100 | 0.318s |
