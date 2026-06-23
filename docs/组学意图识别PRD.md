# 组学智能体意图识别 PRD 文档

> **文档版本**：v1.0
> **创建日期**：2026-06-22
> **最后更新**：2026-06-22
> **负责人**：lvyizhuo

---

## 一、项目背景

### 1.1 项目概述

"农业大模型"项目包含多个智能体模块，其中"组学智能体"负责基因序列相关的预测和分析任务。组学智能体目前包含两个已部署的模型：

- **EVO2**：基因序列预测生成模型
- **PlantCAD2**：植物基因组DNA语言模型（694M参数），支持嵌入提取、变异打分、掩码预测、LoRA功能预测等10个接口

（AlphaFold3 暂未接入）

### 1.2 问题描述

当前系统的总体流程为：用户在"主智能体"界面提问 → 主智能体进行第一层意图识别 → 跳转到对应智能体。由于"组学智能体"包含多个模型和多种任务，需要在主智能体跳转到组学智能体后，进行**二次意图识别（组学意图识别）**，以确定用户具体需要调用哪个任务接口。

### 1.3 系统流程

```
用户提问
    ↓
"主智能体"收到请求，进行第一层意图识别
    ↓
返回智能体标号（如：3 = 组学智能体）
    ↓
前端请求"组学智能体意图识别"接口（端口8010）
    ↓
二次意图识别（组学意图识别）
    ↓
返回具体任务对应的接口序号
    ↓
前端调用对应的任务接口（PlantCAD2:8005 / EVO2转发接口）
```

---

## 二、需求分析

### 2.1 功能需求

| 需求编号 | 需求描述 | 优先级 |
|---------|---------|--------|
| F-001 | 支持EVO2和PlantCAD2两个模型的任务跳转 | P0 |
| F-002 | 对每个任务设计全面的、多场景的过滤提示词 | P0 |
| F-003 | 接收用户问题，通过大模型进行意图识别 | P0 |
| F-004 | 返回置信度较高的任务序号和结果 | P0 |
| F-005 | 返回引导词，引导用户选择任务和输入数据 | P1 |
| F-006 | 支持高置信度场景下直接提取参数并调用接口 | P0 |

### 2.2 返回场景设计

意图识别的返回包含三种情况：

#### 场景一：高置信度（直接执行）

**触发条件**：用户输入明确包含任务类型、数据和参数

**处理逻辑**：
1. 返回任务接口序号
2. 大模型提取用户问题中的数据和参数
3. 直接调用该任务接口进行计算
4. 返回任务序号和计算结果

**示例**：
```
用户输入："帮我预测这段DNA序列的表达量开/关：CTTAATTAATATTGCCTTTGTAATAACGCGCGAAACACAAATCTTCTCTGCCTAATGCAGTAGTCATGTGTTGACTCCTTCAAAATTTCCAAGAAGTTAGTGGCTGGTGTGTCATTGTCTTCATCTTTTTTTTTTTTTTTTTAAAAATTGAATGCGACATGTACTCCTCAACGTATAAGCTCAATGCTTGTTACTGAAACATCTCTTGTCTGATTTTTTCAGGCTAAGTCTTACAGAAAGTGATTGGGCACTTCAATGGCTTTCACAAATGAAAAAGATGGATCTAAGGGATTTGTGAAGAGAGTGGCTTCATCTTTCTCCATGAGGAAGAAGAAGAATGCAACAAGTGAACCCAAGTTGCTTCCAAGATCGAAATCAACAGGTTCTGCTAACTTTGAATCCATGAGGCTACCTGCAACGAAGAAGATTTCAGATGTCACAAACAAAACAAGGATCAAACCATTAGGTGGTGTAGCACCAGCACAACCAAGAAGGGAAAAGATCGATGATCG"

返回：
{
  "confidence": "high",
  "task_id": 207,
  "task_name": "表达量开/关预测",
  "model": "PlantCAD2",
  "params": {
    "sequence": "CTTAATTAATATTGCCTTTGTAA...",
    "task": "expression_on_off"
  },
  "result": {
    "task": "expression_on_off",
    "prediction": "POSITIVE",
    "probability": 0.87
  },
  "guide_message": "已为您完成表达量开/关预测，结果为 POSITIVE（概率：87%）"
}
```

#### 场景二：中置信度（推荐任务）

**触发条件**：用户只提供了关键字，但未提供数据和具体任务指示

**处理逻辑**：
1. 返回置信度较高的4-5个任务序号
2. 返回每个任务的简要说明和引导词

**示例**：
```
用户输入："我想分析一下基因序列"

返回：
{
  "confidence": "medium",
  "suggested_tasks": [
    {
      "task_id": 201,
      "task_name": "嵌入提取",
      "model": "PlantCAD2",
      "description": "提取DNA序列每个位置的1536维向量表示，用于序列相似性比较和聚类分析",
      "guide_message": "请提供DNA序列（IUPAC碱基，最长8192bp），我将为您提取嵌入向量"
    },
    {
      "task_id": 202,
      "task_name": "变异打分",
      "model": "PlantCAD2",
      "description": "评估单核苷酸变异的致病性，判断变异是否有生物学意义",
      "guide_message": "请提供DNA序列、变异位置、参考碱基和变异碱基，我将为您评估变异影响"
    },
    {
      "task_id": 203,
      "task_name": "掩码预测",
      "model": "PlantCAD2",
      "description": "预测指定位置各碱基的概率分布，识别保守位点",
      "guide_message": "请提供DNA序列和要预测的位置列表，我将为您分析各碱基的概率"
    },
    {
      "task_id": 101,
      "task_name": "基因序列预测生成",
      "model": "EVO2",
      "description": "给定一段基因序列，预测并生成后续序列",
      "guide_message": "请提供起始DNA序列，我将为您预测生成后续序列"
    }
  ],
  "guide_message": "您想进行哪种基因序列分析？请提供DNA序列数据"
}
```

#### 场景三：低置信度（兜底引导）

**触发条件**：用户问题模糊，无法判别具体任务

**处理逻辑**：
1. 返回兜底提示语
2. 列出所有可用任务供用户选择
3. 引导用户输入数据

**示例**：
```
用户输入："帮我看看"

返回：
{
  "confidence": "low",
  "guide_message": "您好！我是组学智能体，可以为您提供以下基因序列分析服务：\n\n【PlantCAD2 模型】\n1. 嵌入提取 - 提取DNA序列的向量表示\n2. 变异打分 - 评估变异的致病性\n3. 掩码预测 - 预测指定位置的碱基概率\n4. ACR预测 - 预测活跃顺式调控元件\n5. 表达量预测 - 预测基因表达水平\n6. 翻译效率预测 - 预测翻译效率\n\n【EVO2 模型】\n7. 基因序列预测生成 - 预测并生成后续序列\n\n请选择您需要的任务，并提供相应的DNA序列数据。",
  "available_tasks": [
    {"task_id": 201, "task_name": "嵌入提取", "model": "PlantCAD2"},
    {"task_id": 202, "task_name": "变异打分", "model": "PlantCAD2"},
    {"task_id": 203, "task_name": "掩码预测", "model": "PlantCAD2"},
    {"task_id": 204, "task_name": "ACR预测-拟南芥", "model": "PlantCAD2"},
    {"task_id": 205, "task_name": "ACR预测-九物种", "model": "PlantCAD2"},
    {"task_id": 206, "task_name": "ACR预测-细胞类型", "model": "PlantCAD2"},
    {"task_id": 207, "task_name": "表达量预测-开/关", "model": "PlantCAD2"},
    {"task_id": 208, "task_name": "表达量预测-绝对值", "model": "PlantCAD2"},
    {"task_id": 209, "task_name": "翻译效率预测-开/关", "model": "PlantCAD2"},
    {"task_id": 210, "task_name": "翻译效率预测-绝对值", "model": "PlantCAD2"},
    {"task_id": 101, "task_name": "基因序列预测生成", "model": "EVO2"}
  ]
}
```

---

## 三、任务编号设计

### 3.1 任务ID规则

- **1xx**：EVO2 模型任务
- **2xx**：PlantCAD2 模型任务

### 3.2 任务列表

| 任务ID | 任务名称 | 模型 | 接口路径 | 请求参数 | 输出类型 |
|--------|---------|------|----------|----------|----------|
| 101 | 基因序列预测生成 | EVO2 | POST /api/v1/generate | prompt, numTokens, temperature, topK, topP, showLogits | 序列+置信度 |
| 201 | 嵌入提取 | PlantCAD2 | POST /embeddings | sequence, normalize | 向量矩阵 |
| 202 | 变异打分 | PlantCAD2 | POST /variant-score | sequence, position, ref_allele, alt_alleles | LLR分数 |
| 203 | 掩码预测 | PlantCAD2 | POST /masked-predict | sequence, positions | 碱基概率分布 |
| 204 | ACR预测-拟南芥 | PlantCAD2 | POST /predict | sequence, task="acr_arabidopsis" | 二分类 |
| 205 | ACR预测-九物种 | PlantCAD2 | POST /predict | sequence, task="acr_nine_species" | 二分类 |
| 206 | ACR预测-细胞类型 | PlantCAD2 | POST /predict | sequence, task="acr_cell_type" | 多标签分类(92类) |
| 207 | 表达量预测-开/关 | PlantCAD2 | POST /predict | sequence, task="expression_on_off" | 二分类 |
| 208 | 表达量预测-绝对值 | PlantCAD2 | POST /predict | sequence, task="expression_absolute" | 回归 |
| 209 | 翻译效率预测-开/关 | PlantCAD2 | POST /predict | sequence, task="translation_on_off" | 二分类 |
| 210 | 翻译效率预测-绝对值 | PlantCAD2 | POST /predict | sequence, task="translation_absolute" | 回归 |

---

## 四、接口设计

### 4.1 意图识别接口

**接口地址**：`POST http://localhost:8010/intent/recognize`

**请求参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_input | string | ✅ | 用户输入的问题文本 |
| session_id | string | ❌ | 会话ID，用于上下文关联 |

**请求示例**：
```json
{
  "user_input": "帮我预测这段DNA序列的表达量开/关：CTTAATTAATATTGCCTTTGTAA...",
  "session_id": "session_123456"
}
```

**响应参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| confidence | string | 置信度：high/medium/low |
| task_id | int | 任务ID（高置信度时返回） |
| task_name | string | 任务名称（高置信度时返回） |
| model | string | 使用的模型名称 |
| params | object | 提取的参数（高置信度时返回） |
| result | object | 计算结果（高置信度时返回） |
| suggested_tasks | array | 推荐任务列表（中置信度时返回） |
| guide_message | string | 引导消息 |
| available_tasks | array | 可用任务列表（低置信度时返回） |

### 4.2 接口服务信息

| 项目 | 说明 |
|------|------|
| 框架 | FastAPI |
| 端口 | 8010 |
| 大模型 | qwen3.7-max-2026-05-17（阿里云百炼） |
| 启动命令 | `uvicorn app.main:app --host 0.0.0.0 --port 8010` |

---

## 五、提示词设计

### 5.1 系统提示词

```
你是一个组学智能体意图识别助手。你的任务是根据用户输入，精准识别其意图，并映射到具体的任务接口。

### 🎯 任务能力矩阵

| 任务ID | 任务名称 | 模型 | 核心职责 | 输入要求 |
|--------|---------|------|----------|----------|
| 101 | 基因序列预测生成 | EVO2 | 给定一段基因序列，预测并生成后续序列 | prompt（DNA序列） |
| 201 | 嵌入提取 | PlantCAD2 | 提取DNA序列每个位置的1536维向量表示 | sequence（DNA序列） |
| 202 | 变异打分 | PlantCAD2 | 评估单核苷酸变异的致病性 | sequence, position, ref_allele, alt_alleles |
| 203 | 掩码预测 | PlantCAD2 | 预测指定位置各碱基的概率分布 | sequence, positions |
| 204 | ACR预测-拟南芥 | PlantCAD2 | 预测DNA是否为活跃调控区域（拟南芥训练） | sequence |
| 205 | ACR预测-九物种 | PlantCAD2 | 预测DNA是否为活跃调控区域（9物种联合训练） | sequence |
| 206 | ACR预测-细胞类型 | PlantCAD2 | 预测DNA在92种细胞类型中是否为活跃调控区域 | sequence |
| 207 | 表达量预测-开/关 | PlantCAD2 | 预测基因在叶片中是否表达 | sequence |
| 208 | 表达量预测-绝对值 | PlantCAD2 | 预测基因在叶片中的表达水平 | sequence |
| 209 | 翻译效率预测-开/关 | PlantCAD2 | 预测mRNA是否会被翻译 | sequence |
| 210 | 翻译效率预测-绝对值 | PlantCAD2 | 预测mRNA的翻译效率 | sequence |

### ⚙️ 意图识别逻辑（按优先级执行）

1. **数据完整性检查**：检查用户是否提供了完整的数据和参数
   - 如果用户提供了完整的任务指示+数据+参数 → 高置信度，直接执行
   - 如果用户只提供了关键字或部分信息 → 中置信度，推荐任务
   - 如果用户问题模糊不清 → 低置信度，兜底引导

2. **任务匹配规则**：
   - 关键词"生成/预测序列/续写" → 101（EVO2）
   - 关键词"嵌入/向量/表示/相似性" → 201
   - 关键词"变异/SNP/突变/致病性/打分" → 202
   - 关键词"掩码/遮盖/保守/概率分布" → 203
   - 关键词"ACR/染色质/调控元件/顺式调控" → 204/205/206
   - 关键词"表达量/表达水平/基因表达" → 207/208
   - 关键词"翻译/翻译效率/mRNA" → 209/210

3. **数据提取规则**：
   - DNA序列：识别A/C/G/T/N/R/Y/M/K/S/W/H/V/D组成的序列
   - 变异位置：识别数字位置信息
   - 碱基信息：识别A/C/G/T参考碱基和变异碱基

### 📝 输出格式要求

**高置信度输出**（JSON格式）：
```json
{
  "confidence": "high",
  "task_id": 207,
  "task_name": "表达量开/关预测",
  "model": "PlantCAD2",
  "params": {
    "sequence": "提取的DNA序列",
    "task": "expression_on_off"
  }
}
```

**中置信度输出**（JSON格式）：
```json
{
  "confidence": "medium",
  "suggested_tasks": [201, 202, 203, 101],
  "guide_message": "您想进行哪种分析？"
}
```

**低置信度输出**（JSON格式）：
```json
{
  "confidence": "low",
  "guide_message": "请选择任务并提供数据"
}
```

### ⚠️ 输出强制约束
- 输出必须是有效的JSON格式
- 禁止输出任何解释性文字
- 必须包含confidence字段
- 高置信度必须包含task_id和params字段
```

### 5.2 任务过滤提示词

#### 5.2.1 EVO2 基因序列预测生成（任务ID: 101）

**触发关键词**：
- 生成序列、预测序列、续写、序列生成、基因生成、DNA生成
- 后续序列、延伸、扩展序列

**典型用户问题场景**：
```
1. "帮我生成这段DNA的后续序列：ACGT..."
2. "预测一下这个基因序列后面是什么"
3. "续写这段DNA：ACGT..."
4. "根据这段序列生成100bp的后续序列"
5. "基因序列预测，输入：ACGT..."
6. "帮我延伸这段DNA序列到500bp"
7. "这段序列后面会是什么碱基？"
8. "用EVO2预测一下"
```

**数据提取规则**：
- 提取prompt字段：用户提供的DNA序列
- 提取numTokens字段：生成序列长度（默认1200）
- 提取temperature字段：温度系数（默认0.1）
- 提取topK字段：候选词数量（默认4）
- 提取topP字段：累积概率（默认0.5）

---

#### 5.2.2 PlantCAD2 嵌入提取（任务ID: 201）

**触发关键词**：
- 嵌入、向量、表示、embedding、特征提取
- 相似性比较、聚类、降维、可视化

**典型用户问题场景**：
```
1. "提取这段DNA序列的嵌入向量"
2. "帮我把这段序列转成向量表示"
3. "计算这两段序列的相似性"
4. "提取特征用于聚类分析"
5. "这段DNA的embedding是什么？"
6. "序列特征提取：ACGT..."
7. "帮我生成这段序列的向量表示"
8. "提取DNA嵌入用于下游分析"
```

**数据提取规则**：
- 提取sequence字段：用户提供的DNA序列
- 提取normalize字段：是否归一化（默认true）

---

#### 5.2.3 PlantCAD2 变异打分（任务ID: 202）

**触发关键词**：
- 变异、SNP、突变、打分、致病性、LLR
- 碱基变化、单核苷酸多态性、变异评估

**典型用户问题场景**：
```
1. "评估这个SNP的致病性"
2. "变异打分：位置100，A变成G"
3. "这个突变有没有影响？"
4. "帮我分析一下这个碱基变化"
5. "LLR分数是多少？"
6. "变异位点100，参考碱基A，变异碱基G/C/T"
7. "这段序列的第100位A->G变异有害吗？"
8. "评估变异：sequence=ACGT..., position=100, ref=A, alt=G"
```

**数据提取规则**：
- 提取sequence字段：包含变异位点的上下文DNA序列
- 提取position字段：变异位点位置（0-based）
- 提取ref_allele字段：参考碱基（A/C/G/T）
- 提取alt_alleles字段：变异碱基列表

---

#### 5.2.4 PlantCAD2 掩码预测（任务ID: 203）

**触发关键词**：
- 掩码、遮盖、保守、概率分布、完形填空
- 位置预测、碱基概率、进化保守

**典型用户问题场景**：
```
1. "预测位置100的碱基概率"
2. "这个位置保守吗？"
3. "掩码预测：序列ACGT...，位置[100,200]"
4. "帮我分析这些位置的碱基分布"
5. "位置255应该是什么碱基？"
6. "完形填空预测"
7. "哪些位置是保守的？"
8. "预测序列ACGT...的第100、200、255位碱基"
```

**数据提取规则**：
- 提取sequence字段：DNA序列
- 提取positions字段：要预测的位置列表（0-based）

---

#### 5.2.5 PlantCAD2 ACR预测（任务ID: 204/205/206）

**触发关键词**：
- ACR、染色质、调控元件、顺式调控、开放染色质
- 活跃区域、调控区域、染色质可及性

**典型用户问题场景**：
```
1. "预测这段DNA是否为ACR区域"
2. "这段序列是活跃调控元件吗？"
3. "染色质可及性预测"
4. "帮我分析这段序列的调控功能"
5. "这段DNA在拟南芥中是ACR吗？"
6. "细胞类型特异性ACR预测"
7. "顺式调控元件预测"
8. "这段序列在不同细胞类型中的调控状态"
```

**任务选择逻辑**：
- 如果用户提到"拟南芥"或"Arabidopsis" → 204
- 如果用户提到"多物种"、"泛化"、"九物种" → 205
- 如果用户提到"细胞类型"、"特异性"、"92种" → 206
- 如果未指定，默认推荐205（泛化能力最强）

**数据提取规则**：
- 提取sequence字段：DNA序列

---

#### 5.2.6 PlantCAD2 表达量预测（任务ID: 207/208）

**触发关键词**：
- 表达量、表达水平、基因表达、转录水平
- 开/关、是否表达、表达绝对值

**典型用户问题场景**：
```
1. "预测这个基因的表达量"
2. "这段DNA在叶片中会表达吗？"
3. "基因表达水平预测"
4. "表达量开/关分类"
5. "预测表达的绝对值"
6. "这个基因活跃吗？"
7. "叶片表达量预测"
8. "基因转录水平分析"
```

**任务选择逻辑**：
- 如果用户提到"开/关"、"是否表达"、"会不会表达" → 207
- 如果用户提到"绝对值"、"表达水平"、"具体数值" → 208
- 如果未指定，默认推荐207（开/关分类）

**数据提取规则**：
- 提取sequence字段：DNA序列

---

#### 5.2.7 PlantCAD2 翻译效率预测（任务ID: 209/210）

**触发关键词**：
- 翻译、翻译效率、mRNA翻译、蛋白质合成
- 翻译开/关、翻译绝对值

**典型用户问题场景**：
```
1. "预测这段mRNA的翻译效率"
2. "这段序列会被翻译吗？"
3. "翻译效率预测"
4. "mRNA翻译开/关分类"
5. "翻译效率绝对值预测"
6. "蛋白质合成效率"
7. "这段mRNA能翻译成蛋白质吗？"
8. "翻译丰度预测"
```

**任务选择逻辑**：
- 如果用户提到"开/关"、"会不会翻译"、"是否翻译" → 209
- 如果用户提到"绝对值"、"效率数值"、"翻译丰度" → 210
- 如果未指定，默认推荐209（开/关分类）

**数据提取规则**：
- 提取sequence字段：DNA/mRNA序列

---

## 六、技术实现

### 6.1 服务架构

```
┌─────────────────────────────────────────────────────────────┐
│                    组学意图识别服务 (8010)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   FastAPI     │    │  意图识别引擎  │    │  参数提取器   │  │
│  │   路由层      │ →  │  (LLM调用)    │ →  │  (LLM调用)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                              ↓                    ↓         │
│                    ┌─────────────────────────────────────┐  │
│                    │         接口调用层                    │  │
│                    │  PlantCAD2 (8005)  EVO2 (转发接口)    │  │
│                    └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 核心流程

```python
async def recognize_intent(user_input: str) -> dict:
    """
    意图识别主流程
    """
    # 1. 调用LLM进行意图识别
    intent_result = await llm_intent_recognize(user_input)
    
    # 2. 根据置信度处理
    if intent_result["confidence"] == "high":
        # 高置信度：提取参数并调用接口
        params = await llm_extract_params(user_input, intent_result["task_id"])
        result = await call_task_api(intent_result["task_id"], params)
        return {
            "confidence": "high",
            "task_id": intent_result["task_id"],
            "task_name": intent_result["task_name"],
            "model": intent_result["model"],
            "params": params,
            "result": result,
            "guide_message": generate_success_message(intent_result, result)
        }
    elif intent_result["confidence"] == "medium":
        # 中置信度：返回推荐任务
        return {
            "confidence": "medium",
            "suggested_tasks": intent_result["suggested_tasks"],
            "guide_message": generate_suggest_message(intent_result)
        }
    else:
        # 低置信度：返回兜底引导
        return {
            "confidence": "low",
            "guide_message": generate_fallback_message(),
            "available_tasks": get_all_tasks()
        }
```

### 6.3 大模型配置

| 配置项 | 值 |
|--------|-----|
| 模型名称 | qwen3.7-max-2026-05-17 |
| API Key | sk-ws-H.RPYIIPP.0jom.MEQCIFn-x-MmtOHGQ3D_ajVCz42gWmaAAVXkolyVJxEGXJCWAiAGKAfKZTvIwTZHoARJC0tmW0jhxEytahVxOumDcwSsQQ |
| Base URL | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| 请求地址 | POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions |

### 6.4 下游服务地址

| 服务 | 地址 | 端口 |
|------|------|------|
| PlantCAD2 推理服务 | http://localhost:8005 | 8005 |
| EVO2 转发接口 | http://36.137.205.153:8666/api/v1/generate | 8666 |
| 组学意图识别服务 | http://localhost:8010 | 8010 |

---

## 七、目录结构

```
omics-intent-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI入口
│   ├── config.py               # 配置文件
│   ├── routers/
│   │   ├── __init__.py
│   │   └── intent.py           # 意图识别路由
│   ├── services/
│   │   ├── __init__.py
│   │   ├── intent_recognizer.py  # 意图识别引擎
│   │   ├── param_extractor.py    # 参数提取器
│   │   └── api_caller.py         # 接口调用层
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── system_prompt.py      # 系统提示词
│   │   └── task_prompts.py       # 任务过滤提示词
│   └── schemas/
│       ├── __init__.py
│       └── requests.py          # 请求/响应模型
├── logs/                        # 日志目录
├── requirements.txt             # 依赖
└── README.md                    # 说明文档
```

---

## 八、测试用例

### 8.1 高置信度测试用例

| 测试ID | 用户输入 | 预期任务ID | 预期置信度 |
|--------|---------|-----------|-----------|
| TC-001 | "帮我预测这段DNA序列的表达量开/关：CTTAATTAATATTGCCTTTGTAA..." | 207 | high |
| TC-002 | "评估这个SNP：序列ACGT...位置100，A变成G" | 202 | high |
| TC-003 | "生成后续序列，输入ACGT..." | 101 | high |
| TC-004 | "提取嵌入向量：ACGT..." | 201 | high |
| TC-005 | "掩码预测位置100、200：ACGT..." | 203 | high |

### 8.2 中置信度测试用例

| 测试ID | 用户输入 | 预期推荐任务数 | 预期置信度 |
|--------|---------|---------------|-----------|
| TC-006 | "我想分析基因序列" | 4-5 | medium |
| TC-007 | "变异分析" | 3-4 | medium |
| TC-008 | "预测表达量" | 2-3 | medium |

### 8.3 低置信度测试用例

| 测试ID | 用户输入 | 预期置信度 |
|--------|---------|-----------|
| TC-009 | "帮我看看" | low |
| TC-010 | "你好" | low |
| TC-011 | "能做什么" | low |

---

## 九、异常处理

### 9.1 异常场景

| 异常类型 | 处理方式 |
|---------|---------|
| 用户输入为空 | 返回低置信度兜底引导 |
| LLM调用超时 | 重试3次，失败返回兜底引导 |
| LLM返回格式错误 | 解析失败返回兜底引导 |
| 下游接口调用失败 | 返回任务推荐，提示用户手动调用 |
| DNA序列格式错误 | 提示用户检查序列格式 |

### 9.2 错误码定义

| 错误码 | 说明 |
|--------|------|
| 0 | 成功 |
| 1001 | 意图识别失败 |
| 1002 | 参数提取失败 |
| 1003 | 下游接口调用失败 |
| 1004 | LLM服务异常 |
| 1005 | 请求参数格式错误 |

---

## 十、性能要求

| 指标 | 要求 |
|------|------|
| 意图识别响应时间 | ≤ 3秒（不含下游接口调用） |
| 高置信度端到端响应时间 | ≤ 10秒（含下游接口调用） |
| 并发支持 | ≥ 50 QPS |
| 可用性 | ≥ 99.5% |

---

## 十一、后续规划

### 11.1 迭代计划

| 版本 | 内容 | 时间 |
|------|------|------|
| v1.0 | 基础意图识别+三种置信度场景 | 2026-06-25 |
| v1.1 | 多轮对话支持+上下文记忆 | 2026-07-01 |
| v1.2 | AlphaFold3接入 | 待定 |

### 11.2 优化方向

1. **提示词优化**：根据实际使用情况持续优化过滤提示词
2. **置信度调优**：调整置信度阈值，提高识别准确率
3. **缓存机制**：对常见问题添加缓存，提高响应速度
4. **A/B测试**：对比不同提示词版本的效果

---

## 附录

### A. 术语表

| 术语 | 说明 |
|------|------|
| ACR | Accessible Chromatin Region，活跃顺式调控元件 |
| SNP | Single Nucleotide Polymorphism，单核苷酸多态性 |
| LLR | Log-Likelihood Ratio，对数似然比 |
| LoRA | Low-Rank Adaptation，低秩适配 |
| IUPAC | 国际纯粹与应用化学联合会碱基代码标准 |

### B. 参考文档

- [PlantCAD2 API文档](./api-reference.md)
- [PlantCAD2接口文档](./api-接口文档.md)
- [EVO2推理接口文档](./EVO2推理——接口文档.docx)
- [EVO2转发接口文档](./EVO2转发接口接口文档.xlsx)
- [主智能体意图识别实现](../task06-ytModule/intent_recognizer.py)
