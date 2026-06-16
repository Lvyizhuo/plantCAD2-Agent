# PlantCAD2 二次开发文档

## 项目目标

搭建 PlantCAD2 推理服务，作为组学智能体的一部分，与已部署的 Evo 2 模型对接。Evo 2 负责 DNA 序列生成，PlantCAD2 负责接收生成序列并执行下游评估任务，通过 FastAPI 对外提供 REST API。

## 架构设计

```
PlantCaduceus/
├── src/                    # 原始仓库代码（只读，不修改）
├── pretrain/               # 原始仓库代码（只读，不修改）
├── models/                 # 模型权重（基座 + 7 个 LoRA）
├── app/                    # FastAPI 服务（二次开发）
│   ├── main.py             # FastAPI 入口
│   ├── routers/            # 路由定义
│   │   ├── embeddings.py   # /embeddings 端点
│   │   ├── variant.py      # /variant-score 端点
│   │   ├── predict.py      # /predict 端点
│   │   └── masked.py       # /masked-predict 端点
│   ├── schemas/            # 请求/响应 Pydantic 模型
│   │   └── requests.py
│   └── config.py           # 服务配置（模型路径、设备等）
├── modules/                # 推理业务逻辑（二次开发）
│   ├── engine.py           # PlantCAD2Engine 推理引擎主类
│   ├── model_loader.py     # 模型加载（从 src/ 提取并适配）
│   ├── embedding.py        # 序列嵌入提取
│   ├── variant_score.py    # 变异打分（从 src/zero_shot_score.py 提取）
│   ├── lora_predict.py     # LoRA 功能预测（从 src/lora_fine_tune.py 提取）
│   └── masked_predict.py   # 掩码位置预测
└── docs/
    └── secondary-development.md  # 本文档
```

**原则：原始仓库 `src/`、`pretrain/` 等目录不做任何修改。** 需要复用的逻辑从原始代码中提取到 `modules/` 下，按功能分类组织，通过 import 使用。

```
用户输入（fasta/gff/vcf）
        │
        ▼
    Evo 2 生成序列
        │
        ▼
┌───────────────────────────────────────┐
│     app/main.py — FastAPI 服务         │
│                                       │
│  /embeddings    → modules/embedding    │
│  /variant-score → modules/variant_score│
│  /predict       → modules/lora_predict │
│  /masked-predict→ modules/masked_predict│
│                                       │
│  统一由 modules/engine.py 调度         │
└───────────────────────────────────────┘
        │
        ▼
    返回 JSON 结果
```

## API 接口设计

### 1. 序列嵌入提取 `POST /embeddings`

接收 DNA 序列，返回每个碱基位置的向量表示。

**请求：**
```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAATAACGCGCGA...",
  "normalize": true
}
```

**响应：**
```json
{
  "embeddings": [[0.12, -0.34, ...], ...],
  "shape": [8192, 1536],
  "sequence_length": 8192
}
```

**实现要点：**
- 输入序列最长 8192bp，超过报错
- 模型输出维度 3072（正向 1536 + 反向互补 1536），需拆分取均值得到 1536 维
- 可选 L2 归一化

**复用代码：** `notebooks/examples.ipynb` 中的 embedding 提取逻辑，需封装

### 2. 变异效应打分 `POST /variant-score`

接收参考序列和变异信息，返回 LLR 打分。

**请求：**
```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAATAACGCGCGA...",
  "position": 255,
  "ref_allele": "A",
  "alt_alleles": ["G", "C", "T"]
}
```

**响应：**
```json
{
  "scores": {
    "G": -2.34,
    "C": -4.56,
    "T": -3.21
  },
  "ref_prob": 0.97,
  "alt_probs": {"G": 0.011, "C": 0.008, "T": 0.011}
}
```

**实现要点：**
- mask 指定位置，推理获取 ACGT 概率
- 计算 `log(P_alt / P_ref)` 作为 LLR 分数
- 正值表示变异可能有害，负值表示中性/有利
- 支持一次评估多个 alt allele

**复用代码：** `src/zero_shot_score.py` 中的 `extract_logits` 和 `zero_shot_score` 函数逻辑

### 3. 功能预测 `POST /predict`

接收 DNA 序列和任务类型，返回 LoRA 微调模型的预测结果。

**请求：**
```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAATAACGCGCGA...",
  "task": "acr_arabidopsis"
}
```

**响应（分类任务）：**
```json
{
  "task": "acr_arabidopsis",
  "prediction": "POSITIVE",
  "probability": 0.87
}
```

**响应（回归任务）：**
```json
{
  "task": "expression_absolute",
  "prediction": 3.45
}
```

**支持的 task 值与对应 LoRA 模型：**

| task 参数 | LoRA 目录 | 任务类型 |
|-----------|----------|---------|
| `acr_arabidopsis` | `cross_species_acr_train_on_arabidopsis_plantcad2_large` | 分类 |
| `acr_nine_species` | `cross_species_acr_train_on_nine_species_plantcad2_large` | 分类 |
| `acr_cell_type` | `cell_type_specific_acr_plantcad2_large` | 分类 |
| `expression_on_off` | `cross_species_leaf_on_off_expression_plantcad2_large` | 分类 |
| `expression_absolute` | `cross_species_leaf_absolute_expression_plantcad2_large` | 回归 |
| `translation_on_off` | `cross_species_leaf_on_off_translation_plantcad2_large` | 分类 |
| `translation_absolute` | `cross_species_leaf_absolute_translation_plantcad2_large` | 回归 |

**实现要点：**
- 基座模型启动时加载一次，LoRA adapter 按需叠加（或预加载缓存）
- 分类任务输出 softmax 概率，回归任务输出原始预测值
- 输入序列长度由 LoRA 训练时的 sequence_length 决定（ACR 模型为 600bp）

**复用代码：** `src/lora_fine_tune.py` 中的 `load_base_model`、`predict` 函数逻辑

### 4. 掩码位置预测 `POST /masked-predict`

接收 DNA 序列，返回指定位置（或所有位置）的碱基概率分布。

**请求：**
```json
{
  "sequence": "CTTAATTAATATTGCCTTTGTAATAACGCGCGA...",
  "positions": [100, 200, 300]
}
```

**响应：**
```json
{
  "predictions": {
    "100": {"A": 0.95, "C": 0.02, "G": 0.02, "T": 0.01},
    "200": {"A": 0.10, "C": 0.80, "G": 0.05, "T": 0.05},
    "300": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
  }
}
```

**实现要点：**
- 逐位置 mask 并推理，获取 ACGT 概率
- 可用于评估序列的不确定性/关键位点

**复用代码：** `notebooks/examples.ipynb` 中的 masked token prediction 逻辑

## 需要开发的文件

### `app/` — FastAPI 服务层

```
app/
├── __init__.py
├── main.py                 # FastAPI 入口，启动服务、挂载路由
├── config.py               # 配置：模型路径、设备、LoRA 映射表
├── schemas/
│   ├── __init__.py
│   └── requests.py         # Pydantic 请求/响应模型定义
└── routers/
    ├── __init__.py
    ├── embeddings.py       # POST /embeddings 路由
    ├── variant.py          # POST /variant-score 路由
    ├── predict.py          # POST /predict 路由
    └── masked.py           # POST /masked-predict 路由
```

**`app/main.py` 示例：**

```python
from fastapi import FastAPI
from app.routers import embeddings, variant, predict, masked
from app.config import settings
from modules.engine import PlantCAD2Engine

app = FastAPI(title="PlantCAD2 Inference Service")
engine: PlantCAD2Engine = None

@app.on_event("startup")
async def startup():
    global engine
    engine = PlantCAD2Engine(
        base_model_path=settings.BASE_MODEL_PATH,
        lora_models_path=settings.LORA_MODELS_PATH,
        device=settings.DEVICE,
    )

app.include_router(embeddings.router, prefix="/embeddings")
app.include_router(variant.router, prefix="/variant-score")
app.include_router(predict.router, prefix="/predict")
app.include_router(masked.router, prefix="/masked-predict")
```

**`app/config.py` 示例：**

```python
class Settings:
    BASE_MODEL_PATH: str = "models/PlantCAD2-Large-l48-d1536"
    LORA_MODELS_PATH: str = "models"
    DEVICE: str = "cuda:0"

    TASK_LORA_MAP: dict = {
        "acr_arabidopsis": "cross_species_acr_train_on_arabidopsis_plantcad2_large",
        "acr_nine_species": "cross_species_acr_train_on_nine_species_plantcad2_large",
        "acr_cell_type": "cell_type_specific_acr_plantcad2_large",
        "expression_on_off": "cross_species_leaf_on_off_expression_plantcad2_large",
        "expression_absolute": "cross_species_leaf_absolute_expression_plantcad2_large",
        "translation_on_off": "cross_species_leaf_on_off_translation_plantcad2_large",
        "translation_absolute": "cross_species_leaf_absolute_translation_plantcad2_large",
    }

settings = Settings()
```

**`app/routers/` 每个路由文件的职责：** 定义请求/响应 schema、参数校验、调用 `modules/` 中的引擎函数、返回 JSON。路由层不做任何模型推理逻辑。

### `modules/` — 推理业务逻辑层

```
modules/
├── __init__.py
├── engine.py               # PlantCAD2Engine 主类，统一调度各模块
├── model_loader.py         # 模型加载（从 src/ 提取，适配本地路径）
├── embedding.py            # 序列嵌入提取
├── variant_score.py        # 变异打分（从 src/zero_shot_score.py 提取）
├── lora_predict.py         # LoRA 功能预测（从 src/lora_fine_tune.py 提取）
└── masked_predict.py       # 掩码位置预测
```

**`modules/engine.py` — 推理引擎主类：**

```python
class PlantCAD2Engine:
    """PlantCAD2 推理引擎，管理模型加载和推理调度。"""

    def __init__(self, base_model_path: str, lora_models_path: str, device: str):
        """启动时加载基座模型、tokenizer、预加载 LoRA adapter。"""
        self.mlm_model, self.tokenizer = load_mlm_model(base_model_path, device)
        self.cls_model = load_cls_model(base_model_path, device)
        self.lora_models = preload_lora_adapters(self.cls_model, lora_models_path)
        self.device = device

    def get_embeddings(self, sequence: str, normalize: bool = True) -> dict:
        """调用 modules/embedding.py"""
        pass

    def score_variant(self, sequence: str, position: int,
                      ref_allele: str, alt_alleles: list) -> dict:
        """调用 modules/variant_score.py"""
        pass

    def predict_function(self, sequence: str, task: str) -> dict:
        """调用 modules/lora_predict.py"""
        pass

    def masked_predict(self, sequence: str, positions: list) -> dict:
        """调用 modules/masked_predict.py"""
        pass
```

**`modules/model_loader.py`：**
从 `src/zero_shot_score.py` 的 `load_model_and_tokenizer` 和 `src/lora_fine_tune.py` 的 `load_base_model` 提取，适配本地路径加载。不修改原始代码，复制核心逻辑后做以下改动：
- 支持传入本地模型路径
- LoRA 加载时显式指定基座路径（不依赖 adapter_config.json 中的远程 ID）
- 合并 MLM 和 Classification 两种 head 的加载逻辑

**`modules/variant_score.py`：**
从 `src/zero_shot_score.py` 提取以下函数的核心逻辑：
- `SequenceDataset` → 简化为单条序列处理
- `extract_logits` → 去掉 DataLoader，直接单条推理
- `zero_shot_score` → 改为接受序列+位置+alleles 参数

**`modules/lora_predict.py`：**
从 `src/lora_fine_tune.py` 提取以下逻辑：
- `load_base_model` → 已在 model_loader.py 中处理
- `predict` → 从"读 parquet 文件"改为"接受序列字符串"，去掉 Trainer，直接 forward

**`modules/embedding.py`：**
从 `notebooks/examples.ipynb` 封装，核心逻辑：
- tokenize 序列 → model forward(output_hidden_states=True) → 取最后一层 hidden_states → 拆分正向/反向互补 → 取均值

**`modules/masked_predict.py`：**
从 `notebooks/examples.ipynb` 封装，核心逻辑：
- 对指定位置 mask → model forward → 取 logits → softmax → ACGT 概率

## 关键技术注意事项

### 0. 目录职责划分

| 目录 | 职责 | 是否修改原始代码 |
|------|------|----------------|
| `src/` | 原始仓库推理/训练脚本 | **否，只读** |
| `pretrain/` | 原始仓库预训练框架 | **否，只读** |
| `models/` | 模型权重文件 | 否 |
| `modules/` | 从 `src/` 提取的推理逻辑，按功能分类 | 是（适配后的副本） |
| `app/` | FastAPI 服务层，路由+配置+schema | 是（全新开发） |

`modules/` 中的代码从 `src/` 复制而来，做以下适配：
- 函数化：从脚本的 `main()` 流程改为可调用的函数
- 参数化：硬编码路径改为函数参数
- 单条化：批量 DataLoader 处理改为单条序列直接推理
- 不添加任何对 `src/` 的 import，完全自包含

### 1. LoRA adapter_config.json 路径修复

当前 `models/cross_species_acr_train_on_arabidopsis_plantcad2_large/adapter_config.json` 中：
```json
"base_model_name_or_path": "kuleshov-group/PlantCAD2-Large-l48-d1536"
```
这是 HuggingFace 远程 ID。如果服务器无网络或需离线运行，加载时必须显式传入本地基座路径，不能依赖 config 中的远程 ID。代码中应这样处理：

```python
from peft import PeftConfig, PeftModel

# 方式一：显式加载基座再叠加 LoRA（推荐，离线可用）
base_model = AutoModelForSequenceClassification.from_pretrained(
    "models/PlantCAD2-Large-l48-d1536", trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, lora_local_path)

# 方式二：修改 adapter_config.json 的 base_model_name_or_path 为本地路径
```

### 2. 双 head 架构

PlantCAD2 的 backbone 可以挂两种 head：
- `AutoModelForMaskedLM` → LM head，用于 embedding 提取、掩码预测、变异打分
- `AutoModelForSequenceClassification` → classification head，用于 LoRA 微调任务

两个 head 共享 backbone 参数。如果同时需要两种能力，服务启动时需加载两份模型（各带不同 head），backbone 权重会被 GPU 缓存复用，显存增量很小。

### 3. 显存估算

| 组件 | 显存占用 |
|------|---------|
| 基座模型（float16） | ~3.5 GB |
| 基座模型（bfloat16） | ~3.5 GB |
| LoRA adapter × 7（单个 ~38MB） | ~0.3 GB |
| 推理中间状态（seq_len=8192） | ~1-2 GB |
| **总计** | **~5-7 GB** |

单条序列推理不需要很大显存。推荐使用 NVIDIA RTX 4090（24GB）或 A100（40/80GB）。

### 4. 输入序列长度约束

| 任务 | 最大长度 | 超长处理 |
|------|---------|---------|
| Embedding | 8192bp | 报错 |
| 变异打分 | 8192bp | 报错 |
| LoRA-ACR 预测 | 600bp | 截断或报错 |
| LoRA-表达/翻译 | 由训练时决定 | 截断或报错 |
| 掩码预测 | 8192bp | 报错 |

### 5. 与 Evo 2 对接

Evo 2 生成的序列通过 HTTP 请求传入 PlantCAD2 服务。智能体的调用流程：

```
1. 智能体调用 Evo 2 API → 获得生成序列
2. 智能体调用 PlantCAD2 /predict → 获得功能预测
3. 智能体调用 PlantCAD2 /variant-score → 评估关键位点
4. 智能体汇总结果返回用户
```

两个服务之间无直接通信，由智能体编排调用顺序。

## 开发计划

| 阶段 | 内容 | 产出文件 | 依赖 |
|------|------|---------|------|
| P0 | 模型加载 + 推理引擎 | `modules/model_loader.py` + `modules/engine.py` | 服务器 GPU 环境 |
| P1 | 4 个推理模块 | `modules/embedding.py` `variant_score.py` `lora_predict.py` `masked_predict.py` | P0 |
| P2 | FastAPI 服务 + 路由 | `app/main.py` `config.py` `schemas/` `routers/` | P1 |
| P3 | 测试：用 examples/ 中的示例数据验证各端点输出 | - | P2 |
| P4 | 与 Evo 2 联调：端到端流程测试 | - | P3 |

## 可复用的现有代码

**原则：不修改原始仓库代码。** 将需要的逻辑从 `src/` 复制到 `modules/` 下，进行适配后使用。

| 原始代码 | 提取到 | 适配改动 |
|---------|--------|---------|
| `src/zero_shot_score.py` → `load_model_and_tokenizer` | `modules/model_loader.py` | 支持本地路径、合并 MLM/CLS 加载 |
| `src/zero_shot_score.py` → `SequenceDataset` + `extract_logits` | `modules/variant_score.py` | 去掉 DataLoader，改为单条序列直接推理 |
| `src/zero_shot_score.py` → `zero_shot_score` | `modules/variant_score.py` | 改为接受序列+位置+alleles 参数 |
| `src/lora_fine_tune.py` → `load_base_model` | `modules/model_loader.py` | 支持本地路径，与 MLM 加载合并 |
| `src/lora_fine_tune.py` → `predict` | `modules/lora_predict.py` | 去掉 parquet 文件读取和 Trainer，直接 forward |
| `src/zero_shot_score.py` → `get_optimal_dtype` | `modules/model_loader.py` | 直接复制 |
| `notebooks/examples.ipynb` → embedding 提取 | `modules/embedding.py` | 封装为函数，加入反向互补拆分逻辑 |
| `notebooks/examples.ipynb` → masked prediction | `modules/masked_predict.py` | 封装为函数，支持多位置批量 mask |
