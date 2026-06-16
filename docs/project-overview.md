# PlantCaduceus 项目全景文档

## 项目简介

PlantCaduceus（简称 PlantCAD）是一个面向植物基因组的 **DNA 预训练语言模型**项目。模型基于 [Caduceus](https://arxiv.org/abs/2403.03234) 架构（改进自 [Mamba](https://arxiv.org/abs/2312.00752) 线性序列建模框架），引入双向性和反向互补等变性，专为 DNA 序列设计。项目包含两代模型：

- **PlantCAD (v1)**：在 16 个被子植物基因组上预训练，最大输入 512bp，参数规模 20M-225M
- **PlantCAD2**：扩展至更长上下文（8192bp）、更大参数（88M-694M），覆盖更多物种

核心能力：零样本变异效应预测、序列嵌入提取、掩码语言模型预测、下游任务微调。

## 目录结构

```
PlantCaduceus/
├── src/                    # 核心推理与训练脚本
├── pretrain/               # 预训练框架与工具
│   ├── scripts/            # 预训练入口脚本
│   └── llmlib/             # 模型架构、数据加载、分词器
├── pipelines/              # 生物信息学分析管道
├── classifiers/            # 预训练 XGBoost 分类器权重
├── notebooks/              # Jupyter 示例
├── examples/               # 示例数据（VCF、TSV）
├── docker/                 # Docker 构建配置
├── docs/                   # 文档
├── models/                 # 本地模型存储（可选）
└── requirements.txt        # Python 依赖
```

## 模型家族

### PlantCAD (v1)

| 模型                | 参数量  | Embedding 维度 | 最大输入  | HuggingFace                                                   |
| ----------------- | ---- | ------------ | ----- | ------------------------------------------------------------- |
| PlantCaduceus_l20 | 20M  | 384          | 512bp | [链接](https://huggingface.co/kuleshov-group/PlantCaduceus_l20) |
| PlantCaduceus_l24 | 40M  | 512          | 512bp | [链接](https://huggingface.co/kuleshov-group/PlantCaduceus_l24) |
| PlantCaduceus_l28 | 128M | 768          | 512bp | [链接](https://huggingface.co/kuleshov-group/PlantCaduceus_l28) |
| PlantCaduceus_l32 | 225M | 1024         | 512bp | [链接](https://huggingface.co/kuleshov-group/PlantCaduceus_l32) |

### PlantCAD2

| 模型               | 参数量  | Embedding 维度 | 最大输入   | HuggingFace                                                            |
| ---------------- | ---- | ------------ | ------ | ---------------------------------------------------------------------- |
| PlantCAD2-Small  | 88M  | 768          | 8192bp | [链接](https://huggingface.co/kuleshov-group/PlantCAD2-Small-l24-d0768)  |
| PlantCAD2-Medium | 311M | 1024         | 8192bp | [链接](https://huggingface.co/kuleshov-group/PlantCAD2-Medium-l48-d1024) |
| PlantCAD2-Large  | 694M | 1536         | 8192bp | [链接](https://huggingface.co/kuleshov-group/PlantCAD2-Large-l48-d1536)  |

### LoRA 微调模型（PlantCAD2 专用）

针对关键下游任务发布了 [LoRA 微调模型集合](https://huggingface.co/collections/plantcad/fine-tuned-plantcad2-models-68b316a57616134fa7a1b6b6)，涵盖：

- 染色质可及区（ACR）预测：跨物种 / 细胞类型特异性
- 基因表达预测：开/关分类 / 绝对表达量回归
- 蛋白质翻译预测：开/关分类 / 绝对翻译丰度回归

每个任务均有 Small/Medium/Large 三个版本。

### 预训练 XGBoost 分类器

`classifiers/` 目录下为四个模型版本（l20/l24/l28/l32）各提供四个分类器：

| 分类器                   | 功能                                     |
| --------------------- | -------------------------------------- |
| TIS_XGBoost.json      | 翻译起始位点（Translation Initiation Site）识别  |
| TTS_XGBoost.json      | 翻译终止位点（Translation Termination Site）识别 |
| Donor_XGBoost.json    | 剪接供体位点识别                               |
| Acceptor_XGBoost.json | 剪接受体位点识别                               |

## 核心模块详解

### src/ -- 推理与训练脚本

#### zero_shot_score.py -- SNV 零样本变异打分

两种互斥工作模式：

- **VCF 模式**：读取 VCF 中的 SNV 位点，从参考基因组 FASTA 提取以变异位点为中心的上下文窗口序列，mask 变异位点后通过模型推理获取 A/C/G/T 概率，计算 `log(P_alt / P_ref)` 作为零样本分数，注释回 VCF 的 INFO 字段。
- **BED 模式**：按 BED 文件定义的基因组区域滑窗扫描，对每个窗口 mask 中心位置后推理，得到每个碱基位置的 LLR 分数，支持 max/average/all 聚合输出。

模型自动适配：检测模型名含 `"plantcad2"` 时强制 contextSize >= 2048，含 `"plantcaduceus"` 时锁定 512。

```bash
# VCF 模式示例
python src/zero_shot_score.py \
  -model kuleshov-group/PlantCAD2-Large-l48-d1536 \
  -contextSize 8192 \
  -input-vcf input.vcf \
  -input-fasta ref.fa \
  -output scored.vcf \
  -device cuda:0

# BED 模式示例
python src/zero_shot_score.py \
  -model kuleshov-group/PlantCAD2-Large-l48-d1536 \
  -contextSize 8192 \
  -input-bed regions.bed \
  -input-fasta ref.fa \
  -output scores.tsv \
  -step-size 8 \
  -device cuda:0
```

#### zero_shot_score_sv.py -- 结构变异打分

处理 VCF 中的 DEL/INS 类型结构变异。对每个 SV 构建参考序列和突变序列两条序列，分别推理获取连接点侧翼的核苷酸概率，计算 `log(P_mut / P_ref)` 平均值作为破坏性评分。默认上下文 8192bp。

#### lora_fine_tune.py -- LoRA 微调全流程

通过 `fire` 库暴露 5 个子命令：

| 子命令        | 功能                                           |
| ---------- | -------------------------------------------- |
| `tokenize` | 将本地 TSV 或 HuggingFace 数据集 tokenize 为 parquet |
| `train`    | 在基座模型上附加 LoRA adapter 进行微调训练                 |
| `evaluate` | 加载 LoRA checkpoint 评估验证集指标                   |
| `predict`  | 加载微调模型生成预测结果（CSV）                            |
| `display`  | 打印 LoRA 模型结构和可训练参数统计                         |

LoRA 配置：r=8, alpha=32, target_modules=x_proj/in_proj/out_proj。支持 classification / regression / multi_label 三种任务类型。

```bash
# tokenize
python src/lora_fine_tune.py tokenize \
  --hf_dataset "plantcad/PlantCAD2_fine_tuning_tasks" \
  --hf_config "cross_species_acr_train_on_arabidopsis" \
  --hf_split "train" \
  --model_name "kuleshov-group/PlantCAD2-Small-l24-d0768" \
  --sequence_length 600 \
  -output_path "train.parquet"

# train
python src/lora_fine_tune.py train \
  --model_name "kuleshov-group/PlantCAD2-Small-l24-d0768" \
  --train_dir train.parquet \
  --valid_dir valid.parquet \
  --output_dir ./output \
  --task_type "classification" \
  --learning_rate 1e-4 \
  --num_train_epochs 1

# predict
python src/lora_fine_tune.py predict \
  --checkpoint_dir "plantcad/cross_species_acr_train_on_arabidopsis_plantcad2_small" \
  --data_dir test.parquet \
  --task_type "classification"
```

#### train_XGBoost.py / predict_XGBoost.py -- XGBoost 分类器

- **训练**：加载预训练 LM，提取序列中心 token 的最后一层隐藏状态，双向 embedding 取均值作为特征，训练 XGBoost（1000 棵树，max_depth=6）。输出 ROC/PR 曲线和 AUC。
- **预测**：加载已训练的 XGBoost 模型对新数据推理，输出 TSV。支持分块推理节省内存。

`predict_XGBoost.py` 直接 import `train_XGBoost` 的全部函数，共享模型加载和 embedding 提取逻辑。

#### HF_pre_train.py -- 掩码语言模型预训练

基于 HuggingFace Trainer 的标准 MLM 预训本。特色功能：

- 自定义 `DataCollatorForLanguageModelingSimplified` 跳过等长序列 padding
- 支持 soft-masked loss weights（小写字母标记低置信度碱基，权重 0.1）
- 三组参数管理：ModelArguments、DataTrainingArguments、TrainingArguments

#### zero-shot-eval.py -- 零样本评估工具集

PlantCAD2 专用评估接口，4 个子命令：

| 子命令            | 评估任务     | 指标                              |
| -------------- | -------- | ------------------------------- |
| `evo_cons`     | 进化保守性    | AUROC / AUPRC                   |
| `motif_acc`    | 基序恢复     | token accuracy / motif accuracy |
| `sv_effect`    | 结构变异效应   | AUPRC                           |
| `core_noncore` | 核心/非核心分类 | AUROC / AUPRC                   |

### pretrain/ -- 预训练框架

#### 框架选择

项目使用两套训练框架并行：

1. **MosaicML Composer**（主要）：`train_mosaic_bert.py` 和 `train_mosaic_bert_hf_data.py`，配合 DecoupledAdamW、线性/余弦 warmup、FSDP、AMP bf16
2. **HuggingFace Trainer**（辅助）：`run_mlm_plants.py`、`train_plant_BERT.py` 等

#### 模型架构（llmlib/architectures/）

| 架构         | 文件                           | 说明                        |
| ---------- | ---------------------------- | ------------------------- |
| MosaicBERT | `models/bert/mosaic_bert.py` | 带 ALiBi + GLU 优化的 BERT    |
| Caduceus   | `models/mamba/caduceus.py`   | Mamba SSM DNA 模型，支持反向互补映射 |
| GPN        | `models/conv/gpn.py`         | 膨胀 1D 卷积 + OneHot 嵌入      |

#### 数据处理

1. **原始数据**：DNA 序列（assembly/chrom/start/end/strand/seq），HuggingFace Dataset 或 TSV 格式
2. **Tokenization**：字符级（A/C/G/T/N 映射为整数 ID），小写字母用于 soft masking
3. **格式转换**：`convert_bert_dataset_to_mds_streaming.py` 将 HF Dataset 转为 MosaicML MDS 二进制流式格式
4. **MLM 掩码**：标准 15% 掩码概率，可配置为 30%

#### 分词器（llmlib/tokenization/）

- `hg38_char_tokenizer.py`：字符级 DNA tokenizer，继承 `PreTrainedTokenizer`
- `hg38_char_tokenizer_mlm.py`：基于 `BertTokenizerFast` 的 MLM 版本

### pipelines/ -- 生物信息学分析管道

#### In-silico Mutagenesis（`pipelines/in-silico-mutagenesis/`）

四步流程，用于大规模模拟变异并评估模型区分能力：

| 步骤     | 脚本                   | 功能                              |
| ------ | -------------------- | ------------------------------- |
| Step 1 | `1_simulation.R`     | 基于 GFF + FASTA 在基因区域生成所有可能的 SNP |
| Step 2 | VEP (Singularity)    | 对变异进行功能注释（错义/同义/内含子等）           |
| Step 3 | `2_down_sampling.py` | 下采样：基因间 20 万条，其他类型各 10 万条       |
| Step 4 | `zero_shot_score.py` | 零样本评分                           |

### docker/ -- 容器化部署

提供两个 Docker 配置：

| 版本                             | PyTorch | CUDA | 说明                       |
| ------------------------------ | ------- | ---- | ------------------------ |
| python3.11-torch2.5.1-cuda12.4 | 2.5.1   | 12.4 | pip 安装 mamba-ssm         |
| python3.11-torch2.7.1-cuda12.8 | 2.7.1   | 12.8 | 源码编译 mamba/causal-conv1d |

预构建镜像：`ghcr.io/plantcad/plantcad:v1.1.0`

### examples/ -- 示例数据

| 文件                      | 说明                       |
| ----------------------- | ------------------------ |
| `example_maize_snp.vcf` | 玉米 SNP 数据，12 个样本，含多种变异类型 |
| `example_snp.tsv`       | 带 512bp 序列上下文的 SNP 表格    |

## 推理性能基准

| 模型                | Seq Len | Batch | 显存 (GB) | Seq/s | Tokens/s |
| ----------------- | ------- | ----- | ------- | ----- | -------- |
| PlantCaduceus_l20 | 512     | 64    | 1.01    | 663   | 339,475  |
| PlantCaduceus_l32 | 512     | 64    | 2.97    | 135   | 69,144   |
| PlantCAD2-Small   | 8192    | 64    | 24.89   | 19    | 155,649  |
| PlantCAD2-Medium  | 8192    | 64    | 33.62   | 6.79  | 55,636   |
| PlantCAD2-Large   | 8192    | 64    | 51.09   | 3.89  | 31,833   |

## 数据流与模块依赖

```
                    ┌─────────────────┐
                    │  预训练数据集     │
                    │  (16 Angiosperm) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   HF_pre_train   │  或  pretrain/scripts/
                    │   (MLM 预训练)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  预训练模型权重    │
                    │  (HuggingFace)   │
                    └───┬────┬────┬───┘
                        │    │    │
           ┌────────────┘    │    └────────────┐
           │                 │                 │
  ┌────────▼───────┐ ┌──────▼──────┐ ┌────────▼───────┐
  │ zero_shot_score │ │ lora_fine_  │ │ train_XGBoost  │
  │ (SNV/SV 打分)   │ │ tune        │ │ (嵌入→分类器)   │
  └────────────────┘ │ (LoRA 微调)  │ └───────┬────────┘
                      └──────┬──────┘         │
                             │         ┌──────▼──────┐
                      ┌──────▼──────┐  │predict_XGBoost│
                      │  LoRA 模型   │  │ (TIS/TTS/     │
                      │  (染色质/表达) │  │  剪接位点)     │
                      └─────────────┘  └─────────────┘
```

## 关键设计决策

1. **反向互补等变性**：模型内部同时维护正向和反向互补序列表示，embedding 输出维度为实际维度的 2 倍，使用时需取均值
2. **上下文窗口自适应**：`zero_shot_score.py` 根据模型名自动调整 contextSize（PlantCAD v1 固定 512，PlantCAD2 最小 2048）
3. **Soft masking**：预训练中用小写字母标记低置信度碱基，赋予 0.1 损失权重，提升模型对基因组注释噪声的鲁棒性
4. **模块化微调**：通过 LoRA adapter 实现参数高效微调，仅训练 ~0.1% 参数即可适配下游任务
